//! The masked-diffusion objective, and the claim that it needs no new
//! Meganeura operator.

use anaphora::loss::{LabelError, MaskedDiffusionLoss};
use anaphora::schedule::NoiseLevel;
use anaphora::view::{CleanSequence, MaskToken, NoisedView};
use meganeura::Graph;

const MASK: MaskToken = MaskToken(0);
const VOCAB: usize = 16;
const SEQ: usize = 8;

fn t(v: f32) -> NoiseLevel {
    NoiseLevel::new(v).expect("in range")
}

fn clean() -> CleanSequence {
    CleanSequence::new((0..SEQ).map(|i| (i + 1) as u32).collect())
}

#[test]
fn only_masked_positions_score() {
    let clean = clean();
    let view = clean.mask_with(t(0.5), MASK, |i| i % 2 == 0);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    let stats = loss
        .build_labels(&view, &clean, &mut labels)
        .expect("valid");

    assert_eq!(stats.scored, 4);
    assert_eq!(stats.seq_len, SEQ);
    assert!((stats.weight - 2.0).abs() < 1e-6, "1/t at t=0.5 is 2");

    for pos in 0..SEQ {
        let row = &labels[pos * VOCAB..(pos + 1) * VOCAB];
        let sum: f32 = row.iter().sum();
        if pos % 2 == 0 {
            // Masked: exactly one weighted entry, at the true token.
            assert!((sum - 2.0).abs() < 1e-6, "row {pos} should carry 1/t");
            assert!((row[pos + 1] - 2.0).abs() < 1e-6, "wrong target at {pos}");
        } else {
            // Unmasked: the position's token is already visible in the input,
            // so scoring it would only teach the model to copy its input.
            assert_eq!(sum, 0.0, "row {pos} must not score");
        }
    }
}

#[test]
fn weight_is_one_over_t_with_a_floor() {
    let loss = MaskedDiffusionLoss::new(VOCAB);
    assert!((loss.weight(t(0.25)) - 4.0).abs() < 1e-6);
    assert!((loss.weight(t(1.0)) - 1.0).abs() < 1e-6);
    // The weight diverges as t goes to zero; the floor bounds it. A tiny t is
    // a legitimate draw from the schedule, not a caller error.
    assert!(loss.weight(NoiseLevel::CLEAN).is_finite());
    assert!((loss.weight(NoiseLevel::CLEAN) - 1000.0).abs() < 1e-3);
}

#[test]
fn targets_from_another_sequence_are_rejected() {
    // The mirror image of the retrieval leak: not the retriever seeing too
    // much, but the loss scoring against the wrong answers.
    let a = clean();
    let b = CleanSequence::new((0..SEQ).map(|i| (i + 2) as u32).collect());
    let view = a.mask_with(t(0.5), MASK, |_| true);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    assert!(matches!(
        loss.build_labels(&view, &b, &mut labels),
        Err(LabelError::SequenceMismatch { .. })
    ));
    assert!(
        labels.is_empty(),
        "a rejected call must not fill the tensor"
    );
}

#[test]
fn a_sampling_view_has_no_targets() {
    // Sampling starts from a canvas, not from a held-out answer.
    let view = NoisedView::all_masked(SEQ, MASK);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    assert!(matches!(
        loss.build_labels(&view, &clean(), &mut labels),
        Err(LabelError::SequenceMismatch {
            view_source: None,
            ..
        })
    ));
}

#[test]
fn out_of_vocabulary_targets_are_rejected() {
    let clean = CleanSequence::new(vec![99; SEQ]);
    let view = clean.mask_with(t(0.5), MASK, |_| true);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    assert!(matches!(
        loss.build_labels(&view, &clean, &mut labels),
        Err(LabelError::TokenOutOfRange { .. })
    ));
}

#[test]
fn a_step_that_masks_nothing_contributes_nothing() {
    let clean = clean();
    let view = clean.mask_with(t(0.5), MASK, |_| false);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    let stats = loss
        .build_labels(&view, &clean, &mut labels)
        .expect("valid");
    assert_eq!(stats.scored, 0);
    assert!(!stats.contributes(), "the caller should skip this step");
    assert!(labels.iter().all(|&w| w == 0.0));
}

#[test]
fn dense_label_cost_is_reported() {
    // The operator takes dense [n, vocab] labels, so the tensor carries at
    // most `n` non-zero values in `n * vocab` floats. At LLaDA's vocabulary
    // that is worth knowing before a run discovers it.
    let llada = MaskedDiffusionLoss::new(126_464);
    assert_eq!(llada.label_bytes(512), 512 * 126_464 * 4);
    assert!(llada.label_bytes(2048) > 1_000_000_000, "over a gigabyte");
}

/// Build `logits = identity @ w`, so `w`'s gradient *is* the logits gradient.
fn logits_from_identity(g: &mut Graph) -> meganeura::NodeId {
    let x = g.input("identity", &[SEQ, SEQ]);
    let w = g.parameter("w", &[SEQ, VOCAB]);
    g.matmul(x, w)
}

fn identity_matrix() -> Vec<f32> {
    let mut m = vec![0.0f32; SEQ * SEQ];
    for i in 0..SEQ {
        m[i * SEQ + i] = 1.0;
    }
    m
}

fn probe_logits() -> Vec<f32> {
    (0..SEQ * VOCAB)
        .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3)
        .collect()
}

#[test]
fn gpu_loss_matches_the_cpu_reference() {
    // The "no new operator" claim: Meganeura's generalized cross-entropy,
    // fed weighted one-hot rows and zero rows, computes the LLaDA objective.
    let clean = clean();
    let view = clean.mask_with(t(0.4), MASK, |i| i % 3 == 0);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    let stats = loss
        .build_labels(&view, &clean, &mut labels)
        .expect("valid");
    assert!(stats.contributes());

    let mut g = Graph::new();
    let logits = logits_from_identity(&mut g);
    let label_input = g.input("labels", &[SEQ, VOCAB]);
    let node = g.cross_entropy_loss(logits, label_input);
    g.set_outputs(vec![node]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let probe = probe_logits();
    session.set_input("identity", &identity_matrix());
    session.set_parameter("w", &probe);
    session.set_input("labels", &labels);
    session.step();
    session.wait();

    let gpu = session.read_loss();
    let cpu = loss.reference_loss(&probe, &labels, SEQ);
    assert!(
        (gpu - cpu).abs() < 1e-4,
        "GPU loss {gpu} disagrees with the reference {cpu}"
    );
}

#[test]
fn unmasked_positions_receive_exactly_zero_gradient() {
    // This is what makes the zero-row trick sound rather than merely
    // convenient. The kernel's gradient is `softmax·S − labels` with
    // `S = Σ labels`; a zero row has `S = 0`, so the whole row is zero — not
    // small, zero. If it were `softmax − labels` with an assumed `S = 1`,
    // every unmasked position would push its logits toward uniform, and the
    // model would train against a target nobody wrote.
    let clean = clean();
    let view = clean.mask_with(t(0.4), MASK, |i| i % 3 == 0);
    let loss = MaskedDiffusionLoss::new(VOCAB);
    let mut labels = Vec::new();
    loss.build_labels(&view, &clean, &mut labels)
        .expect("valid");

    let mut g = Graph::new();
    let logits = logits_from_identity(&mut g);
    let label_input = g.input("labels", &[SEQ, VOCAB]);
    let node = g.cross_entropy_loss(logits, label_input);
    g.set_outputs(vec![node]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    session.set_input("identity", &identity_matrix());
    session.set_parameter("w", &probe_logits());
    session.set_input("labels", &labels);
    session.step();
    session.wait();

    let mut grad = vec![0.0f32; SEQ * VOCAB];
    session.read_param_grad("w", &mut grad);

    for pos in 0..SEQ {
        let row = &grad[pos * VOCAB..(pos + 1) * VOCAB];
        let magnitude: f32 = row.iter().map(|v| v.abs()).sum();
        if view.masked()[pos] {
            assert!(magnitude > 0.0, "masked position {pos} must score");
        } else {
            assert_eq!(magnitude, 0.0, "unmasked position {pos} must not score");
        }
    }
}

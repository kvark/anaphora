//! Phase 1's exit criterion: calibrate the evaluation protocol against a run
//! that is *known* to be leaking.
//!
//! The structural guards in `view` and `retrieval::leakage` prevent the leaks
//! we know how to name. They cannot prove absence, and this project's
//! characteristic failure is silent and flattering — a leaked query improves
//! perplexity. So before any number from a real run is trusted, the protocol
//! that would have to catch such a run is checked against one.
//!
//! Two arms, identical but for where retrieval queries come from:
//!
//! * **honest** — queries built from the noised view, as `chunk_queries`
//!   enforces;
//! * **leaked** — queries built from the clean sequence, via the
//!   `leak-harness` feature. This is the bug the whole `view` module exists
//!   to make unrepresentable, reproduced deliberately.
//!
//! The corpus is arranged so the leak has something to find: every training
//! document has a verbatim copy in the index under a different id, and only
//! the provenance guard is active. A clean query therefore retrieves a
//! passage whose continuation *is* the masked answer; a masked query has to
//! work from what is still visible.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --features leak-harness --example leak_calibration
//! ```
//!
//! Exits non-zero if the protocol fails to flag the leaked arm — that is the
//! criterion, and it is meant to gate the phase.

use anaphora::config::CcaConfig;
use anaphora::corpus::{Document, HashedBagEmbedder, build_corpus, training_sequences};
use anaphora::eval::{EvalReport, Evaluator, NeighbourCondition};
use anaphora::loss::MaskedDiffusionLoss;
use anaphora::model::backbone::{Backbone, BackboneConfig, Freezing};
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::corpus::DocumentId;
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::schedule::NoiseLevel;
use anaphora::train::{
    NoiseSampler, Optimizer, QuerySource, RetrievalSources, Rng, Trainer, TrainingConfig,
    apply_zero_init, configure_optimizer,
};
use anaphora::view::{CleanSequence, MaskToken};
use meganeura::Graph;

const MASK: MaskToken = MaskToken(0);
const PAD: u32 = 1;
const VOCAB: usize = 64;
const SEQ: usize = 16;
const DIM: usize = 64;
const TOPICS: u64 = 24;
const STEPS: usize = 400;

fn cca() -> CcaConfig {
    CcaConfig::new(SEQ, 4, 2, 8, 2, 2, 16, 2, 1).expect("valid shapes")
}

/// Distinct pseudo-random passages, so a match is unambiguous.
fn topic(i: u64) -> Vec<u32> {
    let mut rng = Rng::new(0x70D1C ^ i);
    (0..SEQ * 2)
        .map(|_| 2 + (rng.next_u64() % (VOCAB as u64 - 2)) as u32)
        .collect()
}

/// Topics the backbone is pretrained on. Disjoint from the retrofit's, so
/// retrieval still has something to contribute after pretraining.
fn background_documents() -> Vec<Document> {
    (TOPICS..TOPICS * 3)
        .map(|i| Document {
            id: DocumentId(i),
            tokens: topic(i),
        })
        .collect()
}

/// Topics the retrofit trains on.
fn training_documents() -> Vec<Document> {
    (0..TOPICS)
        .map(|i| Document {
            id: DocumentId(i),
            tokens: topic(i),
        })
        .collect()
}

/// The index: a verbatim copy of every training document under a different
/// id. This is the realistic setting — the training distribution is in the
/// index — and it is what gives a leaked query something to find. Only the
/// provenance guard runs, so the copies survive; enabling the offline n-gram
/// audit is what would remove them, and that is the point being demonstrated.
fn index_documents() -> Vec<Document> {
    (0..TOPICS)
        .map(|i| Document {
            id: DocumentId(1000 + i),
            tokens: topic(i),
        })
        .collect()
}

fn model_config(cca: CcaConfig) -> ModelConfig {
    ModelConfig {
        cca,
        backbone: BackboneConfig {
            vocab_size: VOCAB,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 48,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        },
        encoder: NeighbourEncoderConfig {
            vocab_size: VOCAB,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 48,
            scope: EncoderScope::PerNeighbour,
        },
        activation: GateActivation::Tanh,
        neighbours: NeighbourInput::Encoded,
    }
}

/// One pretrained parameter, ready to write into the retrofit graph.
struct PretrainedWeight {
    name: String,
    values: Vec<f32>,
}

/// Pretrain the backbone that both arms will then freeze.
///
/// A randomly initialised backbone cannot be calibrated against: its LM head
/// cannot express a specific token, so the retrofit has nothing to gain from
/// retrieval and nothing to leak either. Pretraining runs on *background*
/// topics, disjoint from the ones the retrofit trains on, so retrieval still
/// has something to contribute afterwards.
fn pretrain_backbone(cca: CcaConfig, steps: usize) -> Vec<PretrainedWeight> {
    let cfg = model_config(cca).backbone;
    let mut g = Graph::new();
    let backbone = Backbone::with_freezing(&mut g, "backbone", cfg, Freezing::Trainable);
    let token_ids = g.input_u32("token_ids", &[SEQ]);
    let mut x = backbone.embed(&mut g, token_ids);
    for layer in 0..cfg.num_layers {
        x = backbone.layer(&mut g, x, layer);
    }
    let logits = backbone.head(&mut g, x);
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    seed_parameters(&mut session, 0xBA5E);
    configure_optimizer(&mut session, Optimizer::adam(3e-3));

    let docs = background_documents();
    let seqs = training_sequences(&docs, SEQ, PAD, SEQ);
    let objective = MaskedDiffusionLoss::new(VOCAB);
    let mut rng = Rng::new(0xDA7A);
    let mut label_buf = Vec::new();
    let (mut first, mut last) = (Vec::new(), Vec::new());

    for i in 0..steps {
        let seq = &seqs[i % seqs.len()];
        let t = NoiseLevel::saturating(1.0 - rng.next_f32());
        let clean = CleanSequence::new(seq.tokens.clone());
        let draws: Vec<bool> = (0..seq.tokens.len())
            .map(|j| j < seq.content_len && rng.next_f32() < t.get())
            .collect();
        let view = clean.mask_with(t, MASK, |j| draws[j]);
        let stats = objective
            .build_labels(&view, &clean, &mut label_buf)
            .expect("masked from this sequence");
        if !stats.contributes() {
            continue;
        }
        session.set_input_u32("token_ids", view.tokens());
        session.set_input("labels", &label_buf);
        session.step();
        session.wait();
        let l = session.read_loss();
        if i < steps / 10 {
            first.push(l);
        }
        if i >= steps - steps / 10 {
            last.push(l);
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "[pretrain] backbone loss {:.4} -> {:.4} over {steps} steps",
        mean(&first),
        mean(&last)
    );

    Backbone::param_names("backbone", cfg)
        .into_iter()
        .map(|name| {
            let len = session.param_size(&name).expect("declared");
            let mut values = vec![0.0f32; len];
            session.read_param(&name, &mut values);
            PretrainedWeight { name, values }
        })
        .collect()
}

/// The frozen backbone must be the *same* frozen backbone in both arms, or
/// the comparison is between two different models.
fn seed_parameters(session: &mut meganeura::runtime::Session, seed: u64) {
    let names: Vec<String> = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        let len = session.param_size(&name).expect("declared");
        let mixed = name
            .bytes()
            .fold(0u64, |a, b| a.rotate_left(5) ^ u64::from(b));
        let mut rng = Rng::new(seed ^ mixed);
        let is_norm = name.contains("norm");
        let values: Vec<f32> = (0..len)
            .map(|_| {
                if is_norm {
                    1.0
                } else {
                    (rng.next_f32() - 0.5) * 0.3
                }
            })
            .collect();
        session.set_parameter(&name, &values);
    }
}

fn run_arm(label: &str, query_source: QuerySource, pretrained: &[PretrainedWeight]) -> EvalReport {
    let cca = cca();
    let mut embedder = HashedBagEmbedder::bigram(DIM, MASK.0);
    let corpus = build_corpus(&index_documents(), cca, &mut embedder);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();
    let seqs = training_sequences(&training_documents(), SEQ, PAD, SEQ);

    let mut g = Graph::new();
    let model = CcaModel::build(&mut g, model_config(cca));
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    seed_parameters(&mut session, 0xF1_1ED);
    // Both arms freeze the *same* pretrained backbone, or the comparison is
    // between two different models rather than two query sources.
    for weight in pretrained {
        session.set_parameter(&weight.name, &weight.values);
    }
    apply_zero_init(&mut session, &model);
    configure_optimizer(&mut session, Optimizer::adam(3e-3));

    let mut trainer = Trainer::new(
        TrainingConfig {
            noise: NoiseSampler::Uniform,
            query_source,
            ..TrainingConfig::new(cca, VOCAB, MASK)
        },
        0xCA11B,
    );
    let mut sources = RetrievalSources {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        embedder: &mut embedder,
    };

    let mut first = Vec::new();
    let mut last = Vec::new();
    for i in 0..STEPS {
        let seq = &seqs[i % seqs.len()];
        if let Some(report) = trainer
            .step(&mut session, &model, seq, &mut sources)
            .expect("well-formed step")
        {
            if i < STEPS / 10 {
                first.push(report.loss);
            }
            if i >= STEPS - STEPS / 10 {
                last.push(report.loss);
            }
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "[{label}] training loss {:.4} -> {:.4} over {} steps",
        mean(&first),
        mean(&last),
        trainer.steps()
    );

    let mut evaluator = Evaluator::new(cca, VOCAB, MASK, 99);
    let levels: Vec<NoiseLevel> = (1..10)
        .map(|i| NoiseLevel::new(i as f32 / 10.0).expect("in range"))
        .collect();
    let report = evaluator.run(
        &mut session,
        &seqs,
        &levels,
        &NeighbourCondition::ALL,
        &mut sources,
    );
    println!("[{label}]\n{}", report.to_table());
    report
}

fn main() {
    println!(
        "Phase 1 calibration: {TOPICS} documents, each with a verbatim copy in the index.\n\
         The protocol must flag the leaked arm. If it does not, it is not ready to \n\
         judge a real run.\n"
    );

    let pretrained = pretrain_backbone(cca(), 900);
    let honest = run_arm("honest", QuerySource::NoisedView, &pretrained);

    #[cfg(not(feature = "leak-harness"))]
    {
        let _ = (honest, pretrained);
        eprintln!(
            "The leaked arm needs the `leak-harness` feature:\n  \
             cargo run --release --features leak-harness --example leak_calibration"
        );
        std::process::exit(2);
    }

    #[cfg(feature = "leak-harness")]
    {
        let leaked = run_arm("leaked", QuerySource::CleanSequenceLeak, &pretrained);
        verdict(&honest, &leaked);
    }
}

/// Minimum separation before a signal counts as flagged.
///
/// Without it a bare `>` fires on floating-point noise: two runs whose
/// diagnostics agree to three decimal places would still "differ", and the
/// calibration would report a pass it has not earned.
#[cfg(feature = "leak-harness")]
const MARGIN: f32 = 0.02;

/// The retrieval path must be worth at least this much, relative to the
/// ablated baseline, before a verdict means anything.
///
/// A leak detector cannot be calibrated on a run that learned nothing to
/// leak. If the retrofit barely beats its own no-retrieval baseline, the
///difference between arms is measurement noise whichever way it points, and the
/// honest report is that the experiment did not produce a verdict.
#[cfg(feature = "leak-harness")]
const MIN_RELATIVE_GAIN: f32 = 0.05;

#[cfg(feature = "leak-harness")]
fn verdict(honest: &EvalReport, leaked: &EvalReport) {
    println!("--- verdict ---");

    // Precondition: did either arm learn enough for a leak to show?
    let relative_gain = |r: &EvalReport| -> Option<f32> {
        let ablated = r.get(NeighbourCondition::Ablated)?.mean_loss;
        let gain = r.retrieval_gain()?;
        (ablated.abs() > 1e-6).then(|| gain / ablated)
    };
    let best = [relative_gain(honest), relative_gain(leaked)]
        .into_iter()
        .flatten()
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "retrieval gain relative to ablated: {:.1}% (need {:.0}% for a verdict)",
        best * 100.0,
        MIN_RELATIVE_GAIN * 100.0
    );
    // Explicit rather than `!(best >= MIN)`: an empty fold leaves NEG_INFINITY,
    // and either way the inconclusive branch is the safe one to fall into.
    if !best.is_finite() || best < MIN_RELATIVE_GAIN {
        eprintln!(
            "\nINCONCLUSIVE: the retrofit barely beat its own no-retrieval baseline,\n\
             so there is not enough signal for a leak to show against. This is a\n\
             property of the run, not of the protocol -- calibrating a leak detector\n\
             on a model that learned nothing measures nothing.\n\n\
             Needed before this can return PASS or FAIL:\n\
             - a corpus with real predictive structure (Simple English Wikipedia,\n\
               per docs/v0-plan.md) instead of uniform-random synthetic topics;\n\
             - a backbone pretrained to convergence on it, not 900 steps;\n\
             - a retrofit trained long enough that retrieval is demonstrably worth\n\
               something before anyone asks whether it is worth too much."
        );
        std::process::exit(3);
    }

    let mut flagged = Vec::new();
    let mut missed = Vec::new();

    // 1. The copy gap should widen relative to what retrieval is worth.
    match (honest.copy_ratio(), leaked.copy_ratio()) {
        (Some(h), Some(l)) => {
            println!("copy ratio: honest {h:+.3}  leaked {l:+.3}");
            if l > h + MARGIN {
                flagged.push("copy ratio widened");
            } else {
                missed.push("copy ratio did not widen");
            }
        }
        _ => missed.push("copy ratio undefined"),
    }

    // 2. The sharpest signature: random neighbours worse than none at all.
    if leaked.random_worse_than_ablated() == Some(true) {
        flagged.push("random neighbours are worse than no neighbours");
    } else {
        missed.push("random neighbours did not fall below the ablated baseline");
    }

    // 3. Copying concentrates at low t, where few positions are still masked
    //    and a neighbour's continuation can supply them.
    let low = |r: &EvalReport| -> Option<f32> {
        let real = r.get(NeighbourCondition::Real)?.mean_below(0.4)?;
        let random = r.get(NeighbourCondition::Random)?.mean_below(0.4)?;
        Some(random - real)
    };
    if let (Some(h), Some(l)) = (low(honest), low(leaked)) {
        println!("low-t copy gap: honest {h:+.4}  leaked {l:+.4}");
        if l > h + MARGIN {
            flagged.push("low-t copy gap widened");
        } else {
            missed.push("low-t copy gap did not widen");
        }
    }

    for f in &flagged {
        println!("  FLAGGED: {f}");
    }
    for m in &missed {
        println!("  missed:  {m}");
    }

    // One weak signal out of three is not a detection. Requiring a majority
    // is the difference between a protocol and a coin flip.
    if flagged.len() < 2 {
        eprintln!(
            "\nFAIL: {} of 3 signals flagged a run that queries x_0 directly.\n\
             The protocol is not ready to judge a real run.",
            flagged.len()
        );
        std::process::exit(1);
    }
    println!(
        "\nPASS: {} of 3 signals flagged the leaked arm.",
        flagged.len()
    );
}

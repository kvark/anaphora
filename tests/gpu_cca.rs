//! GPU tests for the CCA retrofit, on whatever adapter Meganeura selects
//! (Lavapipe in headless CI).
//!
//! These cover the two claims a retrofit rests on:
//!
//! 1. At initialisation the CCA block is **exactly** the identity, so the
//!    frozen backbone's calibration survives the first forward pass.
//! 2. Only the neighbour encoder, the CCA blocks, and the gates receive
//!    gradients.

use anaphora::config::CcaConfig;
use anaphora::model::backbone::{Backbone, BackboneConfig};
use anaphora::model::cca::input_names;
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use meganeura::Graph;

const SEQ: usize = 32;
const CHUNK: usize = 8;
const K: usize = 2;
const R: usize = 8;
const HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const VOCAB: usize = 64;

fn cca_config() -> CcaConfig {
    // CCA after backbone layers 1 and 3.
    CcaConfig::new(SEQ, CHUNK, K, R, HEADS, HEADS, HEAD_DIM, 2, 1).expect("valid shapes")
}

fn backbone_config() -> BackboneConfig {
    BackboneConfig {
        vocab_size: VOCAB,
        num_layers: 4,
        num_heads: HEADS as u32,
        num_kv_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: 64,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
    }
}

fn encoder_config() -> NeighbourEncoderConfig {
    NeighbourEncoderConfig {
        vocab_size: VOCAB,
        num_layers: 1,
        num_heads: HEADS as u32,
        num_kv_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: 64,
        scope: EncoderScope::PerNeighbour,
    }
}

/// Deterministic parameter values keyed by name, so two graphs that share a
/// parameter name get bit-identical weights for it.
fn param_values(name: &str, len: usize) -> Vec<f32> {
    let mut state = name.bytes().fold(0x2545_F491_4F6C_DD1Du64, |acc, b| {
        (acc ^ b as u64).wrapping_mul(0x0100_0000_01B3)
    });
    let is_norm = name.contains("norm");
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as f32 / 16_777_216.0) - 0.5; // ~[-0.5, 0.5)
            if is_norm {
                1.0 + unit * 0.1
            } else {
                unit * 0.2
            }
        })
        .collect()
}

fn fill_params(session: &mut meganeura::runtime::Session) {
    let names: Vec<String> = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        let len = session.param_size(&name).expect("declared parameter");
        session.set_parameter(&name, &param_values(&name, len));
    }
}

fn zero_params(session: &mut meganeura::runtime::Session, names: &[String]) {
    for name in names {
        let len = session.param_size(name).expect("declared parameter");
        session.set_parameter(name, &vec![0.0; len]);
    }
}

fn token_ids() -> Vec<u32> {
    (0..SEQ).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect()
}

/// Build the retrofit model and its logits, with neighbours fed as cached KV.
fn build_model(activation: GateActivation) -> (Graph, CcaModel) {
    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca: cca_config(),
            backbone: backbone_config(),
            encoder: encoder_config(),
            activation,
            neighbours: NeighbourInput::Cached,
        },
    );
    g.set_outputs(vec![model.logits()]);
    (g, model)
}

/// The same backbone with no CCA blocks at all — the reference.
fn build_bare_backbone() -> Graph {
    let mut g = Graph::new();
    let cfg = backbone_config();
    let backbone = Backbone::new(&mut g, "backbone", cfg);
    let token_ids = g.input_u32("token_ids", &[SEQ]);
    let mut x = backbone.embed(&mut g, token_ids);
    for layer in 0..cfg.num_layers {
        x = backbone.layer(&mut g, x, layer);
    }
    let logits = backbone.head(&mut g, x);
    g.set_outputs(vec![logits]);
    g
}

fn run_model(activation: GateActivation) -> Vec<f32> {
    let (g, model) = build_model(activation);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    fill_params(&mut session);
    // The retrofit's initialisation: the gate's output layer starts at zero.
    zero_params(&mut session, &model.zero_init_param_names());

    let cfg = cca_config();
    let kv_rows = cfg.num_chunks() * cfg.neighbour_kv_rows();
    session.set_input_u32("token_ids", &token_ids());
    session.set_input(input_names::T_COL, &[0.5; SEQ]);
    // Every chunk retrieved, so nothing is masked out: the identity must hold
    // because of the gate, not because retrieval was switched off.
    session.set_input(input_names::RETRIEVAL_MASK, &[1.0; SEQ]);
    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("neighbour_kv_probe", kv_rows * cfg.model_dim()),
    );
    session.step();
    session.wait();
    session.read_output(SEQ * VOCAB)
}

fn run_bare_backbone() -> Vec<f32> {
    let g = build_bare_backbone();
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    fill_params(&mut session);
    session.set_input_u32("token_ids", &token_ids());
    session.step();
    session.wait();
    session.read_output(SEQ * VOCAB)
}

fn assert_matches_backbone(activation: GateActivation, label: &str) {
    let with_cca = run_model(activation);
    let bare = run_bare_backbone();
    assert_eq!(with_cca.len(), bare.len());

    let mut worst = 0.0f32;
    for (i, (a, b)) in with_cca.iter().zip(&bare).enumerate() {
        assert!(a.is_finite(), "{label}: non-finite logit at {i}");
        worst = worst.max((a - b).abs());
    }
    // Not "close enough": a zero gate contributes an exact zero, so the two
    // graphs differ only by the floating-point noise of the extra add.
    assert!(
        worst < 1e-4,
        "{label}: CCA block is not the identity at init (max |diff| = {worst})"
    );
}

#[test]
fn tanh_gate_is_identity_at_init() {
    // The design sketch zero-inits the gate's last layer and applies a
    // sigmoid — but sigmoid(0) = 0.5, so the block would return h + 0.5*ctx
    // and inject half an untrained cross-attention output into the frozen
    // residual stream. tanh(0) = 0 exactly.
    assert_matches_backbone(GateActivation::Tanh, "tanh");
}

#[test]
fn scaled_sigmoid_gate_is_identity_at_init() {
    // The alternative: keep the gate in [0, 1] and multiply by a zero-init
    // learned scalar.
    assert_matches_backbone(GateActivation::ScaledSigmoid, "scaled sigmoid");
}

#[test]
fn only_the_retrieval_path_receives_gradients() {
    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca: cca_config(),
            backbone: backbone_config(),
            encoder: encoder_config(),
            activation: GateActivation::Tanh,
            neighbours: NeighbourInput::Encoded,
        },
    );
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);

    let session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    // The frozen backbone: no gradients, so the optimiser never touches it.
    for name in [
        "backbone.embed",
        "backbone.layers.0.q_proj",
        "backbone.layers.3.down_proj",
        "backbone.lm_head",
    ] {
        assert!(
            !session.has_param_grad(name),
            "{name} is frozen and must not receive a gradient"
        );
    }

    // The three things a retrofit trains.
    let trainable = model.trainable_param_names();
    assert!(
        trainable
            .iter()
            .any(|n| n.starts_with("neighbour_encoder.")),
        "the neighbour encoder must be trainable"
    );
    for name in ["cca.1.q_proj", "cca.1.gate.w1", "cca.3.gate.w2"] {
        assert!(
            trainable.iter().any(|n| n == name),
            "{name} should be listed trainable"
        );
        assert!(
            session.has_param_grad(name),
            "{name} must receive a gradient"
        );
    }
}

#[test]
fn zero_init_gate_still_learns() {
    // Zero-init is only safe if it does not also zero the gradient. A gate
    // that starts closed and cannot open is a block that never trains, which
    // is what a large negative sigmoid bias would have produced.
    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca: cca_config(),
            backbone: backbone_config(),
            encoder: encoder_config(),
            activation: GateActivation::Tanh,
            neighbours: NeighbourInput::Cached,
        },
    );
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    fill_params(&mut session);
    zero_params(&mut session, &model.zero_init_param_names());

    let cfg = cca_config();
    let kv_rows = cfg.num_chunks() * cfg.neighbour_kv_rows();
    session.set_input_u32("token_ids", &token_ids());
    session.set_input(input_names::T_COL, &[0.5; SEQ]);
    session.set_input(input_names::RETRIEVAL_MASK, &[1.0; SEQ]);
    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("neighbour_kv_probe", kv_rows * cfg.model_dim()),
    );
    let mut one_hot = vec![0.0f32; SEQ * VOCAB];
    for pos in 0..SEQ {
        one_hot[pos * VOCAB + (pos % VOCAB)] = 1.0;
    }
    session.set_input("labels", &one_hot);

    session.step();
    session.wait();

    let name = "cca.1.gate.w2";
    let len = session.param_size(name).expect("declared");
    let mut grad = vec![0.0f32; len];
    session.read_param_grad(name, &mut grad);
    let magnitude: f32 = grad.iter().map(|v| v.abs()).sum();
    assert!(
        magnitude > 0.0,
        "the zero-initialised gate layer must still receive a gradient"
    );
    assert!(
        grad.iter().all(|v| v.is_finite()),
        "gate gradient must be finite"
    );
}

#[test]
fn neighbour_content_changes_the_output() {
    // Guards against a vacuous pass of the identity tests above. If the
    // neighbour block were disconnected from the logits, "the CCA block is
    // the identity at init" would hold for the wrong reason, and every
    // retrieval measurement downstream would be reading a constant.
    let (g, _model) = build_model(GateActivation::Tanh);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    fill_params(&mut session);
    // Deliberately NOT zero-initialising the gate: the question here is
    // whether the path carries signal when the gate is open.

    let cfg = cca_config();
    let kv_rows = cfg.num_chunks() * cfg.neighbour_kv_rows();
    session.set_input_u32("token_ids", &token_ids());
    session.set_input(input_names::T_COL, &[0.5; SEQ]);
    session.set_input(input_names::RETRIEVAL_MASK, &[1.0; SEQ]);

    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("kv_a", kv_rows * cfg.model_dim()),
    );
    session.step();
    session.wait();
    let with_a = session.read_output(SEQ * VOCAB);

    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("kv_b", kv_rows * cfg.model_dim()),
    );
    session.step();
    session.wait();
    let with_b = session.read_output(SEQ * VOCAB);

    let diff: f32 = with_a
        .iter()
        .zip(&with_b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(
        diff > 1e-6,
        "different neighbours produced identical logits: the CCA path is not connected"
    );
}

#[test]
fn the_retrieval_mask_gates_the_whole_block() {
    // A chunk that retrieved nothing must contribute nothing, whatever the
    // gate has learned. Attending to a zero-filled key/value block is not the
    // same as not attending.
    let (g, _model) = build_model(GateActivation::Tanh);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    fill_params(&mut session);

    let cfg = cca_config();
    let kv_rows = cfg.num_chunks() * cfg.neighbour_kv_rows();
    session.set_input_u32("token_ids", &token_ids());
    session.set_input(input_names::T_COL, &[0.5; SEQ]);
    session.set_input(input_names::RETRIEVAL_MASK, &[0.0; SEQ]);

    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("kv_a", kv_rows * cfg.model_dim()),
    );
    session.step();
    session.wait();
    let masked_a = session.read_output(SEQ * VOCAB);

    session.set_input(
        input_names::NEIGHBOUR_KV,
        &param_values("kv_b", kv_rows * cfg.model_dim()),
    );
    session.step();
    session.wait();
    let masked_b = session.read_output(SEQ * VOCAB);

    assert_eq!(
        masked_a, masked_b,
        "with the mask at zero the neighbours must not reach the output"
    );
}

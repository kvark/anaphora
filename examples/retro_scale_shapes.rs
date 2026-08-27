//! Build the retrofit at RETRO's published shape and report graph size.
//!
//! Graph construction only — no session, no GPU. This is the cheap check
//! that the shapes compose at `n = 2048`, and it reports the operator and
//! dispatch-relevant node counts that decide whether a fused chunked
//! cross-attention operator is worth building (see `docs/roadmap.md`).

use anaphora::config::CcaConfig;
use anaphora::model::backbone::BackboneConfig;
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use meganeura::Graph;

fn main() {
    // LLaDA-8B-ish geometry, at RETRO's retrieval shape.
    let heads = 32;
    let head_dim = 128;
    let cca = CcaConfig::retro_like(heads, heads, head_dim).expect("valid shapes");
    let backbone = BackboneConfig {
        vocab_size: 32_000,
        num_layers: 32,
        num_heads: heads as u32,
        num_kv_heads: heads as u32,
        head_dim: head_dim as u32,
        intermediate_size: 11_008,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    };
    let encoder = NeighbourEncoderConfig {
        vocab_size: 32_000,
        num_layers: 2,
        num_heads: heads as u32,
        num_kv_heads: heads as u32,
        head_dim: head_dim as u32,
        intermediate_size: 11_008,
        scope: EncoderScope::PerNeighbour,
    };

    for neighbours in [NeighbourInput::Cached, NeighbourInput::Encoded] {
        let mut g = Graph::new();
        let model = CcaModel::build(
            &mut g,
            ModelConfig {
                cca,
                backbone,
                encoder,
                activation: GateActivation::Tanh,
                neighbours,
            },
        );
        g.set_outputs(vec![model.logits()]);

        println!("--- {neighbours:?} ---");
        println!(
            "  n={} m={} l={} k={} r={} d={}",
            cca.seq_len(),
            cca.chunk_size(),
            cca.num_chunks(),
            cca.neighbours_per_chunk(),
            cca.neighbour_len(),
            cca.model_dim()
        );
        println!(
            "  CCA blocks after backbone layers {:?}",
            model.blocks().iter().map(|i| i.layer).collect::<Vec<_>>()
        );
        println!("  graph nodes: {}", g.nodes().len());
        println!(
            "  cross-attentions: {} ({} blocks x l={})",
            model.blocks().len() * cca.num_chunks(),
            model.blocks().len(),
            cca.num_chunks()
        );
        println!(
            "  trainable params: {}",
            model.trainable_param_names().len()
        );
        println!(
            "  zero-init params: {}",
            model.zero_init_param_names().len()
        );
    }
}

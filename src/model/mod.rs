//! Graph construction: the frozen backbone, the CCA blocks, and the gates.

pub mod backbone;
pub mod cca;
pub mod encoder;
pub mod gate;
pub mod rows;

use crate::config::CcaConfig;
use backbone::{Backbone, BackboneConfig};
use cca::{CcaBlock, CcaInputs, input_names};
use encoder::{NeighbourEncoder, NeighbourEncoderConfig};
use gate::GateActivation;
use meganeura::{Graph, NodeId};

/// Where the encoded neighbour keys/values come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighbourInput {
    /// Encode neighbour token ids inside this graph.
    ///
    /// The training arrangement: gradients reach the encoder, which is one of
    /// the three things a retrofit trains.
    Encoded,
    /// Take the encoded keys/values as a graph input.
    ///
    /// The sampling arrangement. Encoding neighbours is the expensive half of
    /// retrieval, and a denoising trajectory refreshes only at a few
    /// thresholds — so the host encodes once per refresh and feeds the cached
    /// result on every step in between. Feeding it as an input is what makes
    /// that cache expressible; with the encoder in-graph, every step would
    /// re-encode.
    Cached,
}

/// A CCA block and the backbone layer it was inserted after.
#[derive(Debug, Clone)]
pub struct CcaInsertion {
    /// Index of the backbone layer this block follows.
    pub layer: usize,
    /// The block itself.
    pub block: CcaBlock,
}

/// A retrieval-augmented masked-diffusion model.
#[derive(Debug, Clone)]
pub struct CcaModel {
    cfg: CcaConfig,
    backbone: Backbone,
    encoder: Option<NeighbourEncoder>,
    blocks: Vec<CcaInsertion>,
    logits: NodeId,
}

/// What to build.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelConfig {
    /// Retrieval shapes.
    pub cca: CcaConfig,
    /// The frozen backbone.
    pub backbone: BackboneConfig,
    /// The trained neighbour encoder.
    pub encoder: NeighbourEncoderConfig,
    /// Gate activation.
    pub activation: GateActivation,
    /// Where encoded neighbours come from.
    pub neighbours: NeighbourInput,
}

impl CcaModel {
    /// Build the forward graph.
    ///
    /// Declares these inputs:
    ///
    /// | name | shape | dtype |
    /// |---|---|---|
    /// | `token_ids` | `[n]` | U32 |
    /// | `cca.t` | `[n, 1]` | f32 |
    /// | `cca.retrieval_mask` | `[n, 1]` | f32 |
    /// | `cca.neighbour_tokens` | `[l * k * r]` | U32 (when [`NeighbourInput::Encoded`]) |
    /// | `cca.neighbour_kv` | `[l * k * r, d]` | f32 (when [`NeighbourInput::Cached`]) |
    ///
    /// Returns the model and the `[n, vocab]` logits node.
    pub fn build(g: &mut Graph, cfg: ModelConfig) -> Self {
        let cca = cfg.cca;
        let n = cca.seq_len();
        let d = cca.model_dim();
        assert_eq!(
            d,
            cfg.backbone.model_dim(),
            "CCA width {d} must match the backbone residual width {}",
            cfg.backbone.model_dim()
        );
        assert_eq!(
            d,
            cfg.encoder.model_dim(),
            "CCA width {d} must match the neighbour encoder width {}",
            cfg.encoder.model_dim()
        );

        let backbone = Backbone::new(g, "backbone", cfg.backbone);
        let token_ids = g.input_u32("token_ids", &[n]);
        let t_col = g.input(input_names::T_COL, &[n, 1]);
        let retrieval_mask = g.input(input_names::RETRIEVAL_MASK, &[n, 1]);

        let kv_rows = cca.num_chunks() * cca.neighbour_kv_rows();
        let (encoder, neighbour_kv) = match cfg.neighbours {
            NeighbourInput::Encoded => {
                let encoder = NeighbourEncoder::new(g, "neighbour_encoder", cfg.encoder);
                let group_rows =
                    encoder.group_rows(cca.neighbours_per_chunk(), cca.neighbour_len());
                let groups = kv_rows / group_rows;
                let tokens = g.input_u32("cca.neighbour_tokens", &[kv_rows]);
                // Embed once for the whole neighbour block, then slice per
                // group in f32 space. Groups are independent — that
                // independence is the point of `EncoderScope` — so they run
                // as separate slices rather than one long sequence, which
                // would let neighbours attend across group boundaries.
                let embedded = encoder.embed(g, tokens);
                let mut parts = Vec::with_capacity(groups);
                for group in 0..groups {
                    let slice =
                        rows::slice_rows(g, embedded, kv_rows, d, group * group_rows, group_rows);
                    parts.push((encoder.forward_embedded(g, slice, group_rows), group_rows));
                }
                let kv = rows::concat_rows(g, &parts, d);
                (Some(encoder), kv)
            }
            NeighbourInput::Cached => (None, g.input(input_names::NEIGHBOUR_KV, &[kv_rows, d])),
        };

        let inputs = CcaInputs {
            t_col,
            retrieval_mask,
            neighbour_kv,
        };

        let mut x = backbone.embed(g, token_ids);
        let mut blocks = Vec::new();
        for layer in 0..backbone.num_layers() {
            x = backbone.layer(g, x, layer);
            if cca.is_cca_layer(layer) {
                let block = CcaBlock::new(g, &format!("cca.{layer}"), cca, cfg.activation);
                x = block.forward(g, x, inputs);
                blocks.push(CcaInsertion { layer, block });
            }
        }
        let logits = backbone.head(g, x);
        let logits = g.named(logits, "logits");

        Self {
            cfg: cca,
            backbone,
            encoder,
            blocks,
            logits,
        }
    }

    /// The `[n, vocab]` logits node.
    pub fn logits(&self) -> NodeId {
        self.logits
    }

    /// The retrieval configuration.
    pub fn config(&self) -> CcaConfig {
        self.cfg
    }

    /// The frozen backbone.
    pub fn backbone(&self) -> &Backbone {
        &self.backbone
    }

    /// The neighbour encoder, when one is in-graph.
    pub fn encoder(&self) -> Option<&NeighbourEncoder> {
        self.encoder.as_ref()
    }

    /// The CCA blocks, paired with the backbone layer they follow.
    pub fn blocks(&self) -> &[CcaInsertion] {
        &self.blocks
    }

    /// Every trainable parameter name: the encoder, the CCA blocks, and the
    /// gates. The backbone is absent by construction.
    pub fn trainable_param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(ref encoder) = self.encoder {
            names.extend(encoder.param_names());
        }
        for insertion in &self.blocks {
            names.extend(insertion.block.param_names());
        }
        names
    }

    /// Parameters that must be written as exactly zero before the first step,
    /// so every CCA block starts as the identity.
    pub fn zero_init_param_names(&self) -> Vec<String> {
        self.blocks
            .iter()
            .flat_map(|insertion| {
                insertion
                    .block
                    .gate()
                    .params()
                    .zero_init_names()
                    .into_iter()
                    .map(str::to_owned)
            })
            .collect()
    }
}

//! The neighbour encoder: retrieved tokens to cross-attention keys/values.
//!
//! Trained, not frozen. RETRO's encoder was a small bidirectional transformer
//! and this is the same shape: retrieved neighbours are clean text with no
//! ordering constraint to respect, so attention over them is full rather than
//! causal.
//!
//! Encoding neighbours is the expensive half of retrieval, which is what
//! makes [`crate::schedule::RefreshSchedule`] worth having — the output of
//! this module is what gets cached between refresh thresholds.

use meganeura::{Graph, NodeId};

/// How much context one encoder pass sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderScope {
    /// Encode each neighbour's `r` tokens on its own. RETRO's arrangement:
    /// a neighbour's representation does not depend on which other
    /// neighbours the search happened to return alongside it, which keeps
    /// the encoding reproducible under index changes.
    #[default]
    PerNeighbour,
    /// Encode a chunk's whole `k * r` key/value block at once. The `k`
    /// neighbours attend to each other, and the operator count per CCA
    /// refresh drops by a factor of `k`.
    PerChunk,
}

/// Shape of the neighbour encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighbourEncoderConfig {
    /// Token vocabulary, shared with the backbone.
    pub vocab_size: usize,
    /// Transformer layers.
    pub num_layers: usize,
    /// Query heads.
    pub num_heads: u32,
    /// Key/value heads.
    pub num_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// Encoding granularity.
    pub scope: EncoderScope,
}

impl NeighbourEncoderConfig {
    /// Model width, `num_heads * head_dim`.
    pub fn model_dim(&self) -> usize {
        (self.num_heads * self.head_dim) as usize
    }

    /// Key/value width.
    pub fn kv_dim(&self) -> usize {
        (self.num_kv_heads * self.head_dim) as usize
    }
}

const RMS_EPS: f32 = 1e-5;

/// Parameter node ids for one encoder layer.
#[derive(Debug, Clone, Copy)]
struct LayerNodes {
    attn_norm: NodeId,
    q_proj: NodeId,
    k_proj: NodeId,
    v_proj: NodeId,
    o_proj: NodeId,
    ffn_norm: NodeId,
    gate_proj: NodeId,
    up_proj: NodeId,
    down_proj: NodeId,
}

/// A bidirectional transformer over retrieved neighbour tokens.
///
/// Parameters are declared once, in [`NeighbourEncoder::new`], and every
/// [`NeighbourEncoder::forward`] call reuses those node ids. One set of
/// weights, many applications — which is what makes per-neighbour encoding
/// affordable in parameter terms even at `l * k` groups per refresh, and is
/// also required for correctness: Meganeura pairs parameters with gradients
/// positionally, one per `Op::Parameter` node, so re-declaring a name inside
/// the loop would allocate one buffer per application and split the weight
/// into `l * k` independent copies.
#[derive(Debug, Clone)]
pub struct NeighbourEncoder {
    cfg: NeighbourEncoderConfig,
    prefix: String,
    embed: NodeId,
    layers: Vec<LayerNodes>,
    out_norm: NodeId,
}

impl NeighbourEncoder {
    /// Declare the encoder's parameters under `prefix`.
    pub fn new(g: &mut Graph, prefix: &str, cfg: NeighbourEncoderConfig) -> Self {
        let d = cfg.model_dim();
        let kv = cfg.kv_dim();
        let embed = g.parameter(&format!("{prefix}.embed"), &[cfg.vocab_size, d]);
        let layers = (0..cfg.num_layers)
            .map(|layer| {
                let p = format!("{prefix}.layers.{layer}");
                LayerNodes {
                    attn_norm: g.parameter(&format!("{p}.attn_norm"), &[d]),
                    q_proj: g.parameter(&format!("{p}.q_proj"), &[d, d]),
                    k_proj: g.parameter(&format!("{p}.k_proj"), &[d, kv]),
                    v_proj: g.parameter(&format!("{p}.v_proj"), &[d, kv]),
                    o_proj: g.parameter(&format!("{p}.o_proj"), &[d, d]),
                    ffn_norm: g.parameter(&format!("{p}.ffn_norm"), &[d]),
                    gate_proj: g.parameter(&format!("{p}.gate_proj"), &[d, cfg.intermediate_size]),
                    up_proj: g.parameter(&format!("{p}.up_proj"), &[d, cfg.intermediate_size]),
                    down_proj: g.parameter(&format!("{p}.down_proj"), &[cfg.intermediate_size, d]),
                }
            })
            .collect();
        let out_norm = g.parameter(&format!("{prefix}.out_norm"), &[d]);
        Self {
            cfg,
            prefix: prefix.to_owned(),
            embed,
            layers,
            out_norm,
        }
    }

    /// The encoder's configuration.
    pub fn config(&self) -> NeighbourEncoderConfig {
        self.cfg
    }

    /// Rows encoded per pass: `r` for [`EncoderScope::PerNeighbour`],
    /// `k * r` for [`EncoderScope::PerChunk`].
    pub fn group_rows(&self, k: usize, r: usize) -> usize {
        match self.cfg.scope {
            EncoderScope::PerNeighbour => r,
            EncoderScope::PerChunk => k * r,
        }
    }

    /// Number of encoder passes per chunk.
    pub fn groups_per_chunk(&self, k: usize) -> usize {
        match self.cfg.scope {
            EncoderScope::PerNeighbour => k,
            EncoderScope::PerChunk => 1,
        }
    }

    /// Every parameter name this encoder owns.
    pub fn param_names(&self) -> Vec<String> {
        let mut names = vec![format!("{}.embed", self.prefix)];
        for layer in 0..self.cfg.num_layers {
            let p = format!("{}.layers.{layer}", self.prefix);
            names.extend([
                format!("{p}.attn_norm"),
                format!("{p}.q_proj"),
                format!("{p}.k_proj"),
                format!("{p}.v_proj"),
                format!("{p}.o_proj"),
                format!("{p}.ffn_norm"),
                format!("{p}.gate_proj"),
                format!("{p}.up_proj"),
                format!("{p}.down_proj"),
            ]);
        }
        names.push(format!("{}.out_norm", self.prefix));
        names
    }

    /// Embed a `[rows]` U32 token vector into `[rows, d]`.
    ///
    /// Split out from [`NeighbourEncoder::forward`] because the caller embeds
    /// *all* `l * k * r` neighbour tokens in one gather and then slices the
    /// result per group. Slicing before embedding is not an option:
    /// `split_a`/`split_b` are typed f32, and `Graph::reshape` relabels its
    /// output f32 unconditionally, so a U32 vector cannot be sliced without
    /// silently acquiring the wrong dtype.
    pub fn embed(&self, g: &mut Graph, token_ids: NodeId) -> NodeId {
        g.embedding(token_ids, self.embed)
    }

    /// Encode one group of `rows` neighbour token ids into `[rows, d]`.
    ///
    /// `token_ids` is a `[rows]` U32 value.
    pub fn forward(&self, g: &mut Graph, token_ids: NodeId, rows: usize) -> NodeId {
        let embedded = self.embed(g, token_ids);
        self.forward_embedded(g, embedded, rows)
    }

    /// Run the transformer stack over already-embedded `[rows, d]` values.
    pub fn forward_embedded(&self, g: &mut Graph, embedded: NodeId, rows: usize) -> NodeId {
        let cfg = self.cfg;
        let mut x = embedded;

        for nodes in &self.layers {
            let h = g.rms_norm(x, nodes.attn_norm, RMS_EPS);
            let q = g.matmul(h, nodes.q_proj);
            let k = g.matmul(h, nodes.k_proj);
            let v = g.matmul(h, nodes.v_proj);

            // Bidirectional: a retrieved neighbour is complete text, and the
            // encoder is not predicting its continuation.
            let attn = g.multi_head_attn(
                q,
                k,
                v,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                false,
            );
            let attn = g.matmul(attn, nodes.o_proj);
            x = g.add(x, attn);

            let h = g.rms_norm(x, nodes.ffn_norm, RMS_EPS);
            let gate = g.matmul(h, nodes.gate_proj);
            let up = g.matmul(h, nodes.up_proj);
            let ffn = g.swiglu(gate, up);
            let ffn = g.matmul(ffn, nodes.down_proj);
            x = g.add(x, ffn);
        }

        let x = g.rms_norm(x, self.out_norm, RMS_EPS);
        let _ = rows;
        g.named(x, format!("{}.kv", self.prefix))
    }
}

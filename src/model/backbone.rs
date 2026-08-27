//! The frozen masked-diffusion backbone.
//!
//! Anaphora targets a pretrained diffusion LM — LLaDA-8B, Dream-7B — whose
//! weights do not move. Structurally these are LLaMA-shaped transformers with
//! one decisive difference: attention is **bidirectional**. A masked
//! diffusion model predicts every masked position from the whole visible
//! sequence at once, so there is no causal mask and no left-to-right order to
//! respect. That is also what makes the retrieval story here different from
//! RETRO's, and why the chunk offset disappears.
//!
//! Every parameter declared in this module goes through
//! [`crate::model::cca::frozen_parameter`]. The retrofit's claim is that only
//! the encoder, the CCA blocks, and the gates train; freezing at declaration
//! makes that structural rather than a matter of remembering.

use crate::model::cca::frozen_parameter;
use meganeura::{Graph, NodeId};

/// Shape of the frozen backbone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackboneConfig {
    /// Token vocabulary, `[MASK]` included.
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
    /// RoPE base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
}

impl BackboneConfig {
    /// Model width, `num_heads * head_dim`.
    pub fn model_dim(&self) -> usize {
        (self.num_heads * self.head_dim) as usize
    }

    /// Key/value width.
    pub fn kv_dim(&self) -> usize {
        (self.num_kv_heads * self.head_dim) as usize
    }

    /// A small configuration for tests and shape work.
    pub fn small_test() -> Self {
        Self {
            vocab_size: 256,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }
}

/// Whether the backbone's parameters train.
///
/// A retrofit freezes them — that is the premise. But Phase 1 has to *make*
/// the backbone it later freezes, and a random backbone is not a stand-in: a
/// randomly initialised LM head cannot express a specific token, so a
/// retrofit bolted onto one has no way to demonstrate that retrieval helps,
/// and no way to leak either. Calibrating a leak detector against a model
/// that cannot leak measures nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Freezing {
    /// Parameters are routed through `stop_gradient`. The retrofit setting.
    #[default]
    Frozen,
    /// Parameters train. For pretraining the backbone that will later be
    /// frozen.
    Trainable,
}

/// Parameter node ids for one backbone layer.
#[derive(Debug, Clone, Copy)]
pub struct BackboneLayer {
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

/// The frozen backbone's parameters.
#[derive(Debug, Clone)]
pub struct Backbone {
    cfg: BackboneConfig,
    embed: NodeId,
    layers: Vec<BackboneLayer>,
    out_norm: NodeId,
    lm_head: NodeId,
}

impl Backbone {
    /// Declare the backbone's parameters, all frozen, under `prefix`.
    pub fn new(g: &mut Graph, prefix: &str, cfg: BackboneConfig) -> Self {
        Self::with_freezing(g, prefix, cfg, Freezing::Frozen)
    }

    /// Declare the backbone's parameters under `prefix`, frozen or not.
    pub fn with_freezing(
        g: &mut Graph,
        prefix: &str,
        cfg: BackboneConfig,
        freezing: Freezing,
    ) -> Self {
        let declare = |g: &mut Graph, name: &str, shape: &[usize]| match freezing {
            Freezing::Frozen => frozen_parameter(g, name, shape),
            Freezing::Trainable => g.parameter(name, shape),
        };
        let d = cfg.model_dim();
        let kv = cfg.kv_dim();
        let embed = declare(g, &format!("{prefix}.embed"), &[cfg.vocab_size, d]);
        let layers = (0..cfg.num_layers)
            .map(|layer| {
                let p = format!("{prefix}.layers.{layer}");
                BackboneLayer {
                    attn_norm: declare(g, &format!("{p}.attn_norm"), &[d]),
                    q_proj: declare(g, &format!("{p}.q_proj"), &[d, d]),
                    k_proj: declare(g, &format!("{p}.k_proj"), &[d, kv]),
                    v_proj: declare(g, &format!("{p}.v_proj"), &[d, kv]),
                    o_proj: declare(g, &format!("{p}.o_proj"), &[d, d]),
                    ffn_norm: declare(g, &format!("{p}.ffn_norm"), &[d]),
                    gate_proj: declare(g, &format!("{p}.gate_proj"), &[d, cfg.intermediate_size]),
                    up_proj: declare(g, &format!("{p}.up_proj"), &[d, cfg.intermediate_size]),
                    down_proj: declare(g, &format!("{p}.down_proj"), &[cfg.intermediate_size, d]),
                }
            })
            .collect();
        let out_norm = declare(g, &format!("{prefix}.out_norm"), &[d]);
        let lm_head = declare(g, &format!("{prefix}.lm_head"), &[d, cfg.vocab_size]);
        Self {
            cfg,
            embed,
            layers,
            out_norm,
            lm_head,
        }
    }

    /// The backbone's configuration.
    pub fn config(&self) -> BackboneConfig {
        self.cfg
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Every parameter name this backbone owns, under `prefix`.
    ///
    /// The list a pretrained backbone is transferred through: train one graph
    /// with [`Freezing::Trainable`], read these, and write them into the
    /// retrofit graph that declares the same names frozen.
    pub fn param_names(prefix: &str, cfg: BackboneConfig) -> Vec<String> {
        let mut names = vec![format!("{prefix}.embed")];
        for layer in 0..cfg.num_layers {
            let p = format!("{prefix}.layers.{layer}");
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
        names.push(format!("{prefix}.out_norm"));
        names.push(format!("{prefix}.lm_head"));
        names
    }

    /// Embed `token_ids` (`[n]`, U32) into `[n, d]`.
    pub fn embed(&self, g: &mut Graph, token_ids: NodeId) -> NodeId {
        g.embedding(token_ids, self.embed)
    }

    /// Run backbone layer `layer` over `[n, d]`.
    pub fn layer(&self, g: &mut Graph, x: NodeId, layer: usize) -> NodeId {
        let cfg = self.cfg;
        let nodes = self.layers[layer];
        let eps = cfg.rms_norm_eps;

        let h = g.rms_norm(x, nodes.attn_norm, eps);
        let q = g.matmul(h, nodes.q_proj);
        let k = g.matmul(h, nodes.k_proj);
        let v = g.matmul(h, nodes.v_proj);
        let q = g.rope(q, cfg.rope_theta, cfg.head_dim);
        let k = g.rope(k, cfg.rope_theta, cfg.head_dim);

        // Bidirectional. A masked diffusion LM conditions every position on
        // the whole visible sequence, so `causal_attention` would be wrong
        // here in a way that still trains and still lowers loss.
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
        let x = g.add(x, attn);

        let h = g.rms_norm(x, nodes.ffn_norm, eps);
        let gate = g.matmul(h, nodes.gate_proj);
        let up = g.matmul(h, nodes.up_proj);
        let ffn = g.swiglu(gate, up);
        let ffn = g.matmul(ffn, nodes.down_proj);
        g.add(x, ffn)
    }

    /// Final norm and unembedding to `[n, vocab]`.
    pub fn head(&self, g: &mut Graph, x: NodeId) -> NodeId {
        let x = g.rms_norm(x, self.out_norm, self.cfg.rms_norm_eps);
        g.matmul(x, self.lm_head)
    }
}

//! The chunked cross-attention block.
//!
//! Section 2 of the design sketch. Inserted after every `P`th backbone layer
//! (RETRO used `P = 3`, starting at layer 6).
//!
//! # Block-diagonal attention without a mask
//!
//! Chunk `u`'s queries may attend only to `Ret(chunk u)` — its own `k * r`
//! retrieved key/value rows. In a framework with a batch axis and an additive
//! attention mask this is one `[n, l * k * r]` attention with a
//! block-diagonal mask, and most of the score matrix is discarded.
//!
//! Meganeura's attention operators are two-dimensional and take no mask, so
//! the block-diagonal form is written directly: `l` independent
//! `[m, k * r]` cross-attentions, one per chunk. This is not a workaround.
//! The masked formulation computes `n * l * k * r` scores and throws away all
//! but `1 / l` of them; the explicit one computes exactly the `l * m * k * r`
//! scores that survive. What it costs instead is `l` dispatches per block
//! rather than one, which is the trade to revisit if dispatch overhead
//! dominates — see `docs/roadmap.md`.
//!
//! RETRO's `C_u+` offset is absent, as the sketch notes: it existed to stop a
//! chunk from attending to neighbours retrieved with its own tokens, and
//! diffusion has no ordering to exploit for that. The principle it protected
//! is enforced structurally instead, in [`crate::view`].

use crate::config::CcaConfig;
use crate::model::gate::{GateActivation, TimeConditionedGate};
use crate::model::rows::{concat_rows, slice_rows};
use meganeura::{Graph, NodeId};

const RMS_EPS: f32 = 1e-5;

/// Inputs a CCA block reads from the host each step.
///
/// All three change per denoising step and none of them changes shape, so
/// they are graph inputs rather than constants — the graph compiles once and
/// the trajectory runs against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcaInputs {
    /// `[n, 1]`, the current noise level repeated per row.
    pub t_col: NodeId,
    /// `[n, 1]`, `1.0` where the row's chunk retrieved neighbours and `0.0`
    /// where it did not.
    pub retrieval_mask: NodeId,
    /// `[l * k * r, d]` encoded neighbour keys/values, chunk-major.
    pub neighbour_kv: NodeId,
}

/// Names of the inputs a [`crate::model::CcaModel`] declares.
pub mod input_names {
    /// `[n, 1]` noise level column.
    pub const T_COL: &str = "cca.t";
    /// `[n, 1]` per-row retrieval mask.
    pub const RETRIEVAL_MASK: &str = "cca.retrieval_mask";
    /// `[l * k * r, d]` encoded neighbours.
    pub const NEIGHBOUR_KV: &str = "cca.neighbour_kv";
}

/// One chunked cross-attention block.
#[derive(Debug, Clone)]
pub struct CcaBlock {
    cfg: CcaConfig,
    prefix: String,
    q_norm: NodeId,
    kv_norm: NodeId,
    q_proj: NodeId,
    k_proj: NodeId,
    v_proj: NodeId,
    o_proj: NodeId,
    gate: TimeConditionedGate,
}

impl CcaBlock {
    /// Declare one block's parameters under `prefix`.
    pub fn new(g: &mut Graph, prefix: &str, cfg: CcaConfig, activation: GateActivation) -> Self {
        let d = cfg.model_dim();
        let kv = cfg.kv_dim();
        Self {
            cfg,
            prefix: prefix.to_owned(),
            q_norm: g.parameter(&format!("{prefix}.q_norm"), &[d]),
            kv_norm: g.parameter(&format!("{prefix}.kv_norm"), &[d]),
            q_proj: g.parameter(&format!("{prefix}.q_proj"), &[d, d]),
            k_proj: g.parameter(&format!("{prefix}.k_proj"), &[d, kv]),
            v_proj: g.parameter(&format!("{prefix}.v_proj"), &[d, kv]),
            o_proj: g.parameter(&format!("{prefix}.o_proj"), &[d, d]),
            gate: TimeConditionedGate::new(g, &format!("{prefix}.gate"), d, activation),
        }
    }

    /// The gate belonging to this block.
    pub fn gate(&self) -> &TimeConditionedGate {
        &self.gate
    }

    /// Every parameter name this block owns, gate included.
    pub fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["q_norm", "kv_norm", "q_proj", "k_proj", "v_proj", "o_proj"]
            .iter()
            .map(|suffix| format!("{}.{suffix}", self.prefix))
            .collect();
        names.extend(self.gate.params().names().into_iter().map(str::to_owned));
        names
    }

    /// Apply the block to a backbone residual `h` of shape `[n, d]`.
    ///
    /// Returns `h + gate * ctx`, which at initialisation is exactly `h` —
    /// see [`crate::model::gate`] for why that requires more than zeroing the
    /// last layer's weights.
    pub fn forward(&self, g: &mut Graph, h: NodeId, inputs: CcaInputs) -> NodeId {
        let cfg = self.cfg;
        let (n, m, l, d) = (
            cfg.seq_len(),
            cfg.chunk_size(),
            cfg.num_chunks(),
            cfg.model_dim(),
        );
        let kv_rows = cfg.neighbour_kv_rows();

        // Project the encoded neighbours once for the whole sequence; the
        // per-chunk slicing happens on the result, so `l` chunks share one
        // pair of projection dispatches instead of `l`.
        let kv_all = g.rms_norm(inputs.neighbour_kv, self.kv_norm, RMS_EPS);
        let k_all = g.matmul(kv_all, self.k_proj);
        let v_all = g.matmul(kv_all, self.v_proj);

        let h_norm = g.rms_norm(h, self.q_norm, RMS_EPS);
        let q_all = g.matmul(h_norm, self.q_proj);

        let total_kv = l * kv_rows;
        let kv_width = cfg.kv_dim();
        let mut parts = Vec::with_capacity(l);
        for chunk in 0..l {
            let q = slice_rows(g, q_all, n, d, chunk * m, m);
            let k = slice_rows(g, k_all, total_kv, kv_width, chunk * kv_rows, kv_rows);
            let v = slice_rows(g, v_all, total_kv, kv_width, chunk * kv_rows, kv_rows);
            // `is_cross = true`: q_seq (m) and kv_seq (k*r) differ, and the
            // backward pass needs the cross-attention gradient path.
            let ctx = g.multi_head_attn(
                q,
                k,
                v,
                cfg.num_heads() as u32,
                cfg.num_kv_heads() as u32,
                cfg.head_dim() as u32,
                true,
            );
            let ctx = g.named(ctx, format!("{}.chunk{chunk}", self.prefix));
            parts.push((ctx, m));
        }
        let ctx = concat_rows(g, &parts, d);
        let ctx = g.matmul(ctx, self.o_proj);

        // The gate reads the *pre-block* residual, matching the sketch, so
        // that what it sees does not depend on the cross-attention output it
        // is deciding how much of to admit.
        let gate = self.gate.forward(g, h, inputs.t_col, n);

        // Chunks that retrieved nothing carry a zero-filled key/value block.
        // Attending to padding is not the same as not attending, so the mask
        // forces their contribution to zero rather than letting the gate
        // learn to ignore whatever attending to zeros happens to produce.
        let mask = g.broadcast_inner(inputs.retrieval_mask, d);
        let gate = g.mul(gate, mask);
        // Named so `Session::read_node_by_name` can recover it in a debug
        // session: a gate saturating open at low `t` during training is the
        // leak's signature, and it is only visible if the value has a name.
        let gate = g.named(gate, format!("{}.gate", self.prefix));

        let gated = g.mul(gate, ctx);
        let out = g.add(h, gated);
        g.named(out, format!("{}.out", self.prefix))
    }
}

/// Marks the frozen backbone's boundary with the trained retrieval path.
///
/// Meganeura freezes a parameter by routing it through `stop_gradient`:
/// `compile` then sees no gradient path to it and drops it from the
/// param/grad pairs the optimiser iterates. This helper exists so that the
/// retrofit's central claim — *only the encoder, the CCA blocks, and the
/// gates train* — is one call rather than a convention.
pub fn frozen_parameter(g: &mut Graph, name: &str, shape: &[usize]) -> NodeId {
    let p = g.parameter(name, shape);
    g.stop_gradient(p)
}

//! The soft, `t`-conditioned gate on the CCA block's contribution.
//!
//! # The zero-init requirement
//!
//! Anaphora is a retrofit: a pretrained masked-diffusion backbone is frozen
//! and only the neighbour encoder, the CCA blocks, and these gates train. At
//! step 0 the CCA block's output is a function of freshly initialised
//! projections — it is noise. Adding any fixed fraction of it into a frozen
//! backbone's residual stream destroys the calibration that made the backbone
//! worth retrofitting, and the run spends its first phase climbing back to
//! where it started.
//!
//! So the block must be *exactly* the identity at initialisation.
//!
//! # Why the obvious construction is not
//!
//! The design sketch zero-initialises the final linear layer of the gate MLP
//! and then applies a sigmoid:
//!
//! ```text
//! gate = Sequential(Linear(d + 1, d), SiLU(), Linear(d, 1))
//! zeros_(gate[-1].weight); zeros_(gate[-1].bias)
//! g = sigmoid(gate(cat([H, t], -1)))
//! return h + g * ctx
//! ```
//!
//! Zero weights and zero bias make the pre-activation zero, and
//! `sigmoid(0) = 0.5`. The block returns `h + 0.5 * ctx`, not `h`. Half of an
//! untrained cross-attention output goes into the frozen residual stream on
//! the first forward pass, which is the exact outcome the zero-init was
//! written to prevent.
//!
//! Pushing the bias to a large negative number instead gets `sigmoid ≈ 0`,
//! but `sigmoid'` is then also ≈ 0 and the gate is slow to open.
//!
//! # What this module does
//!
//! Both available activations are exactly zero at zero pre-activation *and*
//! have a healthy derivative there:
//!
//! * [`GateActivation::Tanh`] — `tanh(0) = 0`, `tanh'(0) = 1`. One operator,
//!   no extra parameters. The gate may go negative, which lets a block
//!   subtract retrieved context as well as add it. This is Flamingo's gated
//!   cross-attention.
//! * [`GateActivation::ScaledSigmoid`] — `sigmoid(pre) * alpha` with `alpha` a
//!   zero-initialised learned scalar. Keeps the gate proper in `[0, 1]` and
//!   still starts at exact identity, since `alpha = 0`. `d/d alpha` is
//!   `sigmoid(pre) * ctx`, which is non-zero, so `alpha` starts moving on the
//!   first step. This is ReZero/LayerScale applied to the gate.

use crate::model::rows::append_column;
use meganeura::{Graph, NodeId};

/// How the gate's scalar pre-activation becomes a multiplier.
///
/// Both variants satisfy the retrofit requirement: exact identity at
/// initialisation, non-vanishing gradient. Plain `sigmoid` is deliberately
/// not offered — see the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateActivation {
    /// `tanh(pre)`. Exact zero at init, unit derivative, signed.
    #[default]
    Tanh,
    /// `sigmoid(pre) * alpha`, `alpha` a zero-init learned scalar.
    /// Exact zero at init, non-negative.
    ScaledSigmoid,
}

/// Parameter names for one gate, so a checkpoint loader and the freezing pass
/// can find them without re-deriving the format string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateParams {
    /// `[d + 1, d]` first-layer weight.
    pub w1: String,
    /// `[d]` first-layer bias.
    pub b1: String,
    /// `[d, 1]` second-layer weight — zero-initialised.
    pub w2: String,
    /// `[1]` second-layer bias — zero-initialised.
    pub b2: String,
    /// `[1, 1]` output scale, present only for
    /// [`GateActivation::ScaledSigmoid`] — zero-initialised.
    pub alpha: Option<String>,
}

impl GateParams {
    /// Every parameter name this gate owns.
    pub fn names(&self) -> Vec<&str> {
        let mut names = vec![
            self.w1.as_str(),
            self.b1.as_str(),
            self.w2.as_str(),
            self.b2.as_str(),
        ];
        names.extend(self.alpha.as_deref());
        names
    }

    /// The names that must be initialised to exactly zero for the block to
    /// start as the identity.
    ///
    /// Meganeura initialises parameters itself, so a trainer has to write
    /// these explicitly with `Session::set_parameter` before the first step.
    /// [`zero_init_names`] is the same list for a whole model.
    pub fn zero_init_names(&self) -> Vec<&str> {
        let mut names = vec![self.w2.as_str(), self.b2.as_str()];
        names.extend(self.alpha.as_deref());
        names
    }
}

/// The `t`-conditioned gate MLP.
///
/// `[H, t] -> Linear(d+1, d) -> SiLU -> Linear(d, 1) -> activation`, then
/// broadcast across `d` so it scales the whole row.
///
/// Conditioning on `t` is what lets the gate learn the shape the hard gate
/// only approximates: retrieval is worth less when the query is mostly
/// `[MASK]`, and the gate can discover that boundary rather than have it
/// imposed at a fixed threshold.
///
/// Parameter *nodes* are created once, in [`TimeConditionedGate::new`], and
/// the ids are reused by every [`TimeConditionedGate::forward`] call.
/// Meganeura pairs parameters with gradients positionally, one per
/// `Op::Parameter` node, so re-declaring a name would allocate a second
/// buffer holding a second copy of the same logical weight — and only one of
/// them would receive what `Session::set_parameter` writes.
#[derive(Debug, Clone)]
pub struct TimeConditionedGate {
    params: GateParams,
    activation: GateActivation,
    d: usize,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
    alpha: Option<NodeId>,
}

impl TimeConditionedGate {
    /// Declare a gate's parameters under `prefix`.
    pub fn new(g: &mut Graph, prefix: &str, d: usize, activation: GateActivation) -> Self {
        let params = GateParams {
            w1: format!("{prefix}.w1"),
            b1: format!("{prefix}.b1"),
            w2: format!("{prefix}.w2"),
            b2: format!("{prefix}.b2"),
            alpha: match activation {
                GateActivation::ScaledSigmoid => Some(format!("{prefix}.alpha")),
                GateActivation::Tanh => None,
            },
        };
        let w1 = g.parameter(&params.w1, &[d + 1, d]);
        let b1 = g.parameter(&params.b1, &[d]);
        let w2 = g.parameter(&params.w2, &[d, 1]);
        let b2 = g.parameter(&params.b2, &[1]);
        let alpha = params.alpha.as_ref().map(|n| g.parameter(n, &[1, 1]));
        Self {
            params,
            activation,
            d,
            w1,
            b1,
            w2,
            b2,
            alpha,
        }
    }

    /// This gate's parameter names.
    pub fn params(&self) -> &GateParams {
        &self.params
    }

    /// The activation this gate uses.
    pub fn activation(&self) -> GateActivation {
        self.activation
    }

    /// Build the gate multiplier for `h`, shape `[rows, d]`.
    ///
    /// `t_col` is `[rows, 1]`, one noise level per row. It is a graph input
    /// rather than a constant because `t` changes every denoising step and
    /// recompiling the graph per step is not an option.
    pub fn forward(&self, g: &mut Graph, h: NodeId, t_col: NodeId, rows: usize) -> NodeId {
        let d = self.d;
        let conditioned = append_column(g, h, t_col, rows, d);

        let hidden = g.matmul(conditioned, self.w1);
        let hidden = g.bias_add(hidden, self.b1);
        let hidden = g.silu(hidden);

        let pre = g.matmul(hidden, self.w2);
        let pre = g.bias_add(pre, self.b2);

        let scalar = match self.activation {
            GateActivation::Tanh => g.tanh(pre),
            GateActivation::ScaledSigmoid => {
                let squashed = g.sigmoid(pre);
                let alpha = self.alpha.expect("ScaledSigmoid declares alpha");
                // A `[rows, 1]` broadcast of one trainable scalar: multiply a
                // constant ones column by the `[1, 1]` parameter. Meganeura's
                // `mul` is elementwise on matching shapes, so the broadcast
                // has to be explicit.
                let ones = g.constant(vec![1.0; rows], &[rows, 1]);
                let scaled = g.matmul(ones, alpha);
                g.mul(squashed, scaled)
            }
        };

        g.broadcast_inner(scalar, d)
    }
}

/// Collect the zero-init parameter names for a set of gates.
pub fn zero_init_names(gates: &[TimeConditionedGate]) -> Vec<String> {
    gates
        .iter()
        .flat_map(|gate| {
            gate.params()
                .zero_init_names()
                .into_iter()
                .map(str::to_owned)
        })
        .collect()
}

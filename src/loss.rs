//! The masked-diffusion training objective.
//!
//! LLaDA and Dream train against the negative ELBO
//!
//! ```text
//!   L = -1/(t·n) · Σ_{i masked} log p(x₀ⁱ | x_t)
//! ```
//!
//! Three things distinguish it from ordinary language-model cross-entropy,
//! and getting any of them wrong produces a model that still trains:
//!
//! * **Only masked positions score.** An unmasked position's token is already
//!   visible in the input; scoring it teaches the model to copy its own
//!   input, which it can do perfectly and which teaches nothing.
//! * **The `1/t` weight.** A step that masked 5% of the sequence sees far
//!   fewer terms than one that masked 95%, and the weight is what makes the
//!   per-step loss an unbiased estimate of the same ELBO rather than
//!   something dominated by high-noise steps.
//! * **The `1/n` normalisation is over the whole sequence**, not over the
//!   masked positions. Dividing by the masked count instead silently
//!   re-weights the noise schedule, because it cancels most of the `1/t`.
//!
//! # No new operator is needed
//!
//! Meganeura's `cross_entropy_loss` computes
//! `L = -Σ labels · log_softmax(logits)` with gradient
//! `softmax·S − labels`, where `S = Σ labels`. It generalises to arbitrary
//! per-class label *weights* rather than assuming a probability distribution
//! — it was written that way for advantage-scaled policy gradients — and the
//! masked-diffusion objective is the same shape. So:
//!
//! * a masked position's label row is `onehot(x₀ⁱ) · (1/t)`;
//! * an unmasked position's label row is **all zeros**, which contributes no
//!   loss and, since `S = 0`, no gradient;
//! * the kernel's own division by its row count supplies the `1/n`.
//!
//! # Cost
//!
//! The labels tensor is dense `[n, vocab]` f32, because that is the operator's
//! signature. At LLaDA's vocabulary that is
//! `n · 126464 · 4` bytes — 259 MB at `n = 512`, 1.04 GB at `n = 2048` —
//! uploaded every step to carry at most `n` non-zero values. [`MaskedDiffusionLoss::label_bytes`]
//! computes it so a configuration can be rejected before a run discovers it.
//! An indexed-label cross-entropy in Meganeura would reduce this to
//! `n · 8` bytes; until then, keep Phase 1 vocabularies small.

use crate::schedule::NoiseLevel;
use crate::view::{CleanSequence, NoisedView, SequenceId};

/// Why labels could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelError {
    /// The view and the clean sequence are different lengths.
    LengthMismatch { view: usize, clean: usize },
    /// The view was not masked from this clean sequence.
    ///
    /// Scoring against the wrong answers is as silent as the retrieval leak
    /// and looks like nothing but a model that will not converge.
    SequenceMismatch {
        /// The sequence the view came from, if any.
        view_source: Option<SequenceId>,
        /// The sequence whose targets were offered.
        offered: SequenceId,
    },
    /// A target token id is outside the vocabulary.
    TokenOutOfRange {
        position: usize,
        token: u32,
        vocab: usize,
    },
}

impl std::fmt::Display for LabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LengthMismatch { view, clean } => {
                write!(f, "view has {view} tokens, clean sequence has {clean}")
            }
            Self::SequenceMismatch {
                view_source,
                offered,
            } => match view_source {
                Some(src) => write!(
                    f,
                    "view was masked from sequence {} but targets came from {}",
                    src.get(),
                    offered.get()
                ),
                None => write!(
                    f,
                    "view has no clean source (it was built for sampling) but targets from \
                     sequence {} were offered",
                    offered.get()
                ),
            },
            Self::TokenOutOfRange {
                position,
                token,
                vocab,
            } => write!(
                f,
                "target token {token} at position {position} is outside vocabulary {vocab}"
            ),
        }
    }
}

impl std::error::Error for LabelError {}

/// What one label tensor ended up containing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelStats {
    /// Positions that scored.
    pub scored: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// The `1/t` weight applied to each scoring row.
    pub weight: f32,
}

impl LabelStats {
    /// Whether this step contributes any gradient at all.
    ///
    /// A step can legitimately mask nothing — at `t` near zero that is the
    /// common case — and its loss and gradient are then exactly zero. That is
    /// correct, and it is also a wasted forward and backward pass, so a
    /// training loop is better off skipping the step than submitting it.
    pub fn contributes(self) -> bool {
        self.scored > 0
    }
}

/// Builds `[n, vocab]` label tensors for the masked-diffusion objective.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskedDiffusionLoss {
    vocab_size: usize,
    min_t: f32,
}

impl MaskedDiffusionLoss {
    /// Default `t` floor: `1e-3`.
    ///
    /// The `1/t` weight diverges as `t → 0`. A floor bounds it at `1000`,
    /// which keeps one unlucky low-`t` sample from dominating a batch.
    /// Clamping rather than rejecting matters because `t` is sampled, so a
    /// tiny value is a legitimate draw, not a caller error.
    pub const DEFAULT_MIN_T: f32 = 1e-3;

    /// A loss builder over a `vocab_size`-wide vocabulary.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            min_t: Self::DEFAULT_MIN_T,
        }
    }

    /// Override the `t` floor.
    pub fn with_min_t(self, min_t: f32) -> Self {
        Self {
            min_t: min_t.max(f32::MIN_POSITIVE),
            ..self
        }
    }

    /// Vocabulary width.
    pub fn vocab_size(self) -> usize {
        self.vocab_size
    }

    /// The `1/t` weight this builder would apply at noise level `t`.
    pub fn weight(self, t: NoiseLevel) -> f32 {
        1.0 / t.get().max(self.min_t)
    }

    /// Bytes one dense label tensor occupies for a sequence of `seq_len`.
    ///
    /// See the module header: this is uploaded every step to carry at most
    /// `seq_len` non-zero values.
    pub fn label_bytes(self, seq_len: usize) -> usize {
        seq_len * self.vocab_size * size_of::<f32>()
    }

    /// Write the label tensor for `view` into `out`, resizing it to
    /// `seq_len * vocab_size`.
    ///
    /// `out` is reused across steps rather than allocated per step; at these
    /// sizes the allocation is the expensive part.
    pub fn build_labels(
        self,
        view: &NoisedView,
        clean: &CleanSequence,
        out: &mut Vec<f32>,
    ) -> Result<LabelStats, LabelError> {
        if view.len() != clean.len() {
            return Err(LabelError::LengthMismatch {
                view: view.len(),
                clean: clean.len(),
            });
        }
        if view.source() != Some(clean.id()) {
            return Err(LabelError::SequenceMismatch {
                view_source: view.source(),
                offered: clean.id(),
            });
        }

        let weight = self.weight(view.noise_level());
        let targets = clean.targets();

        // Validate before writing, so a rejected call leaves `out` untouched
        // rather than half-filled with a tensor the caller might still upload.
        for (position, &token) in targets.iter().enumerate() {
            if view.masked()[position] && token as usize >= self.vocab_size {
                return Err(LabelError::TokenOutOfRange {
                    position,
                    token,
                    vocab: self.vocab_size,
                });
            }
        }

        out.clear();
        out.resize(view.len() * self.vocab_size, 0.0);
        let mut scored = 0;
        for (position, &token) in targets.iter().enumerate() {
            if view.masked()[position] {
                out[position * self.vocab_size + token as usize] = weight;
                scored += 1;
            }
        }

        Ok(LabelStats {
            scored,
            seq_len: view.len(),
            weight,
        })
    }

    /// The loss Meganeura's kernel will report for `logits`, computed on the
    /// CPU from the same labels.
    ///
    /// This is the reference the GPU path is checked against. It is also
    /// useful on its own for evaluation, where a forward pass is wanted
    /// without a training step.
    pub fn reference_loss(self, logits: &[f32], labels: &[f32], seq_len: usize) -> f32 {
        let vocab = self.vocab_size;
        let mut total = 0.0f64;
        for row in 0..seq_len {
            let start = row * vocab;
            let logit_row = &logits[start..start + vocab];
            let label_row = &labels[start..start + vocab];
            // Skip rows that cannot contribute: an unmasked position's row is
            // all zeros, and that is most of the tensor.
            if label_row.iter().all(|&w| w == 0.0) {
                continue;
            }
            let max = logit_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = logit_row.iter().map(|&v| ((v - max) as f64).exp()).sum();
            let log_sum_exp = sum_exp.ln() + max as f64;
            for (j, &w) in label_row.iter().enumerate() {
                if w != 0.0 {
                    total -= w as f64 * (logit_row[j] as f64 - log_sum_exp);
                }
            }
        }
        (total / seq_len as f64) as f32
    }
}

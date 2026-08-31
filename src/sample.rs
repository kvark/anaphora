//! Inference: retrieval inside the denoising loop.
//!
//! Section 4 of the design sketch, and the part autoregressive RETRO cannot
//! do. An AR model's context only grows, so retrieval can happen once, up
//! front. A diffusion model's context *sharpens*: each step yields a cleaner
//! view of the whole sequence, so the query can be re-issued against a better
//! sketch — early steps on a rough semantic gist, later steps on something
//! close to the final text.
//!
//! The cost of that is index traffic, and the control here is the refresh
//! schedule: re-query at a few thresholds and cache the encoded neighbours in
//! between. Re-querying every step makes traffic scale with `steps` and puts
//! NVMe on the critical path.
//!
//! # This sampler is not the ELBO's reverse process
//!
//! The principled reverse process for masked diffusion unmasks each position
//! *independently at random*, with probability `(α_s - α_t) / (1 - α_t)`.
//! What this module does instead is reveal the positions the model is most
//! confident about, which is what LLaDA does and what works better in
//! practice, but it is a different sampler.
//!
//! The consequence is about reporting, not correctness: numbers produced by
//! running this loop are not likelihood bounds, and should not be compared
//! against a published diffusion perplexity as though they were. The
//! evaluation protocol in [`crate::eval`] sidesteps this by scoring the
//! objective directly at fixed noise levels rather than by sampling.

use crate::chunk::{ChunkAdmission, ChunkedView, RetrieverEncode, chunk_queries};
use crate::config::CcaConfig;
use crate::retrieval::corpus::{DocumentId, NeighbourCorpus};
use crate::retrieval::index::NeighbourIndex;
use crate::retrieval::leakage::LeakageGuard;
use crate::retrieval::{Neighbours, retrieve};
use crate::schedule::{NoiseLevel, Phase, RefreshSchedule, trajectory};
use crate::view::{MaskToken, NoisedView, check_same_view};

/// Runs the model forward for one denoising step.
///
/// A trait rather than a concrete `meganeura::Session` call so the loop's
/// control flow — when to refresh, what to cache, what to reveal — is
/// testable without a GPU, and so the same driver serves a cached-KV
/// inference session and an in-graph-encoder debugging session.
pub trait Denoiser {
    /// Vocabulary width of the returned logits.
    fn vocab_size(&self) -> usize;

    /// Logits for every position, row-major `[n, vocab]`.
    ///
    /// `neighbours` is `None` when the hard gate closed at this `t`. An
    /// implementation must then feed a zero retrieval mask, not a stale
    /// neighbour block.
    fn logits(&mut self, view: &NoisedView, neighbours: Option<&Neighbours>) -> Vec<f32>;
}

/// How many positions a step reveals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RevealPolicy {
    /// Reveal `remaining_masked / remaining_steps`, rounded up, so the
    /// trajectory finishes with everything revealed regardless of where
    /// rounding lands.
    #[default]
    Linear,
    /// Reveal a fixed count per step.
    Fixed(usize),
}

impl RevealPolicy {
    /// Positions to reveal at a step with `remaining_steps` left (this one
    /// included) and `remaining_masked` still masked.
    pub fn count(self, remaining_steps: usize, remaining_masked: usize) -> usize {
        match self {
            Self::Linear => {
                if remaining_steps <= 1 {
                    remaining_masked
                } else {
                    remaining_masked.div_ceil(remaining_steps)
                }
            }
            Self::Fixed(k) => k.min(remaining_masked),
        }
    }
}

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Denoising steps.
    pub steps: usize,
    /// When to re-query the index.
    pub refresh: RefreshSchedule,
    /// Per-chunk admission for query construction.
    pub admission: ChunkAdmission,
    /// How many positions to reveal per step.
    pub reveal: RevealPolicy,
    /// Document id the prompt belongs to, for the leakage guard. At inference
    /// there is no held-out answer, but a corpus that contains the prompt's
    /// own source still returns it verbatim.
    pub document: DocumentId,
}

impl SamplingConfig {
    /// The sketch's defaults: 32 steps, refreshing at `t = 0.8, 0.5, 0.25`.
    pub fn new(document: DocumentId) -> Self {
        Self {
            steps: 32,
            refresh: RefreshSchedule::default_thresholds(),
            admission: ChunkAdmission::default(),
            reveal: RevealPolicy::Linear,
            document,
        }
    }
}

/// What one trajectory did, for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SamplingTrace {
    /// Steps run.
    pub steps: usize,
    /// Steps that re-queried the index.
    pub refreshes: usize,
    /// Steps that ran with no neighbours because the hard gate was closed.
    pub gated_out: usize,
    /// Positions revealed in total.
    pub revealed: usize,
}

/// Everything the sampler needs to answer a query.
pub struct RetrievalContext<'a, I: NeighbourIndex, E: RetrieverEncode> {
    /// The nearest-neighbour index.
    pub index: &'a I,
    /// Token storage behind the index.
    pub corpus: &'a NeighbourCorpus,
    /// Which neighbours this document may retrieve.
    pub guard: &'a LeakageGuard,
    /// Turns chunks of the noised view into query vectors.
    pub encoder: &'a mut E,
}

/// Run one denoising trajectory.
///
/// Returns the final view and a trace. `prompt` supplies the positions that
/// start revealed; the rest of the `n` positions start masked.
pub fn sample<I: NeighbourIndex, E: RetrieverEncode, D: Denoiser>(
    prompt: &[u32],
    mask_token: MaskToken,
    cfg: CcaConfig,
    sampling: &mut SamplingConfig,
    retrieval: &mut RetrievalContext<'_, I, E>,
    denoiser: &mut D,
) -> (NoisedView, SamplingTrace) {
    let n = cfg.seq_len();
    assert!(
        prompt.len() <= n,
        "prompt of {} tokens exceeds the sequence length {n}",
        prompt.len()
    );

    let mut tokens = vec![mask_token.0; n];
    tokens[..prompt.len()].copy_from_slice(prompt);
    let mut view = NoisedView::from_tokens(tokens, NoiseLevel::MASKED, mask_token);

    sampling.refresh.reset();
    let mut cached: Option<Neighbours> = None;
    let mut trace = SamplingTrace::default();
    let steps = trajectory(sampling.steps);

    for (i, &t) in steps.iter().enumerate() {
        // The view carries the noise level the denoiser and the gate read, so
        // it advances before either runs.
        view = view.reveal(&[], t);

        // Between refreshes `cached` is reused as-is. It came from an
        // earlier, noisier view, which is the whole point of caching — but it
        // also means the `ViewId` check cannot apply to a cached block. The
        // staleness is the refresh schedule's decision, made here, rather than
        // something the type system lets pass unnoticed elsewhere.
        if sampling.refresh.advance(t) {
            cached = refresh(&view, cfg, sampling, retrieval);
            if cached.is_some() {
                trace.refreshes += 1;
            }
        }

        if cached.is_none() {
            trace.gated_out += 1;
        }

        let logits = denoiser.logits(&view, cached.as_ref());
        let vocab = denoiser.vocab_size();
        debug_assert_eq!(logits.len(), n * vocab, "logits must be [n, vocab]");

        let remaining_masked = view.num_masked();
        let to_reveal = sampling
            .reveal
            .count(steps.len() - i, remaining_masked)
            .min(remaining_masked);
        let picks = unmask_top_confidence(&view, &logits, vocab, to_reveal);
        trace.revealed += picks.len();
        view = view.reveal(&picks, t);
        trace.steps += 1;
    }

    (view, trace)
}

fn refresh<I: NeighbourIndex, E: RetrieverEncode>(
    view: &NoisedView,
    cfg: CcaConfig,
    sampling: &SamplingConfig,
    retrieval: &mut RetrievalContext<'_, I, E>,
) -> Option<Neighbours> {
    let chunked = ChunkedView::new(view, cfg).ok()?;
    let queries = chunk_queries(
        chunked,
        Phase::Inference,
        sampling.admission,
        retrieval.encoder,
    )?;
    let neighbours = retrieve(
        &queries,
        cfg,
        retrieval.index,
        retrieval.corpus,
        retrieval.guard,
        sampling.document,
    );
    // Cheap, and it catches a query/gather pair that drifted apart.
    check_same_view(view, &neighbours).expect("retrieval ran against the view it was handed");
    Some(neighbours)
}

/// Pick the `count` masked positions the model is most confident about, and
/// the token it would put there.
///
/// Confidence is the maximum softmax probability of the row, computed in a
/// numerically stable way. Comparing raw maximum logits instead would rank
/// positions by their rows' arbitrary offsets rather than by how peaked the
/// distributions are, and the offsets differ per position.
///
/// # `[MASK]` is not a legal output
///
/// The mask token has an id like any other, and an unconstrained `argmax`
/// will happily choose it. That is a hard failure rather than a bad sample:
/// the forward process never re-masks an unmasked position, so a `[MASK]`
/// written into the sequence is stuck there for the rest of the trajectory
/// with no way for the model to correct it.
///
/// It also corrupts the ranking. Confidence is the row's peak probability, so
/// a position where the model is *sure* the answer is `[MASK]` scores as
/// maximally confident and gets revealed first — the sampler would
/// preferentially commit exactly the positions it has nothing to say about.
///
/// So the mask id is excluded from both the `argmax` and the softmax
/// denominator. This is the sampling-time half of MDLM's zero-masking-
/// probability parameterisation; see the note on the training half in
/// [`crate::loss`].
pub fn unmask_top_confidence(
    view: &NoisedView,
    logits: &[f32],
    vocab: usize,
    count: usize,
) -> Vec<(usize, u32)> {
    if count == 0 || vocab == 0 {
        return Vec::new();
    }
    let mask_id = view.mask_token().0 as usize;
    let mut scored: Vec<(usize, u32, f32)> = view
        .masked()
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m)
        .filter_map(|(pos, _)| {
            let row = logits.get(pos * vocab..(pos + 1) * vocab)?;
            let mut arg = None;
            let mut max = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if i != mask_id && v > max {
                    max = v;
                    arg = Some(i);
                }
            }
            let arg = arg?;
            if !max.is_finite() {
                return None;
            }
            // Renormalise over the same restricted support, so the confidence
            // is a probability under the distribution actually being sampled
            // from rather than one that reserves mass for an illegal token.
            let sum_exp: f32 = row
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != mask_id)
                .map(|(_, &v)| (v - max).exp())
                .sum();
            Some((pos, arg as u32, 1.0 / sum_exp))
        })
        .collect();

    scored.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Ties resolve by position so a trajectory is reproducible.
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(count);
    scored.sort_by_key(|&(pos, _, _)| pos);
    scored.into_iter().map(|(pos, tok, _)| (pos, tok)).collect()
}

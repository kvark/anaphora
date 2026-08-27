//! The noised view, and the wall between it and the clean sequence.
//!
//! Section 1 of the design sketch, which calls query construction "the whole
//! ballgame". The rule it states is short:
//!
//! > The retriever may only see what the denoiser sees.
//!
//! Query `x_0` during training and the retrieved neighbours correlate with
//! exactly the tokens that were masked. The loss collapses into
//! copy-from-neighbour and the experiment measures nothing. The failure is
//! silent — perplexity *improves*.
//!
//! A comment cannot enforce that. This module does, in two layers:
//!
//! 1. [`CleanSequence`] exposes no method that yields queries. The only type
//!    the query builder accepts is [`NoisedView`].
//! 2. Every [`NoisedView`] carries a [`ViewId`], and everything derived from
//!    it carries that id forward. The denoiser checks that the neighbours it
//!    was handed came from the view it is denoising, so retrieving from a
//!    *different, cleaner* view of the same sequence is caught too — which
//!    layer 1 alone would not catch.

use crate::schedule::NoiseLevel;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies one noised view.
///
/// Ids are unique per process and are never reused, so an id match means the
/// two values really came from the same masking event, not merely from views
/// that happen to agree on `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ViewId(u64);

impl ViewId {
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw id, for logging.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// The token id standing in for a masked position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskToken(pub u32);

/// `x_0` — the clean sequence.
///
/// Deliberately inert. It is a denoising *target* and nothing else: there is
/// no accessor here that hands its tokens to the retrieval path, and there
/// should never be one. To read it for a loss, use [`CleanSequence::targets`],
/// which is what the loss builder consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanSequence {
    tokens: Vec<u32>,
}

impl CleanSequence {
    /// Wrap a clean token sequence.
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    /// Sequence length.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The denoising targets.
    ///
    /// The one way out of this type, and it exists for the loss. Do not route
    /// the result into query construction — that is precisely the failure
    /// this module is built to prevent.
    pub fn targets(&self) -> &[u32] {
        &self.tokens
    }

    /// Mask this sequence at noise level `t`, producing the view the denoiser
    /// and the retriever will *both* read.
    ///
    /// `should_mask` decides each position, so the caller owns the corruption
    /// process (uniform, block, span) while this constructor owns the
    /// bookkeeping that keeps the two consumers in sync.
    pub fn mask_with(
        &self,
        t: NoiseLevel,
        mask_token: MaskToken,
        mut should_mask: impl FnMut(usize) -> bool,
    ) -> NoisedView {
        let mut tokens = self.tokens.clone();
        let mut masked = vec![false; tokens.len()];
        for (i, slot) in tokens.iter_mut().enumerate() {
            if should_mask(i) {
                *slot = mask_token.0;
                masked[i] = true;
            }
        }
        NoisedView {
            id: ViewId::next(),
            tokens,
            masked,
            t,
            mask_token,
        }
    }
}

/// `x_t` — the partially masked view at noise level `t`.
///
/// This is the *only* thing the retriever is allowed to read. It is also what
/// the denoiser reads, which is the whole point: the two see the same thing
/// by construction rather than by discipline.
#[derive(Debug, Clone, PartialEq)]
pub struct NoisedView {
    id: ViewId,
    tokens: Vec<u32>,
    masked: Vec<bool>,
    t: NoiseLevel,
    mask_token: MaskToken,
}

impl NoisedView {
    /// Build a view directly from already-masked tokens.
    ///
    /// For inference, where there is no clean sequence to mask — sampling
    /// starts from an all-`[MASK]` canvas and reveals tokens into it.
    pub fn from_tokens(tokens: Vec<u32>, t: NoiseLevel, mask_token: MaskToken) -> Self {
        let masked = tokens.iter().map(|&tok| tok == mask_token.0).collect();
        Self {
            id: ViewId::next(),
            tokens,
            masked,
            t,
            mask_token,
        }
    }

    /// An all-`[MASK]` canvas of length `n`, the starting point for sampling.
    pub fn all_masked(n: usize, mask_token: MaskToken) -> Self {
        Self {
            id: ViewId::next(),
            tokens: vec![mask_token.0; n],
            masked: vec![true; n],
            t: NoiseLevel::MASKED,
            mask_token,
        }
    }

    /// This view's identity.
    pub fn id(&self) -> ViewId {
        self.id
    }

    /// The visible tokens, masked positions included as `[MASK]`.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Per-position mask flags.
    pub fn masked(&self) -> &[bool] {
        &self.masked
    }

    /// The noise level this view was produced at.
    pub fn noise_level(&self) -> NoiseLevel {
        self.t
    }

    /// The mask token id.
    pub fn mask_token(&self) -> MaskToken {
        self.mask_token
    }

    /// Sequence length.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Count of masked positions.
    pub fn num_masked(&self) -> usize {
        self.masked.iter().filter(|&&m| m).count()
    }

    /// Fraction of positions that are masked.
    ///
    /// This is the *empirical* mask rate, which is not the same number as
    /// `t`: `t` parameterises the corruption process, this counts what the
    /// process actually did. They diverge for small `n` or a non-uniform
    /// masking rule, and it is the empirical rate that determines whether a
    /// query embedding has any signal in it.
    pub fn mask_rate(&self) -> f32 {
        if self.tokens.is_empty() {
            return 0.0;
        }
        self.num_masked() as f32 / self.tokens.len() as f32
    }

    /// Reveal `position` as `token`, as one denoising step does.
    ///
    /// Revealing produces a *new* view with a new [`ViewId`]: the sequence
    /// has changed, so neighbours retrieved against the previous view are now
    /// stale. Whether that staleness is acceptable is the refresh schedule's
    /// call ([`crate::schedule::RefreshSchedule`]) — but it must be a
    /// decision, not an accident, which is why this does not mutate in place.
    pub fn reveal(&self, positions: &[(usize, u32)], t: NoiseLevel) -> Self {
        let mut tokens = self.tokens.clone();
        let mut masked = self.masked.clone();
        for &(pos, tok) in positions {
            tokens[pos] = tok;
            masked[pos] = false;
        }
        Self {
            id: ViewId::next(),
            tokens,
            masked,
            t,
            mask_token: self.mask_token,
        }
    }
}

/// A value derived from one specific [`NoisedView`].
///
/// Carrying the id forward is what lets the denoiser reject neighbours that
/// were retrieved against a different view — the cleaner-query leak that
/// [`CleanSequence`]'s inertness alone does not cover.
pub trait DerivedFromView {
    /// The view this value was derived from.
    fn view_id(&self) -> ViewId;
}

/// Raised when a value derived from one view is used to denoise another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewMismatch {
    /// The view the denoiser is working on.
    pub denoising: ViewId,
    /// The view the offending value came from.
    pub derived_from: ViewId,
}

impl std::fmt::Display for ViewMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "retrieval came from view {} but the denoiser is on view {}: \
             the retriever must see exactly what the denoiser sees",
            self.derived_from.get(),
            self.denoising.get()
        )
    }
}

impl std::error::Error for ViewMismatch {}

/// Check that `derived` came from `view`.
pub fn check_same_view<D: DerivedFromView>(
    view: &NoisedView,
    derived: &D,
) -> Result<(), ViewMismatch> {
    if derived.view_id() == view.id() {
        Ok(())
    } else {
        Err(ViewMismatch {
            denoising: view.id(),
            derived_from: derived.view_id(),
        })
    }
}

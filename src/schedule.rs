//! Noise level, the retrieval hard gate, and the refresh schedule.
//!
//! Section 3 of the design sketch. The gate is asymmetric between training
//! and inference, and the asymmetry is not an optimization — it is what keeps
//! the low-noise band from turning into a copying task during training while
//! still using it at inference, where it helps most.

/// A masked-diffusion noise level, `t ∈ [0, 1]`.
///
/// `t = 1` is fully masked, `t = 0` is clean. Values outside the unit
/// interval and non-finite values are rejected at construction, so every
/// consumer downstream can treat the invariant as established.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NoiseLevel(f32);

impl NoiseLevel {
    /// Fully masked.
    pub const MASKED: Self = Self(1.0);
    /// Fully denoised.
    pub const CLEAN: Self = Self(0.0);

    /// Construct a noise level, rejecting anything outside `[0, 1]`.
    pub fn new(t: f32) -> Option<Self> {
        (t.is_finite() && (0.0..=1.0).contains(&t)).then_some(Self(t))
    }

    /// Construct a noise level, clamping into `[0, 1]`.
    ///
    /// Non-finite input clamps to [`Self::MASKED`], which is the
    /// conservative direction: a broken schedule skips retrieval rather than
    /// retrieving on a garbage query.
    pub fn saturating(t: f32) -> Self {
        if t.is_nan() {
            Self::MASKED
        } else {
            Self(t.clamp(0.0, 1.0))
        }
    }

    /// The underlying value.
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Which side of the training/inference asymmetry a call is on.
///
/// This is a required argument rather than a defaulted flag on purpose: the
/// two phases gate on different bands, and picking the wrong one is a silent
/// failure in one direction (training on a leaky band) and a quiet loss of
/// quality in the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Denoising loss is being computed and the retriever can leak answers.
    Training,
    /// Sampling. Nothing to leak — the tokens being revealed are the output.
    Inference,
}

/// The band of noise levels in which retrieval is allowed to run.
///
/// Both ends exist for distinct reasons, and only one of them survives into
/// inference:
///
/// * **High `t`** — the query is nearly all `[MASK]`. Its embedding carries
///   no signal, so retrieval returns noise. Skipped in both phases.
/// * **Low `t`** — during *training* the query is nearly clean, so a
///   neighbour's continuation can contain the handful of tokens still masked,
///   and the low-`t` loss goes trivial. Skipped. At *inference* there is no
///   held-out answer to leak, and this is the band where a retrieved
///   continuation is most useful, so it is kept.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetrievalBand {
    low: f32,
    high: f32,
}

impl RetrievalBand {
    /// The sketch's defaults: `0.15 < t < 0.85` training, `t < 0.9` inference.
    pub const DEFAULT_TRAINING: Self = Self {
        low: 0.15,
        high: 0.85,
    };
    /// Inference keeps the low end: only the noise-dominated top is skipped.
    pub const DEFAULT_INFERENCE: Self = Self {
        low: 0.0,
        high: 0.9,
    };

    /// A custom band, `low < t < high`.
    pub fn new(low: f32, high: f32) -> Option<Self> {
        (low.is_finite() && high.is_finite() && low < high).then_some(Self { low, high })
    }

    /// The band that applies in `phase`.
    pub fn for_phase(phase: Phase) -> Self {
        match phase {
            Phase::Training => Self::DEFAULT_TRAINING,
            Phase::Inference => Self::DEFAULT_INFERENCE,
        }
    }

    /// Whether `t` falls inside the band.
    pub fn admits(self, t: NoiseLevel) -> bool {
        self.low < t.get() && t.get() < self.high
    }
}

/// The hard gate: may retrieval run at this noise level, in this phase?
///
/// The soft, learned gate in [`crate::model::gate`] can eventually subsume
/// this — it can learn to output zero wherever retrieval does not help. The
/// hard gate is what makes the *training* run safe before that happens, and
/// it also saves the index traffic outright.
pub fn retrieve_now(t: NoiseLevel, phase: Phase) -> bool {
    RetrievalBand::for_phase(phase).admits(t)
}

/// Tracks which refresh thresholds a descending `t` has crossed.
///
/// Encoding neighbours is the expensive half of retrieval. Re-querying every
/// denoising step makes index traffic scale with `steps` and puts NVMe on the
/// critical path; refreshing at a few thresholds and caching the encoded
/// neighbours between them does not.
///
/// The schedule is stateful because "crosses" is a claim about a *transition*.
/// A stateless predicate on `t` alone either fires on every step below a
/// threshold or needs the caller to remember the previous `t` — and a caller
/// that forgets silently degrades into never refreshing.
#[derive(Debug, Clone)]
pub struct RefreshSchedule {
    /// Descending thresholds.
    thresholds: Vec<f32>,
    /// Index of the next threshold to fire.
    next: usize,
    /// Set once the first step has been observed.
    started: bool,
}

impl RefreshSchedule {
    /// The sketch's defaults: refresh at `t = 0.8`, `0.5`, `0.25`.
    pub fn default_thresholds() -> Self {
        Self::new(&[0.8, 0.5, 0.25])
    }

    /// Build a schedule from a set of thresholds.
    ///
    /// Order does not matter — they are sorted descending, to match a `t`
    /// that starts at 1.0 and falls. Duplicates and out-of-range values are
    /// dropped.
    pub fn new(thresholds: &[f32]) -> Self {
        let mut thresholds: Vec<f32> = thresholds
            .iter()
            .copied()
            .filter(|t| t.is_finite() && (0.0..=1.0).contains(t))
            .collect();
        thresholds.sort_by(|a, b| b.partial_cmp(a).expect("filtered to finite"));
        thresholds.dedup();
        Self {
            thresholds,
            next: 0,
            started: false,
        }
    }

    /// Advance to `t` and report whether a refresh threshold was crossed.
    ///
    /// The first call always refreshes: there is nothing cached yet, and a
    /// first step that starts below the top threshold would otherwise run the
    /// whole trajectory with no neighbours at all.
    ///
    /// If several thresholds fall between the previous `t` and this one — a
    /// coarse step count over a dense schedule — they collapse into a single
    /// refresh, because a refresh recomputes from the current `t` regardless
    /// of how many thresholds it skipped.
    pub fn advance(&mut self, t: NoiseLevel) -> bool {
        let first = !self.started;
        self.started = true;
        let mut crossed = false;
        while self.next < self.thresholds.len() && t.get() <= self.thresholds[self.next] {
            self.next += 1;
            crossed = true;
        }
        crossed || first
    }

    /// Rewind to the start of the trajectory, for the next sample.
    pub fn reset(&mut self) {
        self.next = 0;
        self.started = false;
    }

    /// Thresholds not yet crossed.
    pub fn remaining(&self) -> &[f32] {
        &self.thresholds[self.next..]
    }
}

/// The descending noise levels visited by a `steps`-step denoising loop.
///
/// `linspace(1.0, 0.0, steps)` in the sketch. A single-step schedule yields
/// only `t = 1.0`, matching `linspace`'s degenerate case.
pub fn trajectory(steps: usize) -> Vec<NoiseLevel> {
    match steps {
        0 => Vec::new(),
        1 => vec![NoiseLevel::MASKED],
        _ => (0..steps)
            .map(|i| {
                let frac = i as f32 / (steps - 1) as f32;
                NoiseLevel::saturating(1.0 - frac)
            })
            .collect(),
    }
}

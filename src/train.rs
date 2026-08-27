//! The host training loop.
//!
//! Meganeura's own `Trainer` and `DataLoader` stream `Vec<f32>` pairs, and
//! this graph takes two U32 inputs — `token_ids` and `cca.neighbour_tokens` —
//! alongside its f32 ones. So the loop drives [`meganeura::runtime::Session`]
//! directly: sample `t`, mask, retrieve, bind, step.
//!
//! Each step is one sequence. Meganeura's attention operators are
//! two-dimensional, so a graph describes one sequence and batching means
//! running the session repeatedly — see `docs/roadmap.md`.

use crate::chunk::{ChunkAdmission, ChunkedView, chunk_queries};
use crate::config::CcaConfig;
use crate::corpus::{ChunkEmbedder, TrainingSequence};
use crate::loss::MaskedDiffusionLoss;
use crate::model::cca::input_names;
use crate::model::{CcaModel, NeighbourInput};
use crate::retrieval::corpus::NeighbourCorpus;
use crate::retrieval::index::NeighbourIndex;
use crate::retrieval::leakage::LeakageGuard;
use crate::retrieval::{Neighbours, retrieve};
use crate::schedule::{NoiseLevel, Phase};
use crate::view::{CleanSequence, MaskToken, NoisedView};
use meganeura::runtime::Session;

/// A small deterministic generator.
///
/// SplitMix64. A research run has to be reproducible from a seed, and this is
/// the whole requirement — pulling a general-purpose RNG crate in for one
/// mixing function would be the larger change.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

/// How the noise level is drawn each step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NoiseSampler {
    /// `t ~ U(0, 1]`, as LLaDA trains.
    #[default]
    Uniform,
    /// A fixed level. For diagnostics and for the by-`t` evaluation bands,
    /// where averaging over a sampled `t` is exactly what hides the effect
    /// being measured.
    Fixed(f32),
}

impl NoiseSampler {
    /// Draw a noise level.
    pub fn sample(self, rng: &mut Rng) -> NoiseLevel {
        match self {
            // Excluding zero: at t = 0 nothing is masked, the step scores
            // nothing, and the 1/t weight is at its floor for no reason.
            Self::Uniform => NoiseLevel::saturating(1.0 - rng.next_f32()),
            Self::Fixed(t) => NoiseLevel::saturating(t),
        }
    }
}

/// Training hyper-parameters that are not the model's shape.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Retrieval shapes; must match the model's.
    pub cca: CcaConfig,
    /// Vocabulary, for the label tensor.
    pub vocab_size: usize,
    /// The `[MASK]` token.
    pub mask_token: MaskToken,
    /// How `t` is drawn.
    pub noise: NoiseSampler,
    /// Per-chunk retrieval admission.
    pub admission: ChunkAdmission,
    /// Whether retrieval runs at all.
    ///
    /// `false` gives the no-retrieval baseline the evaluation protocol
    /// compares against, without building a second model.
    pub retrieval_enabled: bool,
}

impl TrainingConfig {
    /// Defaults for a `cca` shape over a `vocab_size` vocabulary.
    pub fn new(cca: CcaConfig, vocab_size: usize, mask_token: MaskToken) -> Self {
        Self {
            cca,
            vocab_size,
            mask_token,
            noise: NoiseSampler::Uniform,
            admission: ChunkAdmission::default(),
            retrieval_enabled: true,
        }
    }
}

/// What one training step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    /// The objective, as Meganeura reported it.
    pub loss: f32,
    /// The noise level drawn.
    pub noise_level: NoiseLevel,
    /// Positions that scored.
    pub scored: usize,
    /// Chunks that retrieved at least one neighbour.
    pub chunks_retrieved: usize,
    /// Whether the hard gate admitted retrieval at all.
    pub gate_open: bool,
}

/// Everything retrieval needs, borrowed for the duration of a step.
pub struct RetrievalSources<'a, I: NeighbourIndex, E: ChunkEmbedder> {
    /// The nearest-neighbour index.
    pub index: &'a I,
    /// Token storage behind it.
    pub corpus: &'a NeighbourCorpus,
    /// Which neighbours a document may retrieve.
    pub guard: &'a LeakageGuard,
    /// Embeds chunks of the noised view into queries. The same type that
    /// embedded the corpus — see [`ChunkEmbedder`].
    pub embedder: &'a mut E,
}

/// Drives training over a built session.
pub struct Trainer {
    cfg: TrainingConfig,
    loss: MaskedDiffusionLoss,
    rng: Rng,
    /// Reused across steps; at `[n, vocab]` the allocation is the expensive
    /// part.
    labels: Vec<f32>,
    neighbour_tokens: Vec<u32>,
    retrieval_mask: Vec<f32>,
    t_col: Vec<f32>,
    steps: usize,
}

/// Why a step could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepError {
    /// The sequence length does not match the configuration.
    LengthMismatch { got: usize, expected: usize },
    /// The model was built with cached neighbours, so there is no
    /// `cca.neighbour_tokens` input to bind and the encoder is not in the
    /// graph to receive gradients.
    NeighboursNotEncoded,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::LengthMismatch { got, expected } => {
                write!(
                    f,
                    "sequence has {got} tokens, configuration says {expected}"
                )
            }
            Self::NeighboursNotEncoded => write!(
                f,
                "training needs NeighbourInput::Encoded so the neighbour encoder \
                 is in the graph and receives gradients"
            ),
        }
    }
}

impl std::error::Error for StepError {}

impl Trainer {
    /// A trainer for `cfg`, seeded for reproducibility.
    pub fn new(cfg: TrainingConfig, seed: u64) -> Self {
        let loss = MaskedDiffusionLoss::new(cfg.vocab_size);
        let n = cfg.cca.seq_len();
        let kv_rows = cfg.cca.num_chunks() * cfg.cca.neighbour_kv_rows();
        Self {
            loss,
            rng: Rng::new(seed),
            labels: Vec::with_capacity(n * cfg.vocab_size),
            neighbour_tokens: vec![cfg.mask_token.0; kv_rows],
            retrieval_mask: vec![0.0; n],
            t_col: vec![0.0; n],
            cfg,
            steps: 0,
        }
    }

    /// Steps taken.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The generator, for callers that want to draw from the same stream.
    pub fn rng(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// Mask a training sequence at a freshly drawn noise level.
    ///
    /// Padding is never masked, so it never scores.
    pub fn corrupt(&mut self, seq: &TrainingSequence) -> (CleanSequence, NoisedView) {
        let t = self.cfg.noise.sample(&mut self.rng);
        let clean = CleanSequence::new(seq.tokens.clone());
        // Draw the per-position decisions up front: `mask_with` takes the
        // predicate by value and the borrow checker will not lend it `self`.
        let draws: Vec<bool> = (0..seq.tokens.len())
            .map(|i| i < seq.content_len && self.rng.next_f32() < t.get())
            .collect();
        let view = clean.mask_with(t, self.cfg.mask_token, |i| draws[i]);
        (clean, view)
    }

    /// Run one training step.
    ///
    /// Returns `None` when the step would score nothing — the masking process
    /// happened to leave every position visible. Its loss and gradient would
    /// be exactly zero, so submitting it costs a forward and backward pass to
    /// learn nothing.
    pub fn step<I: NeighbourIndex, E: ChunkEmbedder>(
        &mut self,
        session: &mut Session,
        model: &CcaModel,
        seq: &TrainingSequence,
        sources: &mut RetrievalSources<'_, I, E>,
    ) -> Result<Option<StepReport>, StepError> {
        let cfg = self.cfg.cca;
        if seq.tokens.len() != cfg.seq_len() {
            return Err(StepError::LengthMismatch {
                got: seq.tokens.len(),
                expected: cfg.seq_len(),
            });
        }
        if model.encoder().is_none() {
            return Err(StepError::NeighboursNotEncoded);
        }

        let (clean, view) = self.corrupt(seq);
        let stats = self
            .loss
            .build_labels(&view, &clean, &mut self.labels)
            .expect("the view was masked from this clean sequence");
        if !stats.contributes() {
            return Ok(None);
        }

        let neighbours = self.retrieve_for(&view, seq, sources);
        self.bind(session, &view, neighbours.as_ref());

        session.step();
        session.wait();
        self.steps += 1;

        Ok(Some(StepReport {
            loss: session.read_loss(),
            noise_level: view.noise_level(),
            scored: stats.scored,
            chunks_retrieved: neighbours.as_ref().map_or(0, |n| {
                (0..n.num_chunks())
                    .filter(|&c| n.chunk_has_neighbours(c))
                    .count()
            }),
            gate_open: neighbours.is_some(),
        }))
    }

    fn retrieve_for<I: NeighbourIndex, E: ChunkEmbedder>(
        &mut self,
        view: &NoisedView,
        seq: &TrainingSequence,
        sources: &mut RetrievalSources<'_, I, E>,
    ) -> Option<Neighbours> {
        if !self.cfg.retrieval_enabled {
            return None;
        }
        let chunked = ChunkedView::new(view, self.cfg.cca).ok()?;
        // `Phase::Training` closes the low-`t` band, where a neighbour's
        // continuation could hold the few remaining answers.
        let queries = chunk_queries(
            chunked,
            Phase::Training,
            self.cfg.admission,
            sources.embedder,
        )?;
        Some(retrieve(
            &queries,
            self.cfg.cca,
            sources.index,
            sources.corpus,
            sources.guard,
            seq.document,
        ))
    }

    fn bind(&mut self, session: &mut Session, view: &NoisedView, neighbours: Option<&Neighbours>) {
        let cfg = self.cfg.cca;
        let (n, m) = (cfg.seq_len(), cfg.chunk_size());

        self.t_col.clear();
        self.t_col.resize(n, view.noise_level().get());

        // A chunk that retrieved nothing carries a zero-filled key/value
        // block. Attending to padding is not the same as not attending, so
        // the mask forces its contribution to zero rather than letting the
        // gate learn what attending to zeros happens to produce.
        self.retrieval_mask.clear();
        self.retrieval_mask.resize(n, 0.0);
        let rows = cfg.neighbour_kv_rows();
        self.neighbour_tokens.clear();
        self.neighbour_tokens
            .resize(cfg.num_chunks() * rows, self.cfg.mask_token.0);

        if let Some(found) = neighbours {
            for chunk in 0..cfg.num_chunks() {
                if !found.chunk_has_neighbours(chunk) {
                    continue;
                }
                for row in chunk * m..(chunk + 1) * m {
                    self.retrieval_mask[row] = 1.0;
                }
                if let Some(tokens) = found.chunk_tokens(chunk) {
                    let start = chunk * rows;
                    self.neighbour_tokens[start..start + rows].copy_from_slice(tokens);
                }
            }
        }

        session.set_input_u32("token_ids", view.tokens());
        session.set_input(input_names::T_COL, &self.t_col);
        session.set_input(input_names::RETRIEVAL_MASK, &self.retrieval_mask);
        session.set_input_u32("cca.neighbour_tokens", &self.neighbour_tokens);
        session.set_input("labels", &self.labels);
    }
}

/// Which optimizer `Session::step` applies.
///
/// Meganeura's optimizer state lives on the session and persists across
/// steps rather than being passed per step, so this is applied once before
/// training rather than inside the loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// Plain SGD.
    Sgd {
        /// Learning rate.
        lr: f32,
    },
    /// Adam.
    Adam {
        /// Learning rate.
        lr: f32,
        /// First-moment decay.
        beta1: f32,
        /// Second-moment decay.
        beta2: f32,
        /// Denominator floor.
        eps: f32,
    },
}

impl Optimizer {
    /// Adam at `lr` with the usual decay constants.
    pub fn adam(lr: f32) -> Self {
        Self::Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// Install `optimizer` on `session` for every subsequent step.
pub fn configure_optimizer(session: &mut Session, optimizer: Optimizer) {
    match optimizer {
        Optimizer::Sgd { lr } => session.set_learning_rate(lr),
        Optimizer::Adam {
            lr,
            beta1,
            beta2,
            eps,
        } => session.set_adam(lr, beta1, beta2, eps),
    }
}

/// Write the zero-init parameters a freshly built model needs before its
/// first step, so every CCA block starts as the identity.
///
/// Meganeura initialises parameters itself, so this has to be done explicitly
/// — and it has to be done, or the retrofit begins by pushing an untrained
/// cross-attention output into a frozen residual stream.
pub fn apply_zero_init(session: &mut Session, model: &CcaModel) {
    for name in model.zero_init_param_names() {
        let len = session
            .param_size(&name)
            .expect("the model declared this parameter");
        session.set_parameter(&name, &vec![0.0; len]);
    }
}

/// Assert a model was built for training rather than for sampling.
pub fn check_trainable(neighbours: NeighbourInput) -> Result<(), StepError> {
    match neighbours {
        NeighbourInput::Encoded => Ok(()),
        NeighbourInput::Cached => Err(StepError::NeighboursNotEncoded),
    }
}

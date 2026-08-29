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

/// Where retrieval queries are built from.
///
/// There is exactly one correct answer, and the other variant exists only
/// behind the `leak-harness` feature so that Phase 1 can calibrate its
/// evaluation protocol against a run known to be leaking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuerySource {
    /// The noised view the denoiser sees. The retriever may only see what the
    /// denoiser sees.
    #[default]
    NoisedView,
    /// **The leak, on purpose.** Query a clean view of the same sequence, so
    /// the retrieved neighbours correlate with exactly the tokens that were
    /// masked.
    ///
    /// Never enable this for a run whose numbers anyone will read. Its whole
    /// purpose is to produce a run that *should* be flagged, so that a
    /// protocol which fails to flag it is known to be broken before it is
    /// trusted with a real result. Perplexity improves under this setting;
    /// that is the point.
    #[cfg(feature = "leak-harness")]
    CleanSequenceLeak,
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
    /// Where queries come from. Leave at the default.
    pub query_source: QuerySource,
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
            query_source: QuerySource::default(),
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
    /// Writes only the entries that changed; see [`SparseLabels`].
    labels: SparseLabels,
    /// The labels buffer is zeroed once, lazily, on the first step.
    labels_zeroed: bool,
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
            labels: SparseLabels::new(n, cfg.vocab_size),
            labels_zeroed: false,
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
        if !self.labels_zeroed {
            // Meganeura does not promise a fresh input buffer is zeroed, and
            // every later step only undoes its own predecessor's writes.
            self.labels
                .zero(session, "labels")
                .expect("the graph declares a labels input of this shape");
            self.labels_zeroed = true;
        }
        let stats = self
            .labels
            .write(session, "labels", self.loss, &view, &clean)
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
        // The view the *retriever* reads. Identical to the denoiser's, unless
        // the calibration harness is deliberately breaking that.
        let query_view = match self.cfg.query_source {
            QuerySource::NoisedView => None,
            #[cfg(feature = "leak-harness")]
            QuerySource::CleanSequenceLeak => Some(NoisedView::from_tokens(
                seq.tokens.clone(),
                view.noise_level(),
                self.cfg.mask_token,
            )),
        };
        let query_view = query_view.as_ref().unwrap_or(view);

        let chunked = ChunkedView::new(query_view, self.cfg.cca).ok()?;
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

        bind_inputs(
            session,
            view,
            &self.t_col,
            &self.retrieval_mask,
            &self.neighbour_tokens,
        );
    }
}

/// Why a sparse label write could not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseLabelError {
    /// The session has no input by that name.
    NoSuchInput,
    /// The buffer is not `seq_len * vocab` f32s.
    SizeMismatch { got: usize, expected: usize },
}

impl std::fmt::Display for SparseLabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NoSuchInput => write!(f, "no such input on this session"),
            Self::SizeMismatch { got, expected } => {
                write!(f, "labels buffer is {got} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for SparseLabelError {}

/// Writes the labels tensor in place, touching only what changed.
///
/// The dense tensor is `n * vocab` floats carrying at most `n` non-zeros. At
/// LLaDA's vocabulary and `n = 512` that is 259 MB re-uploaded every step to
/// convey a few hundred numbers, and over a gigabyte at `n = 2048`.
///
/// Meganeura's input buffers are pinned by the memory plan — never aliased —
/// and allocated `Memory::Shared`, so they are device-local *and*
/// host-coherent, and their contents survive between steps. That makes the
/// dense upload avoidable: keep the buffer zeroed, and each step clear only
/// the entries the previous step wrote before writing this step's. Per-step
/// traffic drops from `n * vocab` floats to about `2n`.
///
/// The allocation itself remains. Removing that needs an indexed-label
/// cross-entropy in Meganeura, which is worth doing for `n = 2048` and is not
/// on Phase 1's path.
#[derive(Debug, Clone)]
pub struct SparseLabels {
    seq_len: usize,
    vocab: usize,
    /// Flat offsets written last step, cleared before the next write.
    dirty: Vec<usize>,
    entries: Vec<(u32, u32)>,
}

impl SparseLabels {
    /// A writer for an `[seq_len, vocab]` labels input.
    pub fn new(seq_len: usize, vocab: usize) -> Self {
        Self {
            seq_len,
            vocab,
            dirty: Vec::with_capacity(seq_len),
            entries: Vec::with_capacity(seq_len),
        }
    }

    fn buffer<'s>(
        &self,
        session: &'s mut Session,
        name: &str,
    ) -> Result<&'s mut [f32], SparseLabelError> {
        // The documented ordering requirement: the previous step's `wait()`
        // must have completed before the host writes, or this frame races the
        // GPU's in-flight read of the last one.
        session.wait();
        let (ptr, size) = session
            .input_host_ptr(name)
            .ok_or(SparseLabelError::NoSuchInput)?;
        let expected = self.seq_len * self.vocab * size_of::<f32>();
        if size != expected {
            return Err(SparseLabelError::SizeMismatch {
                got: size,
                expected,
            });
        }
        // SAFETY: the pointer comes from `input_host_ptr`, which returns the
        // session-owned, host-coherent allocation backing this input and is
        // valid for as long as the session lives. The size is checked to be
        // exactly `seq_len * vocab` f32s just above, so the slice covers the
        // allocation and no more. The buffer holds f32 label data -- it is
        // what `set_input` writes through `bytemuck::cast_slice` -- so it is
        // f32-aligned and initialised. `wait()` above establishes that no GPU
        // read is in flight.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.cast::<f32>(), self.seq_len * self.vocab) })
    }

    /// Zero the whole buffer.
    ///
    /// Must run once before the first [`Self::write`]: Meganeura does not
    /// promise a freshly allocated input buffer is zeroed, and every later
    /// step only undoes its own predecessor's writes.
    pub fn zero(&mut self, session: &mut Session, name: &str) -> Result<(), SparseLabelError> {
        let buf = self.buffer(session, name)?;
        buf.fill(0.0);
        self.dirty.clear();
        Ok(())
    }

    /// Write the labels for `view`, clearing the previous step's entries.
    pub fn write(
        &mut self,
        session: &mut Session,
        name: &str,
        loss: crate::loss::MaskedDiffusionLoss,
        view: &NoisedView,
        clean: &CleanSequence,
    ) -> Result<crate::loss::LabelStats, Box<dyn std::error::Error>> {
        let stats = loss.scatter(view, clean, &mut self.entries)?;
        let vocab = self.vocab;
        let entries = std::mem::take(&mut self.entries);
        let buf = self.buffer(session, name)?;
        for &offset in &self.dirty {
            buf[offset] = 0.0;
        }
        self.dirty.clear();
        for &(position, token) in &entries {
            let offset = position as usize * vocab + token as usize;
            buf[offset] = stats.weight;
            self.dirty.push(offset);
        }
        self.entries = entries;
        Ok(stats)
    }
}

/// Bind one step's inputs onto a session.
///
/// Shared with evaluation, which runs the same graph over substituted
/// neighbour blocks — the random-neighbour and oracle conditions differ from
/// a training step only in what lands in `neighbour_tokens`, and routing both
/// through one function is what keeps them comparable.
///
/// Labels are not bound here. They go through [`SparseLabels`], which writes
/// into the pinned buffer directly rather than re-uploading it.
pub fn bind_inputs(
    session: &mut Session,
    view: &NoisedView,
    t_col: &[f32],
    retrieval_mask: &[f32],
    neighbour_tokens: &[u32],
) {
    session.set_input_u32("token_ids", view.tokens());
    session.set_input(input_names::T_COL, t_col);
    session.set_input(input_names::RETRIEVAL_MASK, retrieval_mask);
    session.set_input_u32("cca.neighbour_tokens", neighbour_tokens);
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

/// Fill parameters with a deterministic small-random init keyed by `seed`
/// and the parameter name.
///
/// `only` restricts the write to that name list (the retrofit's trainable
/// tensors, after a pretrained backbone has already been copied in). Norm
/// scale parameters are initialised near 1; everything else is a small
/// symmetric range so the first forward does not blow up.
pub fn seed_parameters(session: &mut Session, seed: u64, only: Option<&[String]>) {
    let names: Vec<String> = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        if let Some(only) = only
            && !only.contains(&name)
        {
            continue;
        }
        let len = session.param_size(&name).expect("declared");
        let mixed = name
            .bytes()
            .fold(0u64, |a, b| a.rotate_left(5) ^ u64::from(b));
        let mut rng = Rng::new(seed ^ mixed);
        let is_norm = name.contains("norm");
        let scale = (2.0 / len as f32).sqrt().min(0.08);
        let values: Vec<f32> = (0..len)
            .map(|_| {
                if is_norm {
                    1.0
                } else {
                    (rng.next_f32() - 0.5) * 2.0 * scale
                }
            })
            .collect();
        session.set_parameter(&name, &values);
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

//! Chunking the noised view, and building retrieval queries from it.
//!
//! Section 1 of the design sketch. RETRO's chunk offset — chunk `u` attending
//! to `Ret(C_{u-1})` rather than `Ret(C_u)` — existed to stop a chunk from
//! attending to neighbours retrieved using its own tokens. Diffusion has no
//! ordering, so the offset is gone. The principle it protected is not:
//! the retriever may only see what the denoiser sees, which
//! [`crate::view`] enforces structurally.

use crate::config::CcaConfig;
use crate::schedule::{NoiseLevel, Phase, retrieve_now};
use crate::view::{DerivedFromView, NoisedView, ViewId};

/// One chunk of the noised view.
#[derive(Debug, Clone, Copy)]
pub struct Chunk<'v> {
    index: usize,
    tokens: &'v [u32],
    masked: &'v [bool],
}

impl<'v> Chunk<'v> {
    /// Position of this chunk in the sequence, `0..l`.
    pub fn index(self) -> usize {
        self.index
    }

    /// The chunk's tokens, `[MASK]` included.
    pub fn tokens(self) -> &'v [u32] {
        self.tokens
    }

    /// Per-position mask flags.
    pub fn masked(self) -> &'v [bool] {
        self.masked
    }

    /// Fraction of this chunk's positions that are masked.
    ///
    /// The number that decides whether this chunk's query embedding carries
    /// any signal. A chunk sitting at 95% `[MASK]` produces an embedding of
    /// essentially nothing, and the neighbours it retrieves are noise —
    /// regardless of what the global `t` says.
    pub fn mask_rate(self) -> f32 {
        if self.masked.is_empty() {
            return 0.0;
        }
        self.masked.iter().filter(|&&m| m).count() as f32 / self.masked.len() as f32
    }
}

/// A noised view split into `l` chunks of `m` tokens.
#[derive(Debug, Clone, Copy)]
pub struct ChunkedView<'v> {
    view: &'v NoisedView,
    chunk_size: usize,
    num_chunks: usize,
}

/// Why a view could not be chunked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkAlignmentError {
    /// The view's length.
    pub seq_len: usize,
    /// The length the configuration expects.
    pub expected: usize,
}

impl std::fmt::Display for ChunkAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "view has {} tokens but the configuration describes {}",
            self.seq_len, self.expected
        )
    }
}

impl std::error::Error for ChunkAlignmentError {}

impl<'v> ChunkedView<'v> {
    /// Split `view` according to `cfg`.
    pub fn new(view: &'v NoisedView, cfg: CcaConfig) -> Result<Self, ChunkAlignmentError> {
        if view.len() != cfg.seq_len() {
            return Err(ChunkAlignmentError {
                seq_len: view.len(),
                expected: cfg.seq_len(),
            });
        }
        Ok(Self {
            view,
            chunk_size: cfg.chunk_size(),
            num_chunks: cfg.num_chunks(),
        })
    }

    /// The view being chunked.
    pub fn view(self) -> &'v NoisedView {
        self.view
    }

    /// Number of chunks, `l`.
    pub fn len(self) -> usize {
        self.num_chunks
    }

    /// Whether there are no chunks.
    pub fn is_empty(self) -> bool {
        self.num_chunks == 0
    }

    /// Chunk `index`, or `None` if out of range.
    pub fn get(self, index: usize) -> Option<Chunk<'v>> {
        if index >= self.num_chunks {
            return None;
        }
        let start = index * self.chunk_size;
        let end = start + self.chunk_size;
        Some(Chunk {
            index,
            tokens: &self.view.tokens()[start..end],
            masked: &self.view.masked()[start..end],
        })
    }

    /// Iterate the chunks in order.
    pub fn iter(self) -> impl Iterator<Item = Chunk<'v>> {
        (0..self.num_chunks).map(move |i| self.get(i).expect("index below num_chunks"))
    }
}

/// Per-chunk admission policy.
///
/// Option (a) from the design sketch — "restrict retrieval to low-mask steps
/// only" — applied per chunk rather than per step. A global `t` inside the
/// retrieval band says the *average* chunk has signal; it says nothing about
/// a particular chunk that the masking process happened to flatten. Skipping
/// those individually costs one count per chunk and removes the noisiest
/// queries from the index traffic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkAdmission {
    max_mask_rate: f32,
}

impl ChunkAdmission {
    /// Admit chunks masked at no more than `max_mask_rate`.
    pub fn new(max_mask_rate: f32) -> Option<Self> {
        (max_mask_rate.is_finite() && (0.0..=1.0).contains(&max_mask_rate))
            .then_some(Self { max_mask_rate })
    }

    /// Admit every chunk, deferring entirely to the global hard gate.
    pub fn permissive() -> Self {
        Self { max_mask_rate: 1.0 }
    }

    /// Whether `chunk` is admitted.
    pub fn admits(self, chunk: Chunk<'_>) -> bool {
        chunk.mask_rate() <= self.max_mask_rate
    }
}

impl Default for ChunkAdmission {
    /// Matches the top of the sketch's training band: at `t = 0.85` roughly
    /// 85% of positions are masked, so a chunk above that is past the point
    /// where the global gate would have allowed retrieval at all.
    fn default() -> Self {
        Self {
            max_mask_rate: 0.85,
        }
    }
}

/// Encodes chunks of a noised view into retrieval query vectors.
///
/// This is the piece the design sketch marks unsolved. RETRO used a frozen
/// BERT over clean text; here the input is partly `[MASK]`, and a frozen
/// clean-text encoder has never seen that distribution. The sketch lists
/// three ways out, in increasing order of cost:
///
/// * **(a)** restrict retrieval to low-mask steps — implemented by
///   [`ChunkAdmission`] plus the hard gate, and usable with a frozen encoder;
/// * **(b)** fine-tune the retriever on masked inputs with `t` as a
///   conditioning input;
/// * **(c)** train it jointly against the denoising loss.
///
/// The trait takes `t` for the sake of (b) and (c). An implementation of (a)
/// ignores it.
pub trait RetrieverEncode {
    /// Width of the produced query vectors, `d_r`.
    fn query_dim(&self) -> usize;

    /// Encode one chunk at noise level `t` into a query vector of
    /// [`Self::query_dim`] elements, appended to `out`.
    fn encode_chunk(&mut self, chunk: Chunk<'_>, t: NoiseLevel, out: &mut Vec<f32>);
}

/// Query vectors for every chunk of one view.
///
/// Carries the [`ViewId`] it was built from, so the denoiser can reject
/// neighbours retrieved against a different — possibly cleaner — view.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkQueries {
    view_id: ViewId,
    query_dim: usize,
    /// `l * query_dim` values, chunk-major.
    embeddings: Vec<f32>,
    /// Per-chunk admission, `l` entries.
    admitted: Vec<bool>,
}

impl DerivedFromView for ChunkQueries {
    fn view_id(&self) -> ViewId {
        self.view_id
    }
}

impl ChunkQueries {
    /// Number of chunks, `l`.
    pub fn len(&self) -> usize {
        self.admitted.len()
    }

    /// Whether there are no chunks.
    pub fn is_empty(&self) -> bool {
        self.admitted.is_empty()
    }

    /// Query width, `d_r`.
    pub fn query_dim(&self) -> usize {
        self.query_dim
    }

    /// The query vector for chunk `index`.
    pub fn query(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.query_dim)?;
        self.embeddings.get(start..start + self.query_dim)
    }

    /// Whether chunk `index` was admitted for retrieval.
    pub fn is_admitted(&self, index: usize) -> bool {
        self.admitted.get(index).copied().unwrap_or(false)
    }

    /// Indices of the chunks admitted for retrieval.
    pub fn admitted_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.admitted
            .iter()
            .enumerate()
            .filter_map(|(i, &ok)| ok.then_some(i))
    }

    /// How many chunks were admitted.
    pub fn num_admitted(&self) -> usize {
        self.admitted.iter().filter(|&&ok| ok).count()
    }
}

/// Build retrieval queries from the **noised** view.
///
/// The signature is the enforcement: there is no overload taking a
/// [`crate::view::CleanSequence`], and `t` comes from the view rather than
/// from the caller, so a query cannot be built at a noise level the view was
/// not produced at.
///
/// Returns `None` when the hard gate closes at this `t` for this phase —
/// there is nothing to retrieve and no index traffic to spend.
pub fn chunk_queries<E: RetrieverEncode>(
    chunked: ChunkedView<'_>,
    phase: Phase,
    admission: ChunkAdmission,
    encoder: &mut E,
) -> Option<ChunkQueries> {
    let view = chunked.view();
    let t = view.noise_level();
    if !retrieve_now(t, phase) {
        return None;
    }

    let query_dim = encoder.query_dim();
    let mut embeddings = Vec::with_capacity(chunked.len() * query_dim);
    let mut admitted = Vec::with_capacity(chunked.len());

    for chunk in chunked.iter() {
        let ok = admission.admits(chunk);
        admitted.push(ok);
        if ok {
            let before = embeddings.len();
            encoder.encode_chunk(chunk, t, &mut embeddings);
            assert_eq!(
                embeddings.len() - before,
                query_dim,
                "RetrieverEncode::encode_chunk wrote {} values, expected query_dim={}",
                embeddings.len() - before,
                query_dim
            );
        } else {
            // Keep the layout dense so `query(i)` stays a slice index. A
            // skipped chunk's vector is never searched with.
            embeddings.extend(std::iter::repeat_n(0.0, query_dim));
        }
    }

    Some(ChunkQueries {
        view_id: view.id(),
        query_dim,
        embeddings,
        admitted,
    })
}

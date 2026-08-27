//! Turning documents into training sequences and into a retrievable corpus.
//!
//! Two outputs from one input, and they have to agree about provenance: a
//! training sequence carries the [`DocumentId`] it came from, and every
//! neighbour record carries the id it was extracted from, so
//! [`crate::retrieval::leakage::LeakageGuard`] can keep a document out of its
//! own retrieval results exactly rather than heuristically.
//!
//! This is why the corpus format matters more than its size. A dump shipped
//! as one undifferentiated token stream cannot support the provenance guard
//! at all, whatever else it has going for it.

use crate::chunk::{Chunk, RetrieverEncode};
use crate::config::CcaConfig;
use crate::retrieval::corpus::{DocumentId, NeighbourCorpus, NeighbourRecord};
use crate::schedule::NoiseLevel;

/// A tokenized source document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// Stable identity. For Wikipedia this is the article id.
    pub id: DocumentId,
    /// The document's tokens.
    pub tokens: Vec<u32>,
}

/// Embeds a span of tokens into a retrieval vector.
///
/// Deliberately one trait for both sides of retrieval. A query embedding and
/// a corpus embedding must live in the same space or nearest-neighbour search
/// is meaningless, and the cheapest way to guarantee that is to make it the
/// same object — [`RetrieverEncode`] is blanket-implemented for every
/// `ChunkEmbedder`, so the type that indexes the corpus is the type that
/// builds queries against it.
///
/// `t` is `Some` for a query built from a noised view and `None` for a clean
/// corpus chunk. An implementation of the roadmap's option (a) ignores it; one
/// of option (b) conditions on it.
pub trait ChunkEmbedder {
    /// Width of the produced vectors.
    fn embed_dim(&self) -> usize;

    /// Append one `embed_dim`-wide vector for `tokens` to `out`.
    fn embed(&mut self, tokens: &[u32], t: Option<NoiseLevel>, out: &mut Vec<f32>);
}

impl<E: ChunkEmbedder> RetrieverEncode for E {
    fn query_dim(&self) -> usize {
        self.embed_dim()
    }

    fn encode_chunk(&mut self, chunk: Chunk<'_>, t: NoiseLevel, out: &mut Vec<f32>) {
        self.embed(chunk.tokens(), Some(t), out);
    }
}

/// A training sequence: one window of one document, padded to `n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSequence {
    /// The document this window came from.
    pub document: DocumentId,
    /// Exactly `n` tokens, padded at the tail.
    pub tokens: Vec<u32>,
    /// Real tokens before the padding starts.
    ///
    /// Padding must never be masked and must never score: a model that learns
    /// to predict pad tokens is spending capacity on the data loader.
    pub content_len: usize,
}

impl TrainingSequence {
    /// Whether `position` holds a real token.
    pub fn is_content(&self, position: usize) -> bool {
        position < self.content_len
    }

    /// A masking predicate that leaves padding alone.
    ///
    /// Compose with a noise schedule: `seq.content_mask(|i| rng(i) < t)`.
    pub fn content_mask<'a>(
        &'a self,
        mut should_mask: impl FnMut(usize) -> bool + 'a,
    ) -> impl FnMut(usize) -> bool + 'a {
        move |i| i < self.content_len && should_mask(i)
    }
}

/// Split documents into `n`-token training windows.
///
/// Long documents yield consecutive windows rather than being truncated, so
/// no text is dropped, and every window keeps its parent document's id — a
/// window is not a new document for leakage purposes. Short documents yield
/// one padded window.
///
/// `min_content` drops windows with less real text than that. A window that
/// is nearly all padding costs a full forward pass to score a handful of
/// positions.
pub fn training_sequences(
    documents: &[Document],
    seq_len: usize,
    pad_token: u32,
    min_content: usize,
) -> Vec<TrainingSequence> {
    let mut out = Vec::new();
    for doc in documents {
        for window in doc.tokens.chunks(seq_len) {
            if window.len() < min_content {
                continue;
            }
            let mut tokens = window.to_vec();
            let content_len = tokens.len();
            tokens.resize(seq_len, pad_token);
            out.push(TrainingSequence {
                document: doc.id,
                tokens,
                content_len,
            });
        }
    }
    out
}

/// Build the retrievable corpus and its embeddings.
///
/// Each record is a chunk of `m` tokens plus its continuation, `r` tokens in
/// total, taken at stride `m` so the matched chunks tile the document.
///
/// **Only the first `m` tokens are embedded.** The continuation is carried
/// but not indexed, because it is what the model does not have yet — indexing
/// it would make search match on text the query cannot contain, which is a
/// different and much worse retrieval problem than the one intended.
///
/// Records are dropped at the tail of a document when there are not `r`
/// tokens left, since a record padded into its continuation would promise
/// information it does not have.
pub fn build_corpus<E: ChunkEmbedder>(
    documents: &[Document],
    cfg: CcaConfig,
    embedder: &mut E,
) -> NeighbourCorpus {
    let (m, r) = (cfg.chunk_size(), cfg.neighbour_len());
    let mut corpus = NeighbourCorpus::new(r, embedder.embed_dim());
    let mut embedding = Vec::with_capacity(embedder.embed_dim());

    for doc in documents {
        let mut offset = 0;
        while offset + r <= doc.tokens.len() {
            let record = NeighbourRecord {
                document: doc.id,
                offset,
                tokens: doc.tokens[offset..offset + r].to_vec(),
            };
            embedding.clear();
            // The matched chunk only — see the note above.
            embedder.embed(&doc.tokens[offset..offset + m], None, &mut embedding);
            corpus
                .push(&record, &embedding)
                .expect("record and embedding are built to the corpus's shapes");
            offset += m;
        }
    }
    corpus
}

/// A training-free lexical embedder: hashed token n-grams, L2-normalised.
///
/// This is the roadmap's option (a) made concrete. It is not a placeholder —
/// a hashed bag of n-grams is a real lexical retriever, roughly a
/// random-projection BM25 without the term weighting, and it has two
/// properties that matter more than raw quality for a first run:
///
/// * **It needs no training**, so Phase 1 can produce a trustworthy number
///   without first solving the open question of how to encode a
///   `[MASK]`-bearing query.
/// * **It degrades gracefully under masking.** Masked positions contribute
///   nothing rather than contributing noise, so a half-masked chunk yields
///   the embedding of the half that survived — a weaker query, not a wrong
///   one. A dense encoder trained on clean text has no such guarantee, which
///   is the whole reason the question is open.
///
/// It is also the baseline a learned encoder has to beat to justify itself.
#[derive(Debug, Clone)]
pub struct HashedBagEmbedder {
    dim: usize,
    max_order: usize,
    mask_token: u32,
}

impl HashedBagEmbedder {
    /// Hash into `dim` buckets, using n-grams up to `max_order`.
    ///
    /// Bigrams (`max_order = 2`) buy word-order sensitivity for one extra
    /// pass; going higher mostly adds sparsity at these dimensions.
    pub fn new(dim: usize, max_order: usize, mask_token: u32) -> Self {
        assert!(dim > 0, "embedding width must be non-zero");
        assert!(max_order > 0, "n-gram order must be non-zero");
        Self {
            dim,
            max_order,
            mask_token,
        }
    }

    /// Unigrams and bigrams over `dim` buckets.
    pub fn bigram(dim: usize, mask_token: u32) -> Self {
        Self::new(dim, 2, mask_token)
    }

    fn bucket(&self, gram: &[u32]) -> usize {
        // FNV-1a. Cheap, well-mixed for short integer sequences, and
        // deterministic across runs and machines — the corpus is embedded
        // once and queried later, so a hash that varies between processes
        // would silently return nothing.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &tok in gram {
            for byte in tok.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        (h % self.dim as u64) as usize
    }
}

impl ChunkEmbedder for HashedBagEmbedder {
    fn embed_dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, tokens: &[u32], _t: Option<NoiseLevel>, out: &mut Vec<f32>) {
        let start = out.len();
        out.resize(start + self.dim, 0.0);
        let acc = &mut out[start..];

        for order in 1..=self.max_order {
            for gram in tokens.windows(order) {
                // A masked position carries no lexical content, so every gram
                // touching one is skipped rather than hashed as if `[MASK]`
                // were a word.
                if gram.contains(&self.mask_token) {
                    continue;
                }
                acc[self.bucket(gram)] += 1.0;
            }
        }

        // L2-normalise so squared-L2 search ranks by cosine similarity, and so
        // a chunk's length does not decide its distance.
        let norm: f32 = acc.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in acc.iter_mut() {
                *v /= norm;
            }
        }
    }
}

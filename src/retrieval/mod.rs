//! Retrieval: index search, corpus gather, and leakage control.

pub mod corpus;
pub mod index;
pub mod leakage;

use crate::chunk::ChunkQueries;
use crate::config::CcaConfig;
use crate::view::{DerivedFromView, ViewId};
use corpus::{DocumentId, NeighbourCorpus, NeighbourId};
use index::NeighbourIndex;
use leakage::LeakageGuard;

/// Neighbours retrieved for every chunk of one view.
///
/// Shape `[l, k, r]`, flattened chunk-major. Carries the [`ViewId`] the
/// queries came from, so the denoiser can reject neighbours retrieved against
/// a different view of the same sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighbours {
    view_id: ViewId,
    num_chunks: usize,
    k: usize,
    r: usize,
    /// `l * k` ids; `None` where no acceptable neighbour was found.
    ids: Vec<Option<NeighbourId>>,
    /// `l * k * r` token ids; zero-filled where `ids` is `None`.
    tokens: Vec<u32>,
}

impl DerivedFromView for Neighbours {
    fn view_id(&self) -> ViewId {
        self.view_id
    }
}

impl Neighbours {
    /// Number of chunks, `l`.
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Neighbours per chunk, `k`.
    pub fn neighbours_per_chunk(&self) -> usize {
        self.k
    }

    /// Neighbour token length, `r`.
    pub fn neighbour_len(&self) -> usize {
        self.r
    }

    /// The `r` tokens of neighbour `j` of chunk `i`.
    pub fn tokens(&self, chunk: usize, j: usize) -> Option<&[u32]> {
        if chunk >= self.num_chunks || j >= self.k {
            return None;
        }
        let start = (chunk * self.k + j) * self.r;
        self.tokens.get(start..start + self.r)
    }

    /// All `k * r` key/value tokens for chunk `i`, contiguous.
    ///
    /// This is the layout the CCA block's neighbour encoder consumes: one
    /// chunk's entire key/value block in one slice.
    pub fn chunk_tokens(&self, chunk: usize) -> Option<&[u32]> {
        if chunk >= self.num_chunks {
            return None;
        }
        let rows = self.k * self.r;
        let start = chunk * rows;
        self.tokens.get(start..start + rows)
    }

    /// The neighbour id at slot `j` of chunk `i`, if one was found.
    pub fn id(&self, chunk: usize, j: usize) -> Option<NeighbourId> {
        self.ids.get(chunk * self.k + j).copied().flatten()
    }

    /// Whether chunk `i` retrieved at least one neighbour.
    ///
    /// A chunk can come back empty: the hard gate admitted the step, but the
    /// chunk's own mask rate excluded it, or every candidate was excluded by
    /// the leakage guard. Its key/value block is zero-filled, and the CCA
    /// gate must be held at zero for it — attending to a block of padding is
    /// not the same as not attending.
    pub fn chunk_has_neighbours(&self, chunk: usize) -> bool {
        (0..self.k).any(|j| self.id(chunk, j).is_some())
    }

    /// Per-chunk retrieval mask, `l` entries.
    pub fn chunk_mask(&self) -> Vec<bool> {
        (0..self.num_chunks)
            .map(|i| self.chunk_has_neighbours(i))
            .collect()
    }

    /// How many `(chunk, slot)` pairs came back filled.
    pub fn num_filled(&self) -> usize {
        self.ids.iter().filter(|id| id.is_some()).count()
    }
}

/// Search the index for every admitted chunk and gather the neighbour tokens.
///
/// `training_document` identifies the sequence being denoised, so the leakage
/// guard can keep the document out of its own results. At inference, pass the
/// document id the prompt belongs to, or a synthetic id no corpus document
/// uses.
///
/// Exclusions are applied *inside* the search rather than to its output, so a
/// chunk still gets `k` neighbours when its nearest match is excluded.
pub fn retrieve<I: NeighbourIndex>(
    queries: &ChunkQueries,
    cfg: CcaConfig,
    index: &I,
    corpus: &NeighbourCorpus,
    guard: &LeakageGuard,
    training_document: DocumentId,
) -> Neighbours {
    assert_eq!(
        queries.query_dim(),
        index.query_dim(),
        "query width {} does not match index width {}",
        queries.query_dim(),
        index.query_dim()
    );
    assert_eq!(
        corpus.neighbour_len(),
        cfg.neighbour_len(),
        "corpus stores r={} but the configuration says r={}",
        corpus.neighbour_len(),
        cfg.neighbour_len()
    );

    let (l, k, r) = (
        cfg.num_chunks(),
        cfg.neighbours_per_chunk(),
        cfg.neighbour_len(),
    );
    let mut ids = vec![None; l * k];
    let mut tokens = vec![0u32; l * k * r];

    for chunk in 0..l.min(queries.len()) {
        if !queries.is_admitted(chunk) {
            continue;
        }
        let Some(query) = queries.query(chunk) else {
            continue;
        };
        let mut accept = guard.accept_fn(training_document, corpus);
        let found = index.search(query, k, &mut accept);
        for (j, candidate) in found.iter().enumerate() {
            ids[chunk * k + j] = Some(candidate.id);
            if let Some(src) = corpus.tokens(candidate.id) {
                let start = (chunk * k + j) * r;
                tokens[start..start + r].copy_from_slice(src);
            }
        }
    }

    Neighbours {
        view_id: queries.view_id(),
        num_chunks: l,
        k,
        r,
        ids,
        tokens,
    }
}

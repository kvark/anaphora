//! Approximate-nearest-neighbour search over the corpus embeddings.

use super::corpus::{NeighbourCorpus, NeighbourId};
use std::cmp::Ordering;

/// One search result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The matched neighbour.
    pub id: NeighbourId,
    /// Squared L2 distance from the query. Smaller is closer.
    pub distance: f32,
}

/// A nearest-neighbour index over neighbour embeddings.
///
/// The trait exists because the index is the component most likely to be
/// swapped: the design sketch puts it on NVMe at a scale where the
/// implementation below is not the right answer, and an IVF or HNSW backend
/// changes nothing above this line.
pub trait NeighbourIndex {
    /// Query width this index accepts.
    fn query_dim(&self) -> usize;

    /// The `want` closest neighbours to `query`, nearest first.
    ///
    /// `accept` filters candidates during the search rather than after it, so
    /// that excluding the training document does not silently return fewer
    /// than `want` neighbours. See [`crate::retrieval::leakage`].
    fn search(
        &self,
        query: &[f32],
        want: usize,
        accept: &mut dyn FnMut(NeighbourId) -> bool,
    ) -> Vec<Candidate>;
}

/// Exhaustive exact search.
///
/// Exact rather than approximate, and linear in corpus size. That is the
/// right trade for tests, for small corpora, and as the reference an
/// approximate backend is measured against — recall is only meaningful
/// relative to an exact baseline. It is *not* the right trade at the scale
/// the design sketch describes, which is why [`NeighbourIndex`] exists.
#[derive(Debug, Clone)]
pub struct ExactIndex {
    embed_dim: usize,
    /// `count * embed_dim`.
    embeddings: Vec<f32>,
}

impl ExactIndex {
    /// Build an index over every embedding in `corpus`.
    pub fn build(corpus: &NeighbourCorpus) -> Self {
        Self {
            embed_dim: corpus.embed_dim(),
            embeddings: corpus.embeddings().to_vec(),
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.embeddings
            .len()
            .checked_div(self.embed_dim)
            .unwrap_or(0)
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn vector(&self, i: usize) -> &[f32] {
        let start = i * self.embed_dim;
        &self.embeddings[start..start + self.embed_dim]
    }
}

impl NeighbourIndex for ExactIndex {
    fn query_dim(&self) -> usize {
        self.embed_dim
    }

    fn search(
        &self,
        query: &[f32],
        want: usize,
        accept: &mut dyn FnMut(NeighbourId) -> bool,
    ) -> Vec<Candidate> {
        assert_eq!(
            query.len(),
            self.embed_dim,
            "query width {} does not match index width {}",
            query.len(),
            self.embed_dim
        );
        if want == 0 {
            return Vec::new();
        }

        // Keep the `want` best seen so far, worst-first, so the eviction
        // check is a single comparison against the tail.
        let mut best: Vec<Candidate> = Vec::with_capacity(want + 1);
        for i in 0..self.len() {
            let id = NeighbourId(i as u32);
            if !accept(id) {
                continue;
            }
            let distance = squared_l2(query, self.vector(i));
            if best.len() == want && distance >= best[want - 1].distance {
                continue;
            }
            let pos = best
                .binary_search_by(|c| c.distance.partial_cmp(&distance).unwrap_or(Ordering::Equal))
                .unwrap_or_else(|p| p);
            best.insert(pos, Candidate { id, distance });
            best.truncate(want);
        }
        best
    }
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

//! Keeping the training document out of its own retrieval results.
//!
//! The design sketch states the problem in a comment — *"training doc is in
//! the index -- filter aggressively"* — and sketches the fix as an inline
//! `drop_high_ngram_overlap(nbrs, x_t)`. That inline form has a hole, and
//! this module closes it by splitting the job in two.
//!
//! # Why not filter inline against `x_t`
//!
//! Masked positions cannot match anything, so an n-gram filter run against
//! the *noised* view is blindest exactly where the leak lives: a neighbour
//! whose continuation supplies the tokens that were masked has, by
//! construction, no overlap to detect at those positions. The filter would
//! pass it.
//!
//! # Why not filter inline against `x_0`
//!
//! It works, but it opens a second-order channel: which neighbours survive
//! now depends on the clean sequence, so information about `x_0` reaches the
//! forward pass through the *composition* of the surviving set. Smaller than
//! the leak it fixes, and still not something to introduce on purpose.
//!
//! # What this module does instead
//!
//! * **At query time** — exclude by source document ([`LeakageGuard`]).
//!   Exact, cheap, and independent of the masking, so it opens no channel.
//! * **At corpus preparation time** — run the n-gram overlap filter against
//!   the clean document *before training starts* ([`LeakageGuard::audit`]),
//!   and fold the result into the same exclusion set. Near-duplicates that
//!   live under a different document id — boilerplate, licence text, quoted
//!   passages, forks of the same source — get caught here, and because the
//!   audit runs once, offline, its outcome does not vary with `t` or with
//!   which positions a given step happened to mask.

use super::corpus::{DocumentId, NeighbourCorpus, NeighbourId};
use std::collections::{HashMap, HashSet};

/// Fraction of `a`'s distinct `n`-grams that also occur in `b`.
///
/// Returns `0.0` when `a` is shorter than `n`, since there is nothing to
/// compare. Distinct rather than positional: a neighbour that repeats one
/// shared phrase many times is not more of a duplicate than one that says it
/// once.
pub fn ngram_overlap(a: &[u32], b: &[u32], n: usize) -> f32 {
    if n == 0 || a.len() < n {
        return 0.0;
    }
    let a_grams: HashSet<&[u32]> = a.windows(n).collect();
    if a_grams.is_empty() {
        return 0.0;
    }
    let b_grams: HashSet<&[u32]> = if b.len() < n {
        HashSet::new()
    } else {
        b.windows(n).collect()
    };
    let shared = a_grams.iter().filter(|g| b_grams.contains(**g)).count();
    shared as f32 / a_grams.len() as f32
}

/// N-gram overlap threshold for the offline audit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NgramOverlapFilter {
    n: usize,
    max_overlap: f32,
}

impl NgramOverlapFilter {
    /// Reject neighbours sharing more than `max_overlap` of their `n`-grams
    /// with the training document.
    pub fn new(n: usize, max_overlap: f32) -> Option<Self> {
        (n > 0 && max_overlap.is_finite() && (0.0..=1.0).contains(&max_overlap))
            .then_some(Self { n, max_overlap })
    }

    /// N-gram order.
    pub fn order(self) -> usize {
        self.n
    }

    /// Maximum tolerated overlap.
    pub fn max_overlap(self) -> f32 {
        self.max_overlap
    }

    /// Whether `neighbour` overlaps `document` beyond the threshold.
    pub fn rejects(self, neighbour: &[u32], document: &[u32]) -> bool {
        ngram_overlap(neighbour, document, self.n) > self.max_overlap
    }
}

impl Default for NgramOverlapFilter {
    /// 8-grams at 10%. "Filter aggressively": at `r = 128` tokens a
    /// neighbour tripping this shares roughly a dozen 8-grams with the
    /// training document, which is well past coincidence for natural text
    /// and is the regime where copy-from-neighbour starts to pay off.
    fn default() -> Self {
        Self {
            n: 8,
            max_overlap: 0.1,
        }
    }
}

/// Decides which neighbours a given training document is allowed to retrieve.
///
/// Pass [`LeakageGuard::accept_fn`] to [`super::index::NeighbourIndex::search`]
/// so exclusions apply *during* the search. Filtering afterwards silently
/// returns fewer than `k` neighbours — and a chunk that quietly ends up with
/// one neighbour instead of two is a change in the experiment, not in the
/// safety margin.
#[derive(Debug, Clone, Default)]
pub struct LeakageGuard {
    exclude_source_document: bool,
    audited: HashMap<DocumentId, HashSet<NeighbourId>>,
}

impl LeakageGuard {
    /// A guard that excludes neighbours drawn from the training document
    /// itself. This is the minimum; it is not sufficient on its own, because
    /// near-duplicates live under other document ids.
    pub fn by_source_document() -> Self {
        Self {
            exclude_source_document: true,
            audited: HashMap::new(),
        }
    }

    /// A guard that excludes nothing.
    ///
    /// Only correct when the index provably does not contain the training
    /// data — a held-out evaluation corpus, or inference against a corpus
    /// built from a disjoint source.
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Run the offline n-gram audit for one training document.
    ///
    /// Scans the whole corpus once and records every neighbour that overlaps
    /// `clean_tokens` beyond `filter`'s threshold. Intended to run at data
    /// preparation time, against the clean document, before any training
    /// step — see this module's header for why not at query time.
    ///
    /// Returns the number of neighbours newly excluded.
    pub fn audit(
        &mut self,
        document: DocumentId,
        clean_tokens: &[u32],
        corpus: &NeighbourCorpus,
        filter: NgramOverlapFilter,
    ) -> usize {
        let entry = self.audited.entry(document).or_default();
        let mut added = 0;
        for i in 0..corpus.len() {
            let id = NeighbourId(i as u32);
            let Some(tokens) = corpus.tokens(id) else {
                continue;
            };
            if filter.rejects(tokens, clean_tokens) && entry.insert(id) {
                added += 1;
            }
        }
        added
    }

    /// Whether `candidate` may be retrieved while training on `document`.
    pub fn accepts(
        &self,
        document: DocumentId,
        candidate: NeighbourId,
        corpus: &NeighbourCorpus,
    ) -> bool {
        if self.exclude_source_document && corpus.document(candidate) == Some(document) {
            return false;
        }
        !self
            .audited
            .get(&document)
            .is_some_and(|set| set.contains(&candidate))
    }

    /// An `accept` closure for [`super::index::NeighbourIndex::search`].
    pub fn accept_fn<'a>(
        &'a self,
        document: DocumentId,
        corpus: &'a NeighbourCorpus,
    ) -> impl FnMut(NeighbourId) -> bool + 'a {
        move |candidate| self.accepts(document, candidate, corpus)
    }

    /// How many neighbours the audit excluded for `document`.
    pub fn audited_exclusions(&self, document: DocumentId) -> usize {
        self.audited.get(&document).map_or(0, HashSet::len)
    }
}

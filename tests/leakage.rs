//! The leakage guards: view identity, document provenance, n-gram audit.

use anaphora::retrieval::corpus::{DocumentId, NeighbourCorpus, NeighbourRecord};
use anaphora::retrieval::index::{ExactIndex, NeighbourIndex};
use anaphora::retrieval::leakage::{LeakageGuard, NgramOverlapFilter, ngram_overlap};
use anaphora::schedule::NoiseLevel;
use anaphora::view::{CleanSequence, MaskToken, NoisedView, check_same_view};

const MASK: MaskToken = MaskToken(0);

fn corpus_with(docs: &[(u64, &[u32])], r: usize) -> NeighbourCorpus {
    let mut corpus = NeighbourCorpus::new(r, 2);
    for (i, &(doc, tokens)) in docs.iter().enumerate() {
        corpus
            .push(
                &NeighbourRecord {
                    document: DocumentId(doc),
                    offset: 0,
                    tokens: tokens.to_vec(),
                },
                &[i as f32, 0.0],
            )
            .expect("shapes match");
    }
    corpus
}

#[test]
fn derived_values_carry_their_view_identity() {
    let clean = CleanSequence::new(vec![1, 2, 3, 4]);
    let a = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i % 2 == 0);
    let b = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i % 2 == 0);
    // Same sequence, same t, same masking rule — still different views, so a
    // value derived from one is not interchangeable with the other.
    assert_ne!(a.id(), b.id());
}

#[test]
fn revealing_produces_a_new_view() {
    // A denoising step changes the sequence, so neighbours retrieved against
    // the previous view are stale. Staleness has to be the refresh
    // schedule's decision, which it cannot be if reveal mutates in place.
    let view = NoisedView::all_masked(4, MASK);
    let next = view.reveal(&[(0, 7)], NoiseLevel::new(0.5).unwrap());
    assert_ne!(view.id(), next.id());
    assert_eq!(next.tokens()[0], 7);
    assert!(!next.masked()[0]);
    assert_eq!(view.num_masked(), 4);
    assert_eq!(next.num_masked(), 3);
}

#[test]
fn mask_rate_counts_what_masking_actually_did() {
    let clean = CleanSequence::new(vec![1, 2, 3, 4]);
    let view = clean.mask_with(NoiseLevel::new(0.9).unwrap(), MASK, |i| i < 1);
    // t says 0.9; the process masked one position in four.
    assert_eq!(view.noise_level().get(), 0.9);
    assert_eq!(view.mask_rate(), 0.25);
}

#[test]
fn ngram_overlap_is_a_fraction_of_distinct_grams() {
    let a = [1, 2, 3, 4];
    assert_eq!(ngram_overlap(&a, &a, 2), 1.0);
    assert_eq!(ngram_overlap(&a, &[9, 9, 9], 2), 0.0);
    // a's 2-grams: (1,2) (2,3) (3,4); b holds (2,3) only.
    let overlap = ngram_overlap(&a, &[2, 3], 2);
    assert!((overlap - 1.0 / 3.0).abs() < 1e-6);
    // Nothing to compare when a is shorter than n.
    assert_eq!(ngram_overlap(&[1], &[1], 4), 0.0);
}

#[test]
fn source_document_is_excluded() {
    let corpus = corpus_with(&[(1, &[1, 2, 3, 4]), (2, &[5, 6, 7, 8])], 4);
    let guard = LeakageGuard::by_source_document();
    let ids: Vec<_> = (0..corpus.len())
        .map(|i| anaphora::retrieval::corpus::NeighbourId(i as u32))
        .collect();
    assert!(!guard.accepts(DocumentId(1), ids[0], &corpus));
    assert!(guard.accepts(DocumentId(1), ids[1], &corpus));
}

#[test]
fn audit_catches_near_duplicates_under_another_document_id() {
    // The case document provenance alone misses: the same text re-hosted
    // under a different id. This is why the audit exists.
    let training = [10, 11, 12, 13, 14, 15];
    let corpus = corpus_with(
        &[
            (1, &[10, 11, 12, 13, 14, 15]), // the training document itself
            (2, &[10, 11, 12, 13, 14, 15]), // a verbatim copy elsewhere
            (3, &[90, 91, 92, 93, 94, 95]), // unrelated
        ],
        6,
    );
    let mut guard = LeakageGuard::by_source_document();
    let filter = NgramOverlapFilter::new(3, 0.1).expect("valid");
    let excluded = guard.audit(DocumentId(1), &training, &corpus, filter);
    assert_eq!(excluded, 2, "both copies should be caught");

    let ids: Vec<_> = (0..3)
        .map(anaphora::retrieval::corpus::NeighbourId)
        .collect();
    assert!(!guard.accepts(DocumentId(1), ids[0], &corpus));
    assert!(!guard.accepts(DocumentId(1), ids[1], &corpus));
    assert!(guard.accepts(DocumentId(1), ids[2], &corpus));
}

#[test]
fn exclusions_apply_during_search_so_k_is_still_met() {
    // Filtering after the search would silently return fewer than k
    // neighbours — a change in the experiment, not in the safety margin.
    let mut corpus = NeighbourCorpus::new(2, 1);
    for (i, doc) in [1u64, 1, 2, 3].iter().enumerate() {
        corpus
            .push(
                &NeighbourRecord {
                    document: DocumentId(*doc),
                    offset: 0,
                    tokens: vec![i as u32, i as u32],
                },
                &[i as f32],
            )
            .expect("shapes match");
    }
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();

    // Query nearest to the two excluded entries (embeddings 0.0 and 1.0).
    let mut accept = guard.accept_fn(DocumentId(1), &corpus);
    let found = index.search(&[0.0], 2, &mut accept);
    assert_eq!(found.len(), 2, "search must still return k neighbours");
    for candidate in &found {
        assert_ne!(corpus.document(candidate.id), Some(DocumentId(1)));
    }
}

#[test]
fn view_mismatch_is_detected() {
    use anaphora::chunk::{ChunkAdmission, ChunkedView, RetrieverEncode, chunk_queries};
    use anaphora::config::CcaConfig;
    use anaphora::schedule::Phase;

    struct ConstEncoder;
    impl RetrieverEncode for ConstEncoder {
        fn query_dim(&self) -> usize {
            1
        }
        fn encode_chunk(
            &mut self,
            chunk: anaphora::chunk::Chunk<'_>,
            _t: NoiseLevel,
            out: &mut Vec<f32>,
        ) {
            out.push(chunk.index() as f32);
        }
    }

    let cfg = CcaConfig::new(4, 2, 1, 2, 1, 1, 2, 1, 0).expect("valid");
    let clean = CleanSequence::new(vec![1, 2, 3, 4]);
    let view_a = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i == 0);
    let view_b = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i == 0);

    let chunked = ChunkedView::new(&view_a, cfg).expect("aligned");
    let queries = chunk_queries(
        chunked,
        Phase::Training,
        ChunkAdmission::permissive(),
        &mut ConstEncoder,
    )
    .expect("gate open at t=0.5");

    assert!(check_same_view(&view_a, &queries).is_ok());
    // Queries built from a *different* view of the same sequence are
    // rejected — this is the "cleaner query" leak that type inertness alone
    // does not cover.
    assert!(check_same_view(&view_b, &queries).is_err());
}

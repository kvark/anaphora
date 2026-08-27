//! Documents to training windows and to a retrievable, searchable corpus.

use anaphora::chunk::{ChunkAdmission, ChunkedView, chunk_queries};
use anaphora::config::CcaConfig;
use anaphora::corpus::{
    ChunkEmbedder, Document, HashedBagEmbedder, build_corpus, training_sequences,
};
use anaphora::retrieval::corpus::{DocumentId, NeighbourId};
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::retrieval::retrieve;
use anaphora::schedule::{NoiseLevel, Phase};
use anaphora::view::{MaskToken, NoisedView};

const MASK: MaskToken = MaskToken(0);
const PAD: u32 = 1;
const DIM: usize = 64;

/// `n = 16`, `m = 4` → `l = 4`, `k = 2`, `r = 8` (= 2m).
fn cfg() -> CcaConfig {
    CcaConfig::new(16, 4, 2, 8, 1, 1, 8, 1, 0).expect("valid")
}

fn doc(id: u64, tokens: Vec<u32>) -> Document {
    Document {
        id: DocumentId(id),
        tokens,
    }
}

fn embedder() -> HashedBagEmbedder {
    HashedBagEmbedder::bigram(DIM, MASK.0)
}

#[test]
fn long_documents_yield_windows_that_keep_their_parent_id() {
    // A window is not a new document for leakage purposes.
    let docs = vec![doc(7, (10..10 + 40).collect())];
    let seqs = training_sequences(&docs, 16, PAD, 1);
    assert_eq!(seqs.len(), 3, "40 tokens over 16 is three windows");
    assert!(seqs.iter().all(|s| s.document == DocumentId(7)));
    assert_eq!(seqs[0].content_len, 16);
    assert_eq!(seqs[2].content_len, 8, "the tail window is short");
    assert_eq!(&seqs[2].tokens[8..], &[PAD; 8], "and padded");
}

#[test]
fn padding_is_never_masked() {
    // A model that learns to predict pad tokens is spending capacity on the
    // data loader.
    let docs = vec![doc(1, (10..16).collect())];
    let seqs = training_sequences(&docs, 16, PAD, 1);
    let seq = &seqs[0];
    let mut mask = seq.content_mask(|_| true);
    let masked: Vec<bool> = (0..16).map(&mut mask).collect();
    assert_eq!(masked.iter().filter(|&&m| m).count(), 6);
    assert!(masked[..6].iter().all(|&m| m));
    assert!(masked[6..].iter().all(|&m| !m));
}

#[test]
fn short_windows_can_be_dropped() {
    let docs = vec![doc(1, (10..10 + 20).collect())];
    assert_eq!(training_sequences(&docs, 16, PAD, 1).len(), 2);
    assert_eq!(
        training_sequences(&docs, 16, PAD, 5).len(),
        1,
        "the 4-token tail window is below min_content"
    );
}

#[test]
fn records_tile_at_stride_m_and_drop_the_short_tail() {
    let cfg = cfg();
    let docs = vec![doc(3, (100..100 + 20).collect())];
    let corpus = build_corpus(&docs, cfg, &mut embedder());

    // offsets 0, 4, 8, 12 all have r=8 tokens available; 16 does not.
    assert_eq!(corpus.len(), 4);
    for i in 0..corpus.len() {
        let id = NeighbourId(i as u32);
        assert_eq!(corpus.offset(id), Some(i * cfg.chunk_size()));
        assert_eq!(
            corpus.tokens(id).map(<[u32]>::len),
            Some(cfg.neighbour_len())
        );
        assert_eq!(corpus.document(id), Some(DocumentId(3)));
    }
    // A record padded into its continuation would promise information it does
    // not have, so the tail is dropped rather than padded.
    assert_eq!(
        corpus.tokens(NeighbourId(3)),
        Some(&[112u32, 113, 114, 115, 116, 117, 118, 119][..])
    );
}

#[test]
fn only_the_matched_chunk_is_embedded() {
    // The continuation is what the model does not have yet. Indexing it would
    // make search match on text the query cannot possibly contain.
    let cfg = cfg();
    let shared_prefix: Vec<u32> = (200..204).collect();

    let mut a = shared_prefix.clone();
    a.extend([900, 901, 902, 903]);
    let mut b = shared_prefix.clone();
    b.extend([700, 701, 702, 703]);

    let corpus = build_corpus(&[doc(1, a), doc(2, b)], cfg, &mut embedder());
    assert_eq!(corpus.len(), 2);
    let ea = corpus.embedding(NeighbourId(0)).expect("present");
    let eb = corpus.embedding(NeighbourId(1)).expect("present");
    assert_eq!(ea, eb, "same matched chunk must give the same embedding");
    assert_ne!(
        corpus.tokens(NeighbourId(0)),
        corpus.tokens(NeighbourId(1)),
        "but the continuations differ"
    );
}

#[test]
fn embeddings_are_unit_length_and_deterministic() {
    let mut e = embedder();
    let mut a = Vec::new();
    let mut b = Vec::new();
    e.embed(&[5, 6, 7, 8], None, &mut a);
    e.embed(&[5, 6, 7, 8], None, &mut b);
    assert_eq!(a, b, "the corpus is embedded once and queried later");
    let norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "L2-normalised, got {norm}");
}

#[test]
fn masking_degrades_a_query_rather_than_corrupting_it() {
    // The property that makes a lexical embedder a usable answer to the
    // open encoder question: a half-masked chunk yields the embedding of the
    // half that survived — a weaker query, not a wrong one.
    let mut e = embedder();
    let clean: Vec<u32> = (300..308).collect();
    let mut half_masked = clean.clone();
    for slot in half_masked.iter_mut().take(4) {
        *slot = MASK.0;
    }
    let unrelated: Vec<u32> = (900..908).collect();

    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    e.embed(&clean, None, &mut a);
    e.embed(&half_masked, Some(NoiseLevel::new(0.5).unwrap()), &mut b);
    e.embed(&unrelated, None, &mut c);

    let dot = |x: &[f32], y: &[f32]| -> f32 { x.iter().zip(y).map(|(p, q)| p * q).sum() };
    assert!(
        dot(&a, &b) > dot(&a, &c),
        "a masked chunk must stay nearer its own clean form than a stranger"
    );
    assert!(dot(&a, &b) > 0.0, "and must retain some signal at all");
}

#[test]
fn a_fully_masked_chunk_embeds_to_nothing() {
    // Which is exactly why the hard gate and per-chunk admission exist: there
    // is no signal here to search with.
    let mut e = embedder();
    let mut v = Vec::new();
    e.embed(&[MASK.0; 8], Some(NoiseLevel::MASKED), &mut v);
    assert!(v.iter().all(|&x| x == 0.0));
}

#[test]
fn end_to_end_retrieval_finds_the_matching_passage() {
    let cfg = cfg();
    // Three documents; the query will be built from text matching doc 20.
    let target: Vec<u32> = (500..516).collect();
    let docs = vec![
        doc(10, (100..116).collect()),
        doc(20, target.clone()),
        doc(30, (800..816).collect()),
    ];
    let mut e = embedder();
    let corpus = build_corpus(&docs, cfg, &mut e);
    let index = ExactIndex::build(&corpus);
    assert_eq!(index.len(), corpus.len());

    // Query from a lightly masked view of the target text.
    let mut probe = target.clone();
    probe[3] = MASK.0;
    let view = NoisedView::from_tokens(probe, NoiseLevel::new(0.2).unwrap(), MASK);
    let chunked = ChunkedView::new(&view, cfg).expect("aligned");
    let queries = chunk_queries(
        chunked,
        Phase::Inference,
        ChunkAdmission::permissive(),
        &mut e,
    )
    .expect("gate open at t=0.2");

    // Searching as an unrelated document, so nothing is excluded.
    let guard = LeakageGuard::by_source_document();
    let found = retrieve(&queries, cfg, &index, &corpus, &guard, DocumentId(999));
    assert!(found.chunk_has_neighbours(0));
    let top = found.id(0, 0).expect("a neighbour");
    assert_eq!(
        corpus.document(top),
        Some(DocumentId(20)),
        "the nearest passage should come from the document the query quotes"
    );
}

#[test]
fn a_document_does_not_retrieve_itself() {
    let cfg = cfg();
    let target: Vec<u32> = (500..516).collect();
    let docs = vec![doc(20, target.clone()), doc(30, (800..816).collect())];
    let mut e = embedder();
    let corpus = build_corpus(&docs, cfg, &mut e);
    let index = ExactIndex::build(&corpus);

    let view = NoisedView::from_tokens(target, NoiseLevel::new(0.2).unwrap(), MASK);
    let chunked = ChunkedView::new(&view, cfg).expect("aligned");
    let queries = chunk_queries(
        chunked,
        Phase::Inference,
        ChunkAdmission::permissive(),
        &mut e,
    )
    .expect("gate open");

    let guard = LeakageGuard::by_source_document();
    // Now searching *as* document 20 — its own passages must be excluded.
    let found = retrieve(&queries, cfg, &index, &corpus, &guard, DocumentId(20));
    for chunk in 0..cfg.num_chunks() {
        for slot in 0..cfg.neighbours_per_chunk() {
            if let Some(id) = found.id(chunk, slot) {
                assert_ne!(
                    corpus.document(id),
                    Some(DocumentId(20)),
                    "the training document leaked into its own results"
                );
            }
        }
    }
}

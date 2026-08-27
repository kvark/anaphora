//! Chunking the noised view and building queries from it.

use anaphora::chunk::{Chunk, ChunkAdmission, ChunkedView, RetrieverEncode, chunk_queries};
use anaphora::config::CcaConfig;
use anaphora::schedule::{NoiseLevel, Phase};
use anaphora::view::{CleanSequence, MaskToken, NoisedView};

const MASK: MaskToken = MaskToken(0);

/// Records which chunks it was asked to encode, so a test can assert that a
/// skipped chunk was never handed to the retriever at all.
struct RecordingEncoder {
    seen: Vec<usize>,
    dim: usize,
}

impl RetrieverEncode for RecordingEncoder {
    fn query_dim(&self) -> usize {
        self.dim
    }
    fn encode_chunk(&mut self, chunk: Chunk<'_>, _t: NoiseLevel, out: &mut Vec<f32>) {
        self.seen.push(chunk.index());
        out.extend(std::iter::repeat_n(chunk.index() as f32, self.dim));
    }
}

fn cfg() -> CcaConfig {
    CcaConfig::new(8, 4, 1, 4, 1, 1, 4, 1, 0).expect("valid")
}

#[test]
fn chunks_partition_the_sequence() {
    let view = NoisedView::from_tokens(vec![1, 2, 3, 4, 5, 6, 7, 8], NoiseLevel::CLEAN, MASK);
    let chunked = ChunkedView::new(&view, cfg()).expect("aligned");
    assert_eq!(chunked.len(), 2);
    assert_eq!(chunked.get(0).unwrap().tokens(), &[1, 2, 3, 4]);
    assert_eq!(chunked.get(1).unwrap().tokens(), &[5, 6, 7, 8]);
    assert!(chunked.get(2).is_none());
}

#[test]
fn misaligned_view_is_rejected() {
    let view = NoisedView::from_tokens(vec![1, 2, 3], NoiseLevel::CLEAN, MASK);
    assert!(ChunkedView::new(&view, cfg()).is_err());
}

#[test]
fn chunk_mask_rate_is_local() {
    // A global t says the average chunk has signal; it says nothing about a
    // chunk the masking process happened to flatten.
    let clean = CleanSequence::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let view = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i < 4);
    let chunked = ChunkedView::new(&view, cfg()).expect("aligned");
    assert_eq!(chunked.get(0).unwrap().mask_rate(), 1.0);
    assert_eq!(chunked.get(1).unwrap().mask_rate(), 0.0);
}

#[test]
fn fully_masked_chunk_is_not_sent_to_the_retriever() {
    let clean = CleanSequence::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let view = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i < 4);
    let chunked = ChunkedView::new(&view, cfg()).expect("aligned");

    let mut encoder = RecordingEncoder {
        seen: Vec::new(),
        dim: 2,
    };
    let queries = chunk_queries(
        chunked,
        Phase::Training,
        ChunkAdmission::new(0.85).unwrap(),
        &mut encoder,
    )
    .expect("gate open at t=0.5");

    assert_eq!(encoder.seen, vec![1], "the flattened chunk must be skipped");
    assert!(!queries.is_admitted(0));
    assert!(queries.is_admitted(1));
    assert_eq!(queries.num_admitted(), 1);
    assert_eq!(queries.admitted_indices().collect::<Vec<_>>(), vec![1]);
}

#[test]
fn skipped_chunks_keep_the_layout_dense() {
    // A skipped chunk still occupies its slot, so `query(i)` stays a slice
    // index rather than becoming a lookup through a sparse map.
    let clean = CleanSequence::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let view = clean.mask_with(NoiseLevel::new(0.5).unwrap(), MASK, |i| i < 4);
    let chunked = ChunkedView::new(&view, cfg()).expect("aligned");
    let mut encoder = RecordingEncoder {
        seen: Vec::new(),
        dim: 2,
    };
    let queries = chunk_queries(
        chunked,
        Phase::Training,
        ChunkAdmission::new(0.85).unwrap(),
        &mut encoder,
    )
    .unwrap();
    assert_eq!(queries.len(), 2);
    assert_eq!(queries.query(0), Some(&[0.0, 0.0][..]));
    assert_eq!(queries.query(1), Some(&[1.0, 1.0][..]));
}

#[test]
fn closed_hard_gate_spends_no_index_traffic() {
    let clean = CleanSequence::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // t = 0.05: open at inference, closed during training.
    let view = clean.mask_with(NoiseLevel::new(0.05).unwrap(), MASK, |i| i == 0);
    let chunked = ChunkedView::new(&view, cfg()).expect("aligned");

    let mut encoder = RecordingEncoder {
        seen: Vec::new(),
        dim: 2,
    };
    assert!(
        chunk_queries(
            chunked,
            Phase::Training,
            ChunkAdmission::permissive(),
            &mut encoder
        )
        .is_none()
    );
    assert!(encoder.seen.is_empty(), "no chunk should have been encoded");

    assert!(
        chunk_queries(
            chunked,
            Phase::Inference,
            ChunkAdmission::permissive(),
            &mut encoder
        )
        .is_some()
    );
    assert_eq!(encoder.seen, vec![0, 1]);
}

#[test]
fn config_rejects_impossible_shapes() {
    // n not a multiple of m.
    assert!(CcaConfig::new(10, 4, 1, 4, 1, 1, 4, 1, 0).is_err());
    // num_heads not a multiple of num_kv_heads.
    assert!(CcaConfig::new(8, 4, 1, 4, 3, 2, 4, 1, 0).is_err());
    // zero chunk size.
    assert!(CcaConfig::new(8, 0, 1, 4, 1, 1, 4, 1, 0).is_err());
}

#[test]
fn cca_layers_follow_the_insertion_period() {
    // RETRO's P=3 from layer 6.
    let cfg = CcaConfig::retro_like(8, 8, 64).expect("valid");
    assert_eq!(cfg.cca_layers(12), vec![6, 9]);
    assert_eq!(cfg.num_chunks(), 32);
    assert_eq!(cfg.neighbour_kv_rows(), 2 * 128);
}

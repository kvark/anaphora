//! The denoising loop: refresh scheduling, revealing, and the retrieval
//! bookkeeping around them.

use anaphora::chunk::{Chunk, RetrieverEncode};
use anaphora::config::CcaConfig;
use anaphora::retrieval::Neighbours;
use anaphora::retrieval::corpus::{DocumentId, NeighbourCorpus, NeighbourRecord};
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::sample::{
    Denoiser, RetrievalContext, RevealPolicy, SamplingConfig, sample, unmask_top_confidence,
};
use anaphora::schedule::{NoiseLevel, RefreshSchedule};
use anaphora::view::{MaskToken, NoisedView};

const MASK: MaskToken = MaskToken(0);
const SEQ: usize = 8;
const VOCAB: usize = 16;

fn cfg() -> CcaConfig {
    CcaConfig::new(SEQ, 4, 1, 4, 1, 1, 4, 1, 0).expect("valid")
}

struct MeanEncoder;
impl RetrieverEncode for MeanEncoder {
    fn query_dim(&self) -> usize {
        1
    }
    fn encode_chunk(&mut self, chunk: Chunk<'_>, _t: NoiseLevel, out: &mut Vec<f32>) {
        let sum: u32 = chunk.tokens().iter().sum();
        out.push(sum as f32 / chunk.tokens().len() as f32);
    }
}

/// Returns logits that always favour token `position % VOCAB`, with a
/// confidence that rises with position so the reveal order is predictable.
struct RampDenoiser {
    calls: usize,
    saw_neighbours: Vec<bool>,
}

impl Denoiser for RampDenoiser {
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn logits(&mut self, view: &NoisedView, neighbours: Option<&Neighbours>) -> Vec<f32> {
        self.calls += 1;
        self.saw_neighbours.push(neighbours.is_some());
        let mut out = vec![0.0f32; view.len() * VOCAB];
        for pos in 0..view.len() {
            out[pos * VOCAB + (pos % VOCAB)] = 1.0 + pos as f32;
        }
        out
    }
}

fn corpus_and_index() -> (NeighbourCorpus, ExactIndex) {
    let mut corpus = NeighbourCorpus::new(4, 1);
    for i in 0..4u32 {
        corpus
            .push(
                &NeighbourRecord {
                    document: DocumentId(100 + i as u64),
                    offset: 0,
                    tokens: vec![i + 1; 4],
                },
                &[i as f32],
            )
            .expect("shapes match");
    }
    let index = ExactIndex::build(&corpus);
    (corpus, index)
}

#[test]
fn confidence_ranks_by_peakedness_not_raw_logit() {
    // Two rows with the same maximum logit but different spreads: the peaked
    // one is the confident one. Ranking by raw max logit would call them
    // equal.
    let view = NoisedView::all_masked(2, MASK);
    let mut logits = vec![0.0f32; 2 * VOCAB];
    logits[5] = 3.0; // row 0: one spike, rest zero — peaked
    for v in logits[VOCAB..2 * VOCAB].iter_mut() {
        *v = 3.0; // row 1: uniformly 3.0 — maximally unsure
    }
    logits[VOCAB + 7] = 3.0;

    let picks = unmask_top_confidence(&view, &logits, VOCAB, 1);
    assert_eq!(picks, vec![(0, 5)], "the peaked row must win");
}

#[test]
fn only_masked_positions_are_revealed() {
    let view = NoisedView::from_tokens(vec![9, 0, 0, 0], NoiseLevel::new(0.5).unwrap(), MASK);
    let mut logits = vec![0.0f32; 4 * VOCAB];
    // Make position 0 look maximally confident — it is already revealed.
    logits[1] = 100.0;
    for pos in 1..4 {
        logits[pos * VOCAB + pos] = 1.0;
    }
    let picks = unmask_top_confidence(&view, &logits, VOCAB, 4);
    assert_eq!(picks.len(), 3);
    assert!(picks.iter().all(|&(pos, _)| pos != 0));
}

#[test]
fn linear_reveal_finishes_the_trajectory() {
    let policy = RevealPolicy::Linear;
    let mut masked = 10usize;
    let steps = 4;
    for step in 0..steps {
        masked -= policy.count(steps - step, masked);
    }
    assert_eq!(
        masked, 0,
        "every position must be revealed by the last step"
    );
}

#[test]
fn trajectory_reveals_everything_and_refreshes_on_schedule() {
    let (corpus, index) = corpus_and_index();
    let guard = LeakageGuard::by_source_document();
    let mut encoder = MeanEncoder;
    let mut retrieval = RetrievalContext {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        encoder: &mut encoder,
    };
    let mut denoiser = RampDenoiser {
        calls: 0,
        saw_neighbours: Vec::new(),
    };
    let mut sampling = SamplingConfig {
        steps: 8,
        refresh: RefreshSchedule::new(&[0.8, 0.5, 0.25]),
        admission: anaphora::chunk::ChunkAdmission::permissive(),
        reveal: RevealPolicy::Linear,
        document: DocumentId(1),
    };

    let (view, trace) = sample(
        &[7, 7],
        MASK,
        cfg(),
        &mut sampling,
        &mut retrieval,
        &mut denoiser,
    );

    assert_eq!(trace.steps, 8);
    assert_eq!(view.num_masked(), 0, "the trajectory must finish clean");
    assert_eq!(&view.tokens()[..2], &[7, 7], "the prompt must survive");
    assert_eq!(denoiser.calls, 8);

    // t starts at 1.0, where the inference band (t < 0.9) is closed, so the
    // first step runs without neighbours; later steps retrieve.
    assert!(!denoiser.saw_neighbours[0]);
    assert!(denoiser.saw_neighbours.iter().any(|&seen| seen));
    assert!(trace.refreshes >= 1);
    assert!(trace.gated_out >= 1);
}

#[test]
fn empty_neighbour_block_reports_no_chunks() {
    let (corpus, index) = corpus_and_index();
    let mut guard = LeakageGuard::by_source_document();
    guard.audit(
        DocumentId(1),
        &[1, 2, 3, 4, 5],
        &corpus,
        anaphora::retrieval::leakage::NgramOverlapFilter::new(1, 0.0).unwrap(),
    );
    let view = NoisedView::from_tokens(vec![1; SEQ], NoiseLevel::new(0.5).unwrap(), MASK);
    let chunked = anaphora::chunk::ChunkedView::new(&view, cfg()).unwrap();
    let mut encoder = MeanEncoder;
    let queries = anaphora::chunk::chunk_queries(
        chunked,
        anaphora::schedule::Phase::Inference,
        anaphora::chunk::ChunkAdmission::permissive(),
        &mut encoder,
    )
    .unwrap();
    let neighbours =
        anaphora::retrieval::retrieve(&queries, cfg(), &index, &corpus, &guard, DocumentId(1));
    assert_eq!(neighbours.num_filled(), 0);
    assert_eq!(neighbours.chunk_mask(), vec![false, false]);
}

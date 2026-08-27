//! The host training loop, end to end on the GPU.

use anaphora::config::CcaConfig;
use anaphora::corpus::{Document, HashedBagEmbedder, build_corpus, training_sequences};
use anaphora::model::backbone::BackboneConfig;
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::corpus::{DocumentId, NeighbourCorpus};
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::train::{
    NoiseSampler, Optimizer, RetrievalSources, Rng, Trainer, TrainingConfig, apply_zero_init,
    configure_optimizer,
};
use anaphora::view::MaskToken;
use meganeura::Graph;

const MASK: MaskToken = MaskToken(0);
const PAD: u32 = 1;
const VOCAB: usize = 48;
const SEQ: usize = 16;
const DIM: usize = 32;

fn cfg() -> CcaConfig {
    // n=16, m=4 -> l=4, k=2, r=8; d = 2 heads * 16 = 32. CCA after layer 1.
    CcaConfig::new(SEQ, 4, 2, 8, 2, 2, 16, 2, 1).expect("valid")
}

fn model_config(cca: CcaConfig) -> ModelConfig {
    ModelConfig {
        cca,
        backbone: BackboneConfig {
            vocab_size: VOCAB,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        },
        encoder: NeighbourEncoderConfig {
            vocab_size: VOCAB,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 32,
            scope: EncoderScope::PerNeighbour,
        },
        activation: GateActivation::Tanh,
        neighbours: NeighbourInput::Encoded,
    }
}

fn documents() -> Vec<Document> {
    // Tokens from 2 up, so they never collide with MASK(0) or PAD(1).
    (0..6u64)
        .map(|d| Document {
            id: DocumentId(d),
            tokens: (0..SEQ * 2)
                .map(|i| (2 + (d as usize * 5 + i * 3) % (VOCAB - 2)) as u32)
                .collect(),
        })
        .collect()
}

fn build_retrieval(cca: CcaConfig) -> (NeighbourCorpus, HashedBagEmbedder) {
    let mut embedder = HashedBagEmbedder::bigram(DIM, MASK.0);
    let corpus = build_corpus(&documents(), cca, &mut embedder);
    (corpus, embedder)
}

fn build_session(cca: CcaConfig) -> (meganeura::runtime::Session, CcaModel) {
    let mut g = Graph::new();
    let model = CcaModel::build(&mut g, model_config(cca));
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);
    let session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    (session, model)
}

fn seed_parameters(session: &mut meganeura::runtime::Session, seed: u64) {
    let names: Vec<String> = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        let len = session.param_size(&name).expect("declared");
        let mixed = name
            .bytes()
            .fold(0u64, |a, b| a.rotate_left(5) ^ u64::from(b));
        let mut rng = Rng::new(seed ^ mixed);
        let is_norm = name.contains("norm");
        let values: Vec<f32> = (0..len)
            .map(|_| {
                if is_norm {
                    1.0
                } else {
                    (rng.next_f32() - 0.5) * 0.2
                }
            })
            .collect();
        session.set_parameter(&name, &values);
    }
}

#[test]
fn rng_is_reproducible_from_a_seed() {
    let mut r = Rng::new(42);
    let stream: Vec<f32> = (0..64).map(|_| r.next_f32()).collect();
    assert!(stream.iter().all(|&v| (0.0..1.0).contains(&v)));
    let mut again = Rng::new(42);
    let repeat: Vec<f32> = (0..64).map(|_| again.next_f32()).collect();
    assert_eq!(stream, repeat, "a run must be reproducible from its seed");
    let mut other = Rng::new(43);
    let different: Vec<f32> = (0..64).map(|_| other.next_f32()).collect();
    assert_ne!(stream, different);
}

#[test]
fn corrupt_never_masks_padding() {
    let docs = vec![Document {
        id: DocumentId(1),
        tokens: (2..8).collect(),
    }];
    let seqs = training_sequences(&docs, SEQ, PAD, 1);
    let mut trainer = Trainer::new(
        TrainingConfig {
            noise: NoiseSampler::Fixed(1.0),
            ..TrainingConfig::new(cfg(), VOCAB, MASK)
        },
        7,
    );
    let (_, view) = trainer.corrupt(&seqs[0]);
    assert_eq!(view.num_masked(), 6, "only the real tokens");
    assert!(view.masked()[6..].iter().all(|&m| !m));
}

#[test]
fn a_training_step_runs_and_reduces_loss() {
    let cca = cfg();
    let (corpus, mut embedder) = build_retrieval(cca);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();
    let seqs = training_sequences(&documents(), SEQ, PAD, SEQ);
    assert!(!seqs.is_empty());

    let (mut session, model) = build_session(cca);
    seed_parameters(&mut session, 0xA11CE);
    // Without this the retrofit begins by pushing an untrained
    // cross-attention output into the frozen residual stream.
    apply_zero_init(&mut session, &model);
    configure_optimizer(&mut session, Optimizer::adam(3e-3));

    let mut trainer = Trainer::new(
        TrainingConfig {
            // Fixed t keeps the comparison from being dominated by the 1/t
            // weight differing between the two windows.
            noise: NoiseSampler::Fixed(0.5),
            ..TrainingConfig::new(cca, VOCAB, MASK)
        },
        0xBEEF,
    );
    let mut sources = RetrievalSources {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        embedder: &mut embedder,
    };

    let mut losses = Vec::new();
    let mut retrieved_any = false;
    for i in 0..80 {
        let seq = &seqs[i % seqs.len()];
        if let Some(report) = trainer
            .step(&mut session, &model, seq, &mut sources)
            .expect("step is well-formed")
        {
            assert!(report.loss.is_finite(), "loss went non-finite at step {i}");
            assert!(report.scored > 0);
            retrieved_any |= report.chunks_retrieved > 0;
            losses.push(report.loss);
        }
    }

    assert!(losses.len() > 40, "most steps should score something");
    assert!(retrieved_any, "the retrieval path never fired");

    let window = 10;
    let first: f32 = losses[..window].iter().sum::<f32>() / window as f32;
    let last: f32 = losses[losses.len() - window..].iter().sum::<f32>() / window as f32;
    eprintln!("loss {first:.4} -> {last:.4} over {} steps", losses.len());
    assert!(last < first, "loss did not fall: {first:.4} -> {last:.4}");

    // Zero-init is only safe if the gate can still open.
    let moved = model.zero_init_param_names().iter().any(|name| {
        let len = session.param_size(name).expect("declared");
        let mut v = vec![0.0f32; len];
        session.read_param(name, &mut v);
        v.iter().any(|&x| x != 0.0)
    });
    assert!(moved, "the zero-initialised gate never left zero");
}

#[test]
fn the_no_retrieval_baseline_runs_from_the_same_model() {
    // The protocol's gate-zero comparison needs no second model.
    let cca = cfg();
    let (corpus, mut embedder) = build_retrieval(cca);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();
    let seqs = training_sequences(&documents(), SEQ, PAD, SEQ);

    let (mut session, model) = build_session(cca);
    seed_parameters(&mut session, 0xB0B);
    apply_zero_init(&mut session, &model);
    configure_optimizer(&mut session, Optimizer::adam(1e-3));

    let mut trainer = Trainer::new(
        TrainingConfig {
            retrieval_enabled: false,
            ..TrainingConfig::new(cca, VOCAB, MASK)
        },
        1,
    );
    let mut sources = RetrievalSources {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        embedder: &mut embedder,
    };

    let mut ran = 0;
    for seq in seqs.iter().take(12) {
        if let Some(report) = trainer
            .step(&mut session, &model, seq, &mut sources)
            .expect("well-formed")
        {
            assert!(!report.gate_open, "retrieval was disabled");
            assert_eq!(report.chunks_retrieved, 0);
            assert!(report.loss.is_finite());
            ran += 1;
        }
    }
    assert!(ran > 0);
}

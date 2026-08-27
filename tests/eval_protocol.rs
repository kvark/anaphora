//! The evaluation protocol: its arithmetic, and one end-to-end run.

use anaphora::config::CcaConfig;
use anaphora::corpus::{Document, HashedBagEmbedder, build_corpus, training_sequences};
use anaphora::eval::{
    BandLoss, ConditionReport, EvalReport, Evaluator, NeighbourCondition, eval_overlap,
};
use anaphora::model::backbone::BackboneConfig;
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::corpus::DocumentId;
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::schedule::NoiseLevel;
use anaphora::train::{RetrievalSources, Rng};
use anaphora::view::MaskToken;
use meganeura::Graph;

const MASK: MaskToken = MaskToken(0);
const PAD: u32 = 1;
const VOCAB: usize = 48;
const SEQ: usize = 16;
const DIM: usize = 32;

fn report(condition: NeighbourCondition, mean: f32) -> ConditionReport {
    ConditionReport {
        condition,
        mean_loss: mean,
        bands: vec![BandLoss {
            low: 0.0,
            high: 1.0,
            mean_loss: mean,
            steps: 1,
        }],
    }
}

#[test]
fn a_healthy_model_lands_random_near_ablated() {
    // Relevant neighbours help; irrelevant ones simply stop helping, because
    // the gate can learn to shut. The gap should not run far past the gain.
    let r = EvalReport {
        conditions: vec![
            report(NeighbourCondition::Real, 3.0),
            report(NeighbourCondition::Random, 3.4),
            report(NeighbourCondition::Ablated, 3.5),
            report(NeighbourCondition::Oracle, 2.5),
        ],
    };
    assert!((r.retrieval_gain().unwrap() - 0.5).abs() < 1e-6);
    assert!((r.copy_gap().unwrap() - 0.4).abs() < 1e-6);
    assert!(r.copy_ratio().unwrap() < 1.0, "gap stays inside the gain");
    assert_eq!(r.random_worse_than_ablated(), Some(false));
}

#[test]
fn a_copying_model_is_worse_than_ablated_on_random_neighbours() {
    // The sharpest single signature. A model transcribing neighbour content
    // it cannot evaluate does actively worse than one given no neighbours.
    let r = EvalReport {
        conditions: vec![
            report(NeighbourCondition::Real, 1.0),
            report(NeighbourCondition::Random, 6.0),
            report(NeighbourCondition::Ablated, 3.5),
            report(NeighbourCondition::Oracle, 0.8),
        ],
    };
    assert!((r.retrieval_gain().unwrap() - 2.5).abs() < 1e-6);
    assert!((r.copy_gap().unwrap() - 5.0).abs() < 1e-6);
    assert!(r.copy_ratio().unwrap() > 1.0, "gap runs past the gain");
    assert_eq!(r.random_worse_than_ablated(), Some(true));
}

#[test]
fn no_learning_leaves_the_ratio_undefined() {
    // Which is itself a result: nothing was learned for anything to leak
    // through, so the ratio has no denominator.
    let r = EvalReport {
        conditions: vec![
            report(NeighbourCondition::Real, 3.5),
            report(NeighbourCondition::Random, 3.5),
            report(NeighbourCondition::Ablated, 3.5),
        ],
    };
    assert!(r.copy_ratio().is_none());
}

#[test]
fn band_means_weight_by_step_count() {
    let c = ConditionReport {
        condition: NeighbourCondition::Real,
        mean_loss: 0.0,
        bands: vec![
            BandLoss {
                low: 0.0,
                high: 0.2,
                mean_loss: 1.0,
                steps: 3,
            },
            BandLoss {
                low: 0.2,
                high: 0.4,
                mean_loss: 2.0,
                steps: 1,
            },
            BandLoss {
                low: 0.4,
                high: 1.0,
                mean_loss: 9.0,
                steps: 5,
            },
        ],
    };
    // (1*3 + 2*1) / 4
    assert!((c.mean_below(0.4).unwrap() - 1.25).abs() < 1e-6);
    assert!(
        c.mean_below(0.1).is_none(),
        "no band lies entirely below 0.1"
    );
}

#[test]
fn overlap_finds_an_evaluation_document_that_is_in_the_index() {
    // Retrieval papers leak here routinely, and it is cheap to rule out.
    let cca = CcaConfig::new(SEQ, 4, 2, 8, 1, 1, 8, 1, 0).expect("valid");
    let indexed: Vec<u32> = (10..10 + SEQ as u32).collect();
    let corpus = build_corpus(
        &[Document {
            id: DocumentId(1),
            tokens: indexed.clone(),
        }],
        cca,
        &mut HashedBagEmbedder::bigram(DIM, MASK.0),
    );

    let duplicate = Document {
        id: DocumentId(99),
        tokens: indexed,
    };
    let novel = Document {
        id: DocumentId(98),
        tokens: (500..500 + SEQ as u32).collect(),
    };
    let seqs = training_sequences(&[duplicate, novel], SEQ, PAD, 1);
    let overlap = eval_overlap(&seqs, &corpus, 3);

    assert!(
        overlap[0] > 0.5,
        "the duplicate should be caught: {}",
        overlap[0]
    );
    assert_eq!(overlap[1], 0.0, "the novel document should not be");
}

fn cfg() -> CcaConfig {
    CcaConfig::new(SEQ, 4, 2, 8, 2, 2, 16, 2, 1).expect("valid")
}

fn documents() -> Vec<Document> {
    (0..6u64)
        .map(|d| Document {
            id: DocumentId(d),
            tokens: (0..SEQ * 2)
                .map(|i| (2 + (d as usize * 5 + i * 3) % (VOCAB - 2)) as u32)
                .collect(),
        })
        .collect()
}

#[test]
fn the_protocol_runs_and_separates_its_conditions() {
    let cca = cfg();
    let mut embedder = HashedBagEmbedder::bigram(DIM, MASK.0);
    let corpus = build_corpus(&documents(), cca, &mut embedder);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();
    let seqs = training_sequences(&documents(), SEQ, PAD, SEQ);

    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
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
        },
    );
    let labels = g.input("labels", &[SEQ, VOCAB]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    // Open the gate, so the conditions can differ at all: a zero-init gate
    // makes every one of them identical by construction, which is exactly
    // what zero-init is for and exactly wrong here.
    //
    // The weights must also *vary*. Filling every parameter with one constant
    // makes the embedding tables constant, every token then embeds to the
    // same vector, and neighbour content cannot reach the output no matter
    // how the rest of the graph is wired -- the conditions come back
    // bit-identical and the failure reads as a disconnected CCA path.
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
        let mut rng = Rng::new(0x5EED ^ mixed);
        let is_norm = name.contains("norm");
        let values: Vec<f32> = (0..len)
            .map(|_| {
                if is_norm {
                    1.0
                } else {
                    (rng.next_f32() - 0.5) * 0.4
                }
            })
            .collect();
        session.set_parameter(&name, &values);
    }

    let mut evaluator = Evaluator::new(cca, VOCAB, MASK, 5);
    let mut sources = RetrievalSources {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        embedder: &mut embedder,
    };
    let levels: Vec<NoiseLevel> = [0.3f32, 0.5, 0.7]
        .iter()
        .map(|&t| NoiseLevel::new(t).unwrap())
        .collect();

    let report = evaluator.run(
        &mut session,
        &seqs,
        &levels,
        &NeighbourCondition::ALL,
        &mut sources,
    );
    eprintln!("{}", report.to_table());

    assert_eq!(report.conditions.len(), 4);
    for c in &report.conditions {
        assert!(
            c.mean_loss.is_finite(),
            "{:?} produced no number",
            c.condition
        );
    }
    // Different neighbour blocks must actually reach the model. If Real and
    // Ablated agree exactly, the retrieval path is not connected and every
    // downstream number is meaningless.
    let real = report.get(NeighbourCondition::Real).unwrap().mean_loss;
    let ablated = report.get(NeighbourCondition::Ablated).unwrap().mean_loss;
    let oracle = report.get(NeighbourCondition::Oracle).unwrap().mean_loss;
    assert_ne!(real, ablated, "retrieval made no difference at all");
    assert_ne!(oracle, real, "the oracle block made no difference");
}

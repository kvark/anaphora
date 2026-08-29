//! The shipped generate path: `sample::sample` through a session-backed
//! denoiser, not a reimplemented loop.

use anaphora::config::CcaConfig;
use anaphora::corpus::HashedBagEmbedder;
use anaphora::generate::{SessionDenoiser, decode_token_ids, generate};
use anaphora::model::backbone::BackboneConfig;
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::corpus::{DocumentId, NeighbourCorpus};
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::sample::{RetrievalContext, SamplingConfig};
use anaphora::train::apply_zero_init;
use anaphora::view::MaskToken;
use meganeura::Graph;

const MASK: MaskToken = MaskToken(0);
const VOCAB: usize = 48;
const SEQ: usize = 16;
const CHUNK: usize = 4;
const K: usize = 2;
const R: usize = 8;
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 16;
const EMBED: usize = 8;

fn cca() -> CcaConfig {
    CcaConfig::new(
        SEQ,
        CHUNK,
        K,
        R,
        HEADS as usize,
        HEADS as usize,
        HEAD_DIM as usize,
        2,
        1,
    )
    .expect("valid")
}

fn model_config(cca: CcaConfig) -> ModelConfig {
    ModelConfig {
        cca,
        backbone: BackboneConfig {
            vocab_size: VOCAB,
            num_layers: 2,
            num_heads: HEADS,
            num_kv_heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        },
        encoder: NeighbourEncoderConfig {
            vocab_size: VOCAB,
            num_layers: 1,
            num_heads: HEADS,
            num_kv_heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: 32,
            scope: EncoderScope::PerNeighbour,
        },
        activation: GateActivation::Tanh,
        neighbours: NeighbourInput::Encoded,
    }
}

fn param_values(name: &str, len: usize) -> Vec<f32> {
    let mut state = name.bytes().fold(0x2545_F491_4F6C_DD1Du64, |acc, b| {
        (acc ^ b as u64).wrapping_mul(0x0100_0000_01B3)
    });
    let is_norm = name.contains("norm");
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as f32 / 16_777_216.0) - 0.5;
            if is_norm {
                1.0 + unit * 0.1
            } else {
                unit * 0.2
            }
        })
        .collect()
}

fn fill_params(session: &mut meganeura::runtime::Session) {
    let names: Vec<String> = session
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect();
    for name in names {
        let len = session.param_size(&name).expect("declared parameter");
        session.set_parameter(&name, &param_values(&name, len));
    }
}

#[test]
fn generate_keeps_prompt_and_fills_masks() {
    let cca = cca();
    let mut g = Graph::new();
    let model = CcaModel::build(&mut g, model_config(cca));
    g.set_outputs(vec![model.logits()]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    fill_params(&mut session);
    apply_zero_init(&mut session, &model);

    let corpus = NeighbourCorpus::new(R, EMBED);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::disabled();
    let mut embedder = HashedBagEmbedder::bigram(EMBED, MASK.0);
    let prompt = [3u32, 7, 11];

    let tokens = {
        let mut denoiser = SessionDenoiser::new(&mut session, cca, VOCAB, MASK);
        let mut sampling = SamplingConfig::new(DocumentId(u64::MAX));
        sampling.steps = 8;
        let mut retrieval = RetrievalContext {
            index: &index,
            corpus: &corpus,
            guard: &guard,
            encoder: &mut embedder,
        };
        generate(
            &prompt,
            MASK,
            cca,
            &mut sampling,
            &mut retrieval,
            &mut denoiser,
        )
    };

    assert_eq!(tokens.len(), SEQ);
    assert_eq!(
        &tokens[..prompt.len()],
        &prompt,
        "the prompt prefix must survive"
    );
    assert!(
        tokens.iter().all(|&t| (t as usize) < VOCAB),
        "every position must be a vocab id"
    );
    let text = decode_token_ids(&tokens);
    assert!(
        !text.is_empty(),
        "decode of the result must be non-empty text"
    );
    let prompt_text = decode_token_ids(&prompt);
    assert_ne!(
        text, prompt_text,
        "the decoded generation must be more than the prompt"
    );
}

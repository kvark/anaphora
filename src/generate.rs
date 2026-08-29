//! Host-side generation: a session-backed [`Denoiser`] on top of [`sample`].
//!
//! [`sample`] is written against a trait so the trajectory is testable without
//! a GPU. This module is the production implementation of that trait: one
//! already-built [`Session`], one already-built retrieval index, tokens in
//! and tokens out. The graph is not rebuilt per turn — that is the whole
//! point of a playable loop.

use crate::chunk::RetrieverEncode;
use crate::config::CcaConfig;
use crate::retrieval::Neighbours;
use crate::retrieval::index::NeighbourIndex;
use crate::sample::{Denoiser, RetrievalContext, SamplingConfig, sample};
use crate::train::bind_inputs;
use crate::view::{MaskToken, NoisedView};
use meganeura::runtime::Session;

/// A [`Denoiser`] that runs one forward pass of a built Meganeura session.
///
/// Requires a graph built with [`crate::model::NeighbourInput::Encoded`]: the
/// neighbour tokens are a U32 input, and the encoder lives in the graph. The
/// cached-KV sampling arrangement is a later optimisation (see
/// `docs/roadmap.md`); it is not what a first playable loop needs.
pub struct SessionDenoiser<'s> {
    session: &'s mut Session,
    cca: CcaConfig,
    vocab_size: usize,
    mask_token: MaskToken,
    t_col: Vec<f32>,
    retrieval_mask: Vec<f32>,
    neighbour_tokens: Vec<u32>,
}

impl<'s> SessionDenoiser<'s> {
    /// Bind this denoiser to `session` for a model of shape `cca`.
    pub fn new(
        session: &'s mut Session,
        cca: CcaConfig,
        vocab_size: usize,
        mask_token: MaskToken,
    ) -> Self {
        let n = cca.seq_len();
        let kv_rows = cca.num_chunks() * cca.neighbour_kv_rows();
        Self {
            session,
            cca,
            vocab_size,
            mask_token,
            t_col: vec![0.0; n],
            retrieval_mask: vec![0.0; n],
            neighbour_tokens: vec![mask_token.0; kv_rows],
        }
    }
}

impl Denoiser for SessionDenoiser<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn logits(&mut self, view: &NoisedView, neighbours: Option<&Neighbours>) -> Vec<f32> {
        let n = self.cca.seq_len();
        debug_assert_eq!(view.len(), n, "view length must match the compiled graph");
        self.t_col.fill(view.noise_level().get());
        fill_neighbour_inputs(
            self.cca,
            self.mask_token.0,
            neighbours,
            &mut self.retrieval_mask,
            &mut self.neighbour_tokens,
        );
        bind_inputs(
            self.session,
            view,
            &self.t_col,
            &self.retrieval_mask,
            &self.neighbour_tokens,
        );
        self.session.step();
        self.session.wait();
        self.session.read_output(n * self.vocab_size)
    }
}

/// Pack neighbour tokens and the per-row retrieval mask the CCA gate reads.
///
/// A closed hard gate (`neighbours == None`) must produce a zero mask, not a
/// stale block from the previous step — that is the contract on
/// [`Denoiser::logits`].
fn fill_neighbour_inputs(
    cca: CcaConfig,
    mask_token: u32,
    neighbours: Option<&Neighbours>,
    retrieval_mask: &mut Vec<f32>,
    neighbour_tokens: &mut Vec<u32>,
) {
    let n = cca.seq_len();
    let m = cca.chunk_size();
    let rows = cca.neighbour_kv_rows();
    retrieval_mask.clear();
    retrieval_mask.resize(n, 0.0);
    neighbour_tokens.clear();
    neighbour_tokens.resize(cca.num_chunks() * rows, mask_token);

    let Some(found) = neighbours else {
        return;
    };
    for chunk in 0..cca.num_chunks() {
        if !found.chunk_has_neighbours(chunk) {
            continue;
        }
        for row in chunk * m..(chunk + 1) * m {
            retrieval_mask[row] = 1.0;
        }
        if let Some(tokens) = found.chunk_tokens(chunk) {
            let start = chunk * rows;
            neighbour_tokens[start..start + rows].copy_from_slice(tokens);
        }
    }
}

/// Run one denoising trajectory. Tokens in, tokens out.
///
/// This is the only generate function. REPL I/O, argparse, and tokenizer
/// loading live outside it so tests hit this path directly. The prompt prefix
/// of the returned sequence is the prompt that was passed in; remaining
/// positions have been revealed by [`sample`].
pub fn generate<I, E, D>(
    prompt: &[u32],
    mask_token: MaskToken,
    cfg: CcaConfig,
    sampling: &mut SamplingConfig,
    retrieval: &mut RetrievalContext<'_, I, E>,
    denoiser: &mut D,
) -> Vec<u32>
where
    I: NeighbourIndex,
    E: RetrieverEncode,
    D: Denoiser,
{
    let (view, _trace) = sample(prompt, mask_token, cfg, sampling, retrieval, denoiser);
    view.tokens().to_vec()
}

/// Decode token ids as space-separated decimals.
///
/// Tokenizer-free, so the generate-path tests (and anything else that does
/// not want to link `tokenizers`) can assert the result is non-empty text.
/// The interactive binary uses the LLaDA tokenizer for real text.
pub fn decode_token_ids(ids: &[u32]) -> String {
    let mut out = String::new();
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&id.to_string());
    }
    out
}

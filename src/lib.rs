//! **Anaphora** — retrieval-augmented masked-diffusion language modeling.
//!
//! Chunked cross-attention (CCA), as RETRO defined it, under a masked
//! diffusion LM instead of an autoregressive one. The shape is a retrofit:
//! freeze a pretrained diffusion backbone (LLaDA-8B, Dream-7B) and train only
//! the neighbour encoder, the CCA blocks, and the gates.
//!
//! # What changes when the backbone is a diffusion model
//!
//! RETRO could retrieve once, before generation, because an autoregressive
//! model's context only grows. A diffusion model's context *sharpens*: every
//! denoising step produces a cleaner view of the whole sequence. So retrieval
//! moves inside the denoising loop, and the model can re-query on a
//! progressively better sketch — early steps on a rough semantic gist, later
//! steps on something close to the final text. That is the capability
//! autoregressive RETRO does not have.
//!
//! It also removes RETRO's chunk offset. `C_u+` existed so that chunk `u`
//! would not attend to neighbours retrieved using chunk `u`'s own tokens;
//! diffusion has no ordering to exploit for that. The principle survives in a
//! nastier form, and it is the thing this crate is most careful about.
//!
//! # The failure this crate is built to prevent
//!
//! > The retriever may only see what the denoiser sees.
//!
//! Build retrieval queries from `x_0` during training and the neighbours
//! correlate with exactly the tokens that were masked. The loss collapses
//! into copy-from-neighbour and the experiment measures nothing. **The
//! failure is silent — perplexity improves.**
//!
//! Three mechanisms make it hard to write by accident:
//!
//! 1. [`view::CleanSequence`] has no accessor that reaches the retrieval
//!    path. [`chunk::chunk_queries`] accepts only a [`view::NoisedView`].
//! 2. Every [`view::NoisedView`] carries a [`view::ViewId`], and everything
//!    derived from it carries that id forward, so retrieving against a
//!    *different, cleaner* view of the same sequence is caught too.
//! 3. [`retrieval::leakage`] keeps the training document out of its own
//!    results by provenance at query time, and by an offline n-gram audit at
//!    corpus preparation time — not by an inline filter against `x_t`, which
//!    is blindest exactly where the leak lives.
//!
//! # Layout
//!
//! | module | design sketch |
//! |---|---|
//! | [`view`], [`chunk`] | §1 query construction |
//! | [`model::cca`], [`model::gate`] | §2 the CCA block |
//! | [`schedule`] | §3 the hard gate |
//! | [`sample`] | §4 inference inside the denoising loop |
//! | [`mod@generate`] | session-backed sampling for a playable loop |
//!
//! # Status
//!
//! Early. The neighbour encoder's treatment of `[MASK]`-bearing queries is
//! the piece the design sketch marks unsolved, and it is still unsolved here
//! — [`chunk::RetrieverEncode`] is the seam where the three candidate answers
//! plug in. See `docs/roadmap.md`.

pub mod checkpoint;
pub mod chunk;
pub mod config;
pub mod corpus;
pub mod eval;
pub mod generate;
pub mod loss;
pub mod model;
pub mod retrieval;
pub mod sample;
pub mod schedule;
pub mod shard;
pub mod train;
pub mod view;

pub use checkpoint::{Checkpoint, CheckpointError, CheckpointMeta};
pub use chunk::{ChunkAdmission, ChunkQueries, ChunkedView, RetrieverEncode, chunk_queries};
pub use config::{CcaConfig, ConfigError};
pub use corpus::{ChunkEmbedder, Document, HashedBagEmbedder, TrainingSequence};
pub use eval::{EvalReport, Evaluator, NeighbourCondition};
pub use generate::{SessionDenoiser, decode_token_ids, generate};
pub use loss::{LabelStats, MaskedDiffusionLoss};
pub use model::{CcaInsertion, CcaModel, ModelConfig, NeighbourInput};
pub use retrieval::{Neighbours, retrieve};
pub use schedule::{NoiseLevel, Phase, RefreshSchedule, retrieve_now, trajectory};
pub use shard::{CorpusShard, Split, split_of};
pub use train::{RetrievalSources, StepReport, Trainer, TrainingConfig, seed_parameters};
pub use view::{CleanSequence, MaskToken, NoisedView, SequenceId, ViewId};

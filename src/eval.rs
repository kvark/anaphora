//! The evaluation protocol.
//!
//! The structural guards in [`crate::view`] and
//! [`crate::retrieval::leakage`] prevent the leaks we know how to name. They
//! cannot prove absence. This module is the empirical check that a perplexity
//! improvement is retrieval being *used* rather than *copied*.
//!
//! # Conditions
//!
//! The same trained weights are evaluated against several neighbour blocks.
//! Everything else about the step is identical, so the differences between
//! them are attributable:
//!
//! * [`NeighbourCondition::Real`] — what the index actually returns.
//! * [`NeighbourCondition::Ablated`] — no neighbours, gate forced to zero.
//!   `Ablated − Real` is what the retrieval path is worth.
//! * [`NeighbourCondition::Random`] — real passages from the corpus, but the
//!   wrong ones. `Random − Real` is the headline diagnostic: a model that
//!   learned to *use* retrieval degrades gracefully when the neighbours stop
//!   being relevant, and one that learned to *copy* falls off a cliff.
//! * [`NeighbourCondition::Oracle`] — the sequence's own continuation, which
//!   is what a perfect retriever would find. An upper bound, and a check that
//!   the CCA path can carry information at all: if `Oracle` does not beat
//!   `Real`, nothing downstream is measuring retrieval quality.
//!
//! # Why the bands matter
//!
//! Every condition is reported per noise band as well as overall. Copying
//! concentrates at low `t`, where few positions are still masked and a
//! neighbour's continuation can supply them — which is exactly the band the
//! training gate closes and exactly what a single averaged number hides.

use crate::config::CcaConfig;
use crate::corpus::{ChunkEmbedder, TrainingSequence};
use crate::loss::MaskedDiffusionLoss;
use crate::retrieval::corpus::{NeighbourCorpus, NeighbourId};
use crate::retrieval::index::NeighbourIndex;
use crate::retrieval::retrieve;
use crate::schedule::{NoiseLevel, Phase};
use crate::train::{RetrievalSources, Rng, bind_inputs};
use crate::view::{CleanSequence, MaskToken, NoisedView};
use meganeura::runtime::Session;

/// Which neighbour block a condition feeds the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeighbourCondition {
    /// What the index returns for this sequence.
    Real,
    /// Corpus passages chosen at random rather than by relevance.
    Random,
    /// No neighbours; the retrieval mask is zero everywhere.
    Ablated,
    /// The sequence's own continuation — what a perfect retriever finds.
    Oracle,
}

impl NeighbourCondition {
    /// Every condition, in report order.
    pub const ALL: [Self; 4] = [Self::Real, Self::Random, Self::Ablated, Self::Oracle];

    /// A short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Random => "random",
            Self::Ablated => "ablated",
            Self::Oracle => "oracle",
        }
    }
}

/// Mean loss over one noise band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandLoss {
    /// Half-open `[low, high)` noise range.
    pub low: f32,
    /// Upper edge.
    pub high: f32,
    /// Mean loss over the steps that landed in this band.
    pub mean_loss: f32,
    /// Steps that landed here and scored something.
    pub steps: usize,
}

/// One condition's results.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionReport {
    /// Which condition.
    pub condition: NeighbourCondition,
    /// Mean loss across every scoring step.
    pub mean_loss: f32,
    /// The same, split by noise band.
    pub bands: Vec<BandLoss>,
}

impl ConditionReport {
    /// Mean loss below `t`, across the bands that fall entirely under it.
    pub fn mean_below(&self, t: f32) -> Option<f32> {
        let (sum, count) = self
            .bands
            .iter()
            .filter(|b| b.high <= t && b.steps > 0)
            .fold((0.0, 0usize), |(s, c), b| {
                (s + b.mean_loss * b.steps as f32, c + b.steps)
            });
        (count > 0).then(|| sum / count as f32)
    }
}

/// The full protocol run.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// One entry per condition evaluated.
    pub conditions: Vec<ConditionReport>,
}

impl EvalReport {
    /// The report for one condition.
    pub fn get(&self, condition: NeighbourCondition) -> Option<&ConditionReport> {
        self.conditions.iter().find(|c| c.condition == condition)
    }

    fn delta(&self, from: NeighbourCondition, to: NeighbourCondition) -> Option<f32> {
        Some(self.get(from)?.mean_loss - self.get(to)?.mean_loss)
    }

    /// What the retrieval path is worth: `Ablated − Real`.
    ///
    /// Positive means retrieval helps.
    pub fn retrieval_gain(&self) -> Option<f32> {
        self.delta(NeighbourCondition::Ablated, NeighbourCondition::Real)
    }

    /// The copy diagnostic: `Random − Real`.
    ///
    /// How much worse the model gets when its neighbours stop being relevant.
    /// Large in absolute terms is suspicious; large *relative to*
    /// [`Self::retrieval_gain`] is the real signal, which is what
    /// [`Self::copy_ratio`] reports.
    pub fn copy_gap(&self) -> Option<f32> {
        self.delta(NeighbourCondition::Random, NeighbourCondition::Real)
    }

    /// `copy_gap / retrieval_gain`.
    ///
    /// A model using retrieval well has both numbers of a similar size: the
    /// relevant neighbours help, and irrelevant ones simply stop helping,
    /// landing near the ablated baseline. A model that has learned to copy
    /// does *worse than ablated* on random neighbours, because it is
    /// transcribing text that has nothing to do with the target — so the gap
    /// runs well past the gain.
    ///
    /// `None` when the gain is too small to divide by, which is itself a
    /// result: nothing was learned to leak through.
    pub fn copy_ratio(&self) -> Option<f32> {
        let gain = self.retrieval_gain()?;
        let gap = self.copy_gap()?;
        (gain.abs() > 1e-6).then(|| gap / gain)
    }

    /// Whether random neighbours are *worse* than no neighbours at all.
    ///
    /// The sharpest single signature of copying. Retrieval that is merely
    /// unhelpful should degrade to the ablated baseline, because the gate can
    /// learn to shut. Retrieval that is actively harmful means the model is
    /// leaning on neighbour content it cannot evaluate.
    pub fn random_worse_than_ablated(&self) -> Option<bool> {
        let random = self.get(NeighbourCondition::Random)?.mean_loss;
        let ablated = self.get(NeighbourCondition::Ablated)?.mean_loss;
        Some(random > ablated)
    }

    /// Render the protocol's table.
    pub fn to_table(&self) -> String {
        let mut out = String::from("condition   mean     bands\n");
        for c in &self.conditions {
            out.push_str(&format!(
                "{:<10} {:>7.4}  ",
                c.condition.label(),
                c.mean_loss
            ));
            for b in &c.bands {
                if b.steps > 0 {
                    out.push_str(&format!("[{:.2},{:.2})={:.3} ", b.low, b.high, b.mean_loss));
                }
            }
            out.push('\n');
        }
        if let Some(gain) = self.retrieval_gain() {
            out.push_str(&format!("retrieval gain (ablated-real): {gain:+.4}\n"));
        }
        if let Some(gap) = self.copy_gap() {
            out.push_str(&format!("copy gap (random-real):        {gap:+.4}\n"));
        }
        if let Some(ratio) = self.copy_ratio() {
            out.push_str(&format!("copy ratio (gap/gain):         {ratio:+.2}\n"));
        }
        out
    }
}

/// Runs the protocol.
pub struct Evaluator {
    cca: CcaConfig,
    loss: MaskedDiffusionLoss,
    mask_token: MaskToken,
    bands: Vec<(f32, f32)>,
    rng: Rng,
    labels: Vec<f32>,
    t_col: Vec<f32>,
    retrieval_mask: Vec<f32>,
    neighbour_tokens: Vec<u32>,
}

impl Evaluator {
    /// Five equal bands over `[0, 1]`.
    pub fn default_bands() -> Vec<(f32, f32)> {
        (0..5)
            .map(|i| (i as f32 * 0.2, (i + 1) as f32 * 0.2))
            .collect()
    }

    /// An evaluator over `cca` shapes and a `vocab_size` vocabulary.
    pub fn new(cca: CcaConfig, vocab_size: usize, mask_token: MaskToken, seed: u64) -> Self {
        let n = cca.seq_len();
        Self {
            loss: MaskedDiffusionLoss::new(vocab_size),
            mask_token,
            bands: Self::default_bands(),
            rng: Rng::new(seed),
            labels: Vec::with_capacity(n * vocab_size),
            t_col: vec![0.0; n],
            retrieval_mask: vec![0.0; n],
            neighbour_tokens: vec![mask_token.0; cca.num_chunks() * cca.neighbour_kv_rows()],
            cca,
        }
    }

    /// Override the noise bands.
    pub fn with_bands(mut self, bands: Vec<(f32, f32)>) -> Self {
        self.bands = bands;
        self
    }

    /// Evaluate `conditions` over `sequences` at each of `levels`.
    ///
    /// **Clears the session's optimizer.** Evaluation runs the training graph,
    /// which computes gradients whether or not they are wanted; leaving an
    /// optimizer configured would have the evaluation quietly train the model
    /// it is measuring. Reconfigure before resuming training.
    pub fn run<I: NeighbourIndex, E: ChunkEmbedder>(
        &mut self,
        session: &mut Session,
        sequences: &[TrainingSequence],
        levels: &[NoiseLevel],
        conditions: &[NeighbourCondition],
        sources: &mut RetrievalSources<'_, I, E>,
    ) -> EvalReport {
        session.clear_optimizer();

        let mut reports = Vec::new();
        for &condition in conditions {
            let mut band_sums = vec![(0.0f64, 0usize); self.bands.len()];
            let mut total = 0.0f64;
            let mut count = 0usize;

            for seq in sequences {
                for &t in levels {
                    let Some(loss) = self.score_one(session, seq, t, condition, sources) else {
                        continue;
                    };
                    total += loss as f64;
                    count += 1;
                    if let Some(i) = self.band_of(t) {
                        band_sums[i].0 += loss as f64;
                        band_sums[i].1 += 1;
                    }
                }
            }

            reports.push(ConditionReport {
                condition,
                mean_loss: if count == 0 {
                    f32::NAN
                } else {
                    (total / count as f64) as f32
                },
                bands: self
                    .bands
                    .iter()
                    .zip(&band_sums)
                    .map(|(&(low, high), &(sum, steps))| BandLoss {
                        low,
                        high,
                        mean_loss: if steps == 0 {
                            f32::NAN
                        } else {
                            (sum / steps as f64) as f32
                        },
                        steps,
                    })
                    .collect(),
            });
        }

        EvalReport {
            conditions: reports,
        }
    }

    fn band_of(&self, t: NoiseLevel) -> Option<usize> {
        let v = t.get();
        self.bands
            .iter()
            .position(|&(low, high)| v >= low && (v < high || (high >= 1.0 && v <= 1.0)))
    }

    /// Mask deterministically from `t` so every condition sees the *same*
    /// corruption of the same sequence. Re-drawing per condition would put
    /// the masking noise into the differences the protocol is reading.
    fn deterministic_view(
        &self,
        seq: &TrainingSequence,
        t: NoiseLevel,
    ) -> (CleanSequence, NoisedView) {
        let clean = CleanSequence::new(seq.tokens.clone());
        let mut rng = Rng::new(seq.document.0 ^ ((t.get() * 1e6) as u64).wrapping_mul(0x9E37_79B9));
        let draws: Vec<bool> = (0..seq.tokens.len())
            .map(|i| i < seq.content_len && rng.next_f32() < t.get())
            .collect();
        let view = clean.mask_with(t, self.mask_token, |i| draws[i]);
        (clean, view)
    }

    fn score_one<I: NeighbourIndex, E: ChunkEmbedder>(
        &mut self,
        session: &mut Session,
        seq: &TrainingSequence,
        t: NoiseLevel,
        condition: NeighbourCondition,
        sources: &mut RetrievalSources<'_, I, E>,
    ) -> Option<f32> {
        let (clean, view) = self.deterministic_view(seq, t);
        let stats = self
            .loss
            .build_labels(&view, &clean, &mut self.labels)
            .expect("the view was masked from this clean sequence");
        if !stats.contributes() {
            return None;
        }

        let n = self.cca.seq_len();
        let m = self.cca.chunk_size();
        let rows = self.cca.neighbour_kv_rows();
        self.t_col.clear();
        self.t_col.resize(n, t.get());
        self.retrieval_mask.clear();
        self.retrieval_mask.resize(n, 0.0);
        self.neighbour_tokens.clear();
        self.neighbour_tokens
            .resize(self.cca.num_chunks() * rows, self.mask_token.0);

        if condition != NeighbourCondition::Ablated {
            self.fill_neighbours(&view, seq, condition, sources);
            for chunk in 0..self.cca.num_chunks() {
                let start = chunk * rows;
                let filled = self.neighbour_tokens[start..start + rows]
                    .iter()
                    .any(|&tok| tok != self.mask_token.0);
                if filled {
                    for row in chunk * m..(chunk + 1) * m {
                        self.retrieval_mask[row] = 1.0;
                    }
                }
            }
        }

        bind_inputs(
            session,
            &view,
            &self.t_col,
            &self.retrieval_mask,
            &self.neighbour_tokens,
            &self.labels,
        );
        session.step();
        session.wait();
        Some(session.read_loss())
    }

    fn fill_neighbours<I: NeighbourIndex, E: ChunkEmbedder>(
        &mut self,
        view: &NoisedView,
        seq: &TrainingSequence,
        condition: NeighbourCondition,
        sources: &mut RetrievalSources<'_, I, E>,
    ) {
        let (l, k, r, m) = (
            self.cca.num_chunks(),
            self.cca.neighbours_per_chunk(),
            self.cca.neighbour_len(),
            self.cca.chunk_size(),
        );
        let rows = self.cca.neighbour_kv_rows();

        match condition {
            NeighbourCondition::Ablated => {}
            NeighbourCondition::Real => {
                // Evaluation is inference: there is no held-out answer to
                // leak, so the inference band applies.
                let Ok(chunked) = crate::chunk::ChunkedView::new(view, self.cca) else {
                    return;
                };
                let Some(queries) = crate::chunk::chunk_queries(
                    chunked,
                    Phase::Inference,
                    crate::chunk::ChunkAdmission::permissive(),
                    sources.embedder,
                ) else {
                    return;
                };
                let found = retrieve(
                    &queries,
                    self.cca,
                    sources.index,
                    sources.corpus,
                    sources.guard,
                    seq.document,
                );
                for chunk in 0..l {
                    if let Some(tokens) = found.chunk_tokens(chunk) {
                        let start = chunk * rows;
                        self.neighbour_tokens[start..start + rows].copy_from_slice(tokens);
                    }
                }
            }
            NeighbourCondition::Random => {
                // Real passages, wrong ones. The point is that they are
                // well-formed text from the same corpus, so any degradation
                // is about relevance rather than about the input turning to
                // noise.
                let corpus_len = sources.corpus.len();
                if corpus_len == 0 {
                    return;
                }
                for chunk in 0..l {
                    for slot in 0..k {
                        let pick = NeighbourId(self.rng.below(corpus_len) as u32);
                        if let Some(tokens) = sources.corpus.tokens(pick) {
                            let start = chunk * rows + slot * r;
                            self.neighbour_tokens[start..start + r].copy_from_slice(tokens);
                        }
                    }
                }
            }
            NeighbourCondition::Oracle => {
                // The chunk plus its true continuation, which is exactly what
                // a perfect retriever would return. Padded chunks near the
                // tail simply get less of it.
                for chunk in 0..l {
                    let from = chunk * m;
                    let available = seq.content_len.saturating_sub(from);
                    let take = available.min(r);
                    if take == 0 {
                        continue;
                    }
                    for slot in 0..k {
                        let start = chunk * rows + slot * r;
                        self.neighbour_tokens[start..start + take]
                            .copy_from_slice(&seq.tokens[from..from + take]);
                    }
                }
            }
        }
    }
}

/// Compute the n-gram overlap of each evaluation sequence with the corpus.
///
/// Protocol measurement 7. Retrieval papers leak here routinely: an
/// evaluation document that is also in the index makes retrieval look
/// spectacular for reasons that have nothing to do with the model. Report
/// metrics separately on the low-overlap subset.
pub fn eval_overlap(
    sequences: &[TrainingSequence],
    corpus: &NeighbourCorpus,
    order: usize,
) -> Vec<f32> {
    sequences
        .iter()
        .map(|seq| {
            let content = &seq.tokens[..seq.content_len];
            (0..corpus.len())
                .filter_map(|i| corpus.tokens(NeighbourId(i as u32)))
                .map(|tokens| crate::retrieval::leakage::ngram_overlap(tokens, content, order))
                .fold(0.0f32, f32::max)
        })
        .collect()
}

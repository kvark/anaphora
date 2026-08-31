//! Phase 1: train the retrofit on real Wikipedia and run the protocol.
//!
//! Reads the shards produced by `prepare_corpus`, pretrains a small
//! masked-diffusion backbone, freezes it, trains the retrieval path on top,
//! and reports the evaluation protocol.
//!
//! ```sh
//! cargo run --release --example phase1 -- --corpus corpus/
//! ```
//!
//! The defaults are the Phase 1 shape from `docs/v0-plan.md`: `d = 512` over
//! 8 layers, `n = 512`, `m = 64` so `l = 8`, `k = 2`, `r = 128`, CCA after
//! every second layer from layer 2. That is roughly 155M parameters, most of
//! it the vocabulary-sized embedding, and it wants a real GPU — the flags
//! below exist so the wiring can be smoke-tested on a software device first.
//!
//! # What to look at
//!
//! Not the perplexity. The protocol's table: what retrieval is worth
//! (`ablated - real`), and whether the model degrades gracefully when its
//! neighbours stop being relevant (`random - real`). A large gain with a
//! gap that runs well past it is the copying signature, and the whole reason
//! `examples/leak_calibration.rs` exists.

use anaphora::config::CcaConfig;
use anaphora::corpus::{Document, HashedBagEmbedder, build_corpus, training_sequences};
use anaphora::eval::{Evaluator, NeighbourCondition, eval_overlap};
use anaphora::loss::MaskedDiffusionLoss;
use anaphora::model::backbone::{Backbone, BackboneConfig, Freezing};
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::{LeakageGuard, NgramOverlapFilter};
use anaphora::schedule::NoiseLevel;
use anaphora::shard::CorpusShard;
use anaphora::train::{
    NoiseSampler, Optimizer, RetrievalSources, Rng, SparseLabels, Trainer, TrainingConfig,
    apply_zero_init, configure_optimizer, seed_parameters,
};
use anaphora::view::{CleanSequence, MaskToken};
use meganeura::Graph;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    corpus: PathBuf,
    seq_len: usize,
    chunk: usize,
    neighbours: usize,
    layers: usize,
    heads: u32,
    head_dim: u32,
    intermediate: usize,
    cca_every: usize,
    cca_from: usize,
    pretrain_steps: usize,
    train_steps: usize,
    lr: f32,
    embed_dim: usize,
    max_docs: Option<usize>,
    eval_windows: usize,
    eval_levels: usize,
    index_includes_train: bool,
    audit: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            corpus: PathBuf::from("corpus"),
            seq_len: 512,
            chunk: 64,
            neighbours: 2,
            layers: 8,
            heads: 8,
            head_dim: 64,
            intermediate: 1408,
            cca_every: 2,
            cca_from: 2,
            pretrain_steps: 20_000,
            train_steps: 10_000,
            lr: 3e-4,
            embed_dim: 256,
            max_docs: None,
            eval_windows: 256,
            eval_levels: 9,
            index_includes_train: false,
            audit: true,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "phase1 --corpus DIR\n\
         \n\
         Shape:\n  \
           --seq-len N --chunk N --neighbours N --layers N --heads N\n  \
           --head-dim N --intermediate N --embed-dim N\n  \
           --cca-every N --cca-from N\n\
         Schedule:\n  \
           --pretrain-steps N --train-steps N --lr F --max-docs N\n\
         Protocol:\n  \
           --eval-windows N         evaluation windows to score (default 256)\n  \
           --eval-levels N          noise levels per window     (default 9)\n  \
           --index-includes-train   put the training split in the index too\n  \
           --no-audit               skip the offline n-gram exclusion"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut v = || argv.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--corpus" => a.corpus = PathBuf::from(v()),
            "--seq-len" => a.seq_len = v().parse().unwrap_or_else(|_| usage()),
            "--chunk" => a.chunk = v().parse().unwrap_or_else(|_| usage()),
            "--neighbours" => a.neighbours = v().parse().unwrap_or_else(|_| usage()),
            "--layers" => a.layers = v().parse().unwrap_or_else(|_| usage()),
            "--heads" => a.heads = v().parse().unwrap_or_else(|_| usage()),
            "--head-dim" => a.head_dim = v().parse().unwrap_or_else(|_| usage()),
            "--intermediate" => a.intermediate = v().parse().unwrap_or_else(|_| usage()),
            "--cca-every" => a.cca_every = v().parse().unwrap_or_else(|_| usage()),
            "--cca-from" => a.cca_from = v().parse().unwrap_or_else(|_| usage()),
            "--embed-dim" => a.embed_dim = v().parse().unwrap_or_else(|_| usage()),
            "--pretrain-steps" => a.pretrain_steps = v().parse().unwrap_or_else(|_| usage()),
            "--train-steps" => a.train_steps = v().parse().unwrap_or_else(|_| usage()),
            "--lr" => a.lr = v().parse().unwrap_or_else(|_| usage()),
            "--max-docs" => a.max_docs = Some(v().parse().unwrap_or_else(|_| usage())),
            "--eval-windows" => a.eval_windows = v().parse().unwrap_or_else(|_| usage()),
            "--eval-levels" => a.eval_levels = v().parse().unwrap_or_else(|_| usage()),
            "--index-includes-train" => a.index_includes_train = true,
            "--no-audit" => a.audit = false,
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    a
}

fn truncate(mut docs: Vec<Document>, max: Option<usize>) -> Vec<Document> {
    if let Some(max) = max {
        docs.truncate(max);
    }
    docs
}

struct Pretrained {
    name: String,
    values: Vec<f32>,
}

fn pretrain(
    args: &Args,
    backbone_cfg: BackboneConfig,
    docs: &[Document],
    mask: MaskToken,
    pad: u32,
) -> Vec<Pretrained> {
    let mut g = Graph::new();
    let backbone = Backbone::with_freezing(&mut g, "backbone", backbone_cfg, Freezing::Trainable);
    let token_ids = g.input_u32("token_ids", &[args.seq_len]);
    let mut x = backbone.embed(&mut g, token_ids);
    for layer in 0..backbone_cfg.num_layers {
        x = backbone.layer(&mut g, x, layer);
    }
    let logits = backbone.head(&mut g, x);
    let labels = g.input("labels", &[args.seq_len, backbone_cfg.vocab_size]);
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);

    let t0 = Instant::now();
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    println!(
        "[pretrain] session built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    seed_parameters(&mut session, 0xBA5E, None);
    configure_optimizer(&mut session, Optimizer::adam(args.lr));

    let seqs = training_sequences(docs, args.seq_len, pad, args.chunk);
    println!("[pretrain] {} windows", seqs.len());
    let objective = MaskedDiffusionLoss::new(backbone_cfg.vocab_size);
    let mut writer = SparseLabels::new(args.seq_len, backbone_cfg.vocab_size);
    writer.zero(&mut session, "labels").expect("labels input");
    let mut rng = Rng::new(0xDA7A);
    let mut window = Vec::new();

    let t0 = Instant::now();
    for i in 0..args.pretrain_steps {
        let seq = &seqs[i % seqs.len()];
        let t = NoiseLevel::saturating(1.0 - rng.next_f32());
        let clean = CleanSequence::new(seq.tokens.clone());
        let draws: Vec<bool> = (0..seq.tokens.len())
            .map(|j| j < seq.content_len && rng.next_f32() < t.get())
            .collect();
        let view = clean.mask_with(t, mask, |j| draws[j]);
        let stats = writer
            .write(&mut session, "labels", objective, &view, &clean)
            .expect("masked from this sequence");
        if !stats.contributes() {
            continue;
        }
        session.set_input_u32("token_ids", view.tokens());
        session.step();
        session.wait();
        window.push(session.read_loss());
        if window.len() == 500 {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            println!(
                "[pretrain] step {i:>7}  loss {mean:.4}  {:.1} steps/s",
                (i + 1) as f32 / t0.elapsed().as_secs_f32()
            );
            window.clear();
        }
    }

    Backbone::param_names("backbone", backbone_cfg)
        .into_iter()
        .map(|name| {
            let len = session.param_size(&name).expect("declared");
            let mut values = vec![0.0f32; len];
            session.read_param(&name, &mut values);
            Pretrained { name, values }
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    let train = CorpusShard::read(args.corpus.join("train.shard"), None)?;
    let index_shard = CorpusShard::read(args.corpus.join("index.shard"), None)?;
    let eval_shard = CorpusShard::read(args.corpus.join("eval.shard"), None)?;
    let mask = MaskToken(train.mask_token);
    // The tokenizer's `[MASK]` is a real token; padding must be something
    // else or padded positions would read as masked ones.
    let pad = train.mask_token.wrapping_sub(1);
    let vocab = train.vocab_size as usize;
    println!(
        "corpus: train {} docs / {} tokens, index {} docs, eval {} docs, vocab {vocab}",
        train.documents.len(),
        train.total_tokens(),
        index_shard.documents.len(),
        eval_shard.documents.len()
    );

    let cca = CcaConfig::new(
        args.seq_len,
        args.chunk,
        args.neighbours,
        args.chunk * 2,
        args.heads as usize,
        args.heads as usize,
        args.head_dim as usize,
        args.cca_every,
        args.cca_from,
    )?;
    let backbone_cfg = BackboneConfig {
        vocab_size: vocab,
        num_layers: args.layers,
        num_heads: args.heads,
        num_kv_heads: args.heads,
        head_dim: args.head_dim,
        intermediate_size: args.intermediate,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    };

    if cca.cca_layers(args.layers).is_empty() {
        return Err(format!(
            "no CCA block would be inserted: --cca-from {} is past the last of {} layers. \
             A retrofit with no CCA blocks is just the frozen backbone.",
            args.cca_from, args.layers
        )
        .into());
    }
    println!(
        "cca: n={} m={} l={} k={} r={} d={}, blocks after layers {:?}",
        cca.seq_len(),
        cca.chunk_size(),
        cca.num_chunks(),
        cca.neighbours_per_chunk(),
        cca.neighbour_len(),
        cca.model_dim(),
        cca.cca_layers(args.layers)
    );

    let train_docs = truncate(train.documents, args.max_docs);
    let index_docs = truncate(index_shard.documents, args.max_docs);
    let eval_docs = truncate(eval_shard.documents, args.max_docs);

    let pretrained = pretrain(&args, backbone_cfg, &train_docs, mask, pad);

    // Build the retrievable corpus. Which documents go in is the protocol's
    // index-membership A/B: a disjoint index cannot leak and understates what
    // retrieval is worth; one that also holds the training split is the
    // realistic setting and is what the guards are for.
    let mut embedder = HashedBagEmbedder::bigram(args.embed_dim, mask.0);
    let mut corpus_docs = index_docs;
    if args.index_includes_train {
        corpus_docs.extend(train_docs.iter().cloned());
    }
    let t0 = Instant::now();
    let corpus = build_corpus(&corpus_docs, cca, &mut embedder);
    let index = ExactIndex::build(&corpus);
    println!(
        "index: {} passages from {} documents in {:.1}s",
        corpus.len(),
        corpus_docs.len(),
        t0.elapsed().as_secs_f32()
    );

    let mut guard = LeakageGuard::by_source_document();
    if args.audit {
        // Near-duplicates living under another document id are what
        // provenance alone misses, and running the audit offline keeps the
        // exclusion independent of which positions a step happened to mask.
        let t0 = Instant::now();
        let filter = NgramOverlapFilter::default();
        let mut excluded = 0;
        for doc in &train_docs {
            excluded += guard.audit(doc.id, &doc.tokens, &corpus, filter);
        }
        println!(
            "audit: {excluded} passage exclusions across {} documents in {:.1}s",
            train_docs.len(),
            t0.elapsed().as_secs_f32()
        );
    }

    // The retrofit.
    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca,
            backbone: backbone_cfg,
            encoder: NeighbourEncoderConfig {
                vocab_size: vocab,
                num_layers: 2,
                num_heads: args.heads,
                num_kv_heads: args.heads,
                head_dim: args.head_dim,
                intermediate_size: args.intermediate,
                scope: EncoderScope::PerNeighbour,
            },
            activation: GateActivation::Tanh,
            neighbours: NeighbourInput::Encoded,
        },
    );
    let labels = g.input("labels", &[args.seq_len, vocab]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);

    let t0 = Instant::now();
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    println!(
        "[retrofit] session built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let trainable = model.trainable_param_names();
    seed_parameters(&mut session, 0xF17, Some(&trainable));
    for weight in &pretrained {
        session.set_parameter(&weight.name, &weight.values);
    }
    apply_zero_init(&mut session, &model);
    configure_optimizer(&mut session, Optimizer::adam(args.lr));
    println!("[retrofit] {} trainable parameters", trainable.len());

    let seqs = training_sequences(&train_docs, args.seq_len, pad, args.chunk);
    let mut trainer = Trainer::new(
        TrainingConfig {
            noise: NoiseSampler::Uniform,
            ..TrainingConfig::new(cca, vocab, mask)
        },
        0xCA11B,
    );
    let mut sources = RetrievalSources {
        index: &index,
        corpus: &corpus,
        guard: &guard,
        embedder: &mut embedder,
    };

    let t0 = Instant::now();
    let mut window = Vec::new();
    for i in 0..args.train_steps {
        let seq = &seqs[i % seqs.len()];
        if let Some(report) = trainer.step(&mut session, &model, seq, &mut sources)? {
            window.push(report.loss);
        }
        if window.len() == 500 {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            println!(
                "[retrofit] step {i:>7}  loss {mean:.4}  {:.1} steps/s",
                (i + 1) as f32 / t0.elapsed().as_secs_f32()
            );
            window.clear();
        }
    }

    // The protocol.
    // The protocol costs `windows * levels * conditions` forward passes, so
    // the window count is a knob rather than "all of them". Scoring every
    // window of every held-out document is four times the work of the
    // training run it is measuring.
    let mut eval_seqs = training_sequences(&eval_docs, args.seq_len, pad, args.chunk);
    eval_seqs.truncate(args.eval_windows);
    let overlap = eval_overlap(&eval_seqs, &corpus, NgramOverlapFilter::default().order());
    let contaminated = overlap.iter().filter(|&&o| o > 0.1).count();
    println!(
        "eval: {} windows, {contaminated} with >10% n-gram overlap with the index",
        eval_seqs.len()
    );

    let mut evaluator = Evaluator::new(cca, vocab, mask, 99);
    let levels: Vec<NoiseLevel> = (1..=args.eval_levels)
        .map(|i| NoiseLevel::new(i as f32 / (args.eval_levels + 1) as f32).expect("in range"))
        .collect();
    println!(
        "protocol: {} windows x {} levels x {} conditions = {} forward passes",
        eval_seqs.len(),
        levels.len(),
        NeighbourCondition::ALL.len(),
        eval_seqs.len() * levels.len() * NeighbourCondition::ALL.len()
    );
    let report = evaluator.run(
        &mut session,
        &eval_seqs,
        &levels,
        &NeighbourCondition::ALL,
        &mut sources,
    );
    println!("\n{}", report.to_table());
    if report.random_worse_than_ablated() == Some(true) {
        println!(
            "WARNING: random neighbours score worse than no neighbours. That is the\n\
             copying signature -- check the by-band numbers and re-read\n\
             docs/v0-plan.md before believing the perplexity."
        );
    }
    Ok(())
}

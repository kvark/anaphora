//! Interactive generation: type a prompt, get a decoded trajectory back.
//!
//! ```sh
//! cargo run --release --features text --bin play -- --train
//! cargo run --release --features text --bin play -- --prompt "The cat sat"
//! cargo run --release --features text --bin play
//! ```
//!
//! Weights come from a Wikipedia-trained checkpoint when one is present.
//! `--train` (or a missing checkpoint) pretrains the backbone on
//! `train.shard`, trains the retrieval path, and writes the dump. The graph
//! is not rebuilt per turn.

use anaphora::checkpoint::{Checkpoint, CheckpointMeta};
use anaphora::config::CcaConfig;
use anaphora::corpus::{Document, HashedBagEmbedder, build_corpus, training_sequences};
use anaphora::generate::{SessionDenoiser, generate};
use anaphora::loss::MaskedDiffusionLoss;
use anaphora::model::backbone::{Backbone, BackboneConfig, Freezing};
use anaphora::model::encoder::{EncoderScope, NeighbourEncoderConfig};
use anaphora::model::gate::GateActivation;
use anaphora::model::{CcaModel, ModelConfig, NeighbourInput};
use anaphora::retrieval::corpus::DocumentId;
use anaphora::retrieval::index::ExactIndex;
use anaphora::retrieval::leakage::LeakageGuard;
use anaphora::sample::{RetrievalContext, SamplingConfig};
use anaphora::shard::CorpusShard;
use anaphora::train::{
    Optimizer, RetrievalSources, Rng, SparseLabels, Trainer, TrainingConfig, apply_zero_init,
    configure_optimizer, seed_parameters,
};
use anaphora::view::{CleanSequence, MaskToken};
use meganeura::Graph;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const CHUNK: usize = 16;
const HEADS: u32 = 4;
const HEAD_DIM: u32 = 32;
const LAYERS: usize = 6;
const INTERMEDIATE: usize = 256;
const ENCODER_LAYERS: usize = 2;
const EMBED_DIM: usize = 64;

type NamedTensors = Vec<(String, Vec<f32>)>;

struct Args {
    prompt: Option<String>,
    tokenizer: PathBuf,
    corpus: PathBuf,
    checkpoint: PathBuf,
    seq_len: usize,
    steps: usize,
    max_docs: usize,
    train_docs: usize,
    seed: u64,
    train: bool,
    pretrain_steps: usize,
    train_steps: usize,
    lr: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            prompt: None,
            tokenizer: PathBuf::from("corpus/tokenizer.json"),
            corpus: PathBuf::from("corpus"),
            checkpoint: PathBuf::from("runs/play.ckpt"),
            seq_len: 128,
            steps: 16,
            max_docs: 2000,
            train_docs: 20_000,
            seed: 0xBA5E,
            train: false,
            pretrain_steps: 40_000,
            train_steps: 1_000,
            lr: 3e-4,
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "play [--prompt TEXT] [--tokenizer FILE] [--corpus DIR] [--checkpoint FILE]\n\
         \n\
         --prompt TEXT        one-shot: generate and exit (also used when stdin is not a TTY)\n\
         --tokenizer FILE     LLaDA tokenizer.json          (default corpus/tokenizer.json)\n\
         --corpus DIR         shards for train + retrieval  (default corpus/)\n\
         --checkpoint FILE    parameter dump                (default runs/play.ckpt)\n\
         --seq-len N          canvas length, multiple of {CHUNK} (default 128)\n\
         --steps N            denoising steps               (default 16)\n\
         --max-docs N         index documents to load       (default 2000)\n\
         --train-docs N       training documents            (default 20000)\n\
         --seed N             parameter seed                (default 0xBA5E)\n\
         --train              (re)train on Wikipedia and write the checkpoint\n\
         --pretrain-steps N   backbone steps                (default 40000)\n\
         --train-steps N      retrieval-path steps          (default 1000)\n\
         --lr F               Adam learning rate            (default 3e-4)\n\
         \n\
         Interactive: type a line, get a generation. quit / exit / :q ends the loop.\n\
         Pin the GPU with MEGANEURA_DEVICE_ID (0x744c for the 7900 XT)."
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut v = || argv.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--prompt" => a.prompt = Some(v()),
            "--tokenizer" => a.tokenizer = PathBuf::from(v()),
            "--corpus" => a.corpus = PathBuf::from(v()),
            "--checkpoint" => a.checkpoint = PathBuf::from(v()),
            "--seq-len" => a.seq_len = v().parse().unwrap_or_else(|_| usage()),
            "--steps" => a.steps = v().parse().unwrap_or_else(|_| usage()),
            "--max-docs" => a.max_docs = v().parse().unwrap_or_else(|_| usage()),
            "--train-docs" => a.train_docs = v().parse().unwrap_or_else(|_| usage()),
            "--pretrain-steps" => a.pretrain_steps = v().parse().unwrap_or_else(|_| usage()),
            "--train-steps" => a.train_steps = v().parse().unwrap_or_else(|_| usage()),
            "--lr" => a.lr = v().parse().unwrap_or_else(|_| usage()),
            "--train" => a.train = true,
            "--seed" => {
                let raw = v();
                a.seed =
                    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
                        u64::from_str_radix(hex, 16).unwrap_or_else(|_| usage())
                    } else {
                        raw.parse().unwrap_or_else(|_| usage())
                    };
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    a
}

fn mask_token_id(tokenizer: &tokenizers::Tokenizer) -> Result<u32, String> {
    tokenizer
        .token_to_id("<|mdm_mask|>")
        .or_else(|| tokenizer.token_to_id("[MASK]"))
        .or_else(|| tokenizer.token_to_id("<mask>"))
        .ok_or_else(|| "tokenizer has no recognisable mask token".to_string())
}

fn encode_prompt(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    seq_len: usize,
) -> Result<Vec<u32>, String> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| format!("encoding prompt: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    // Leave at least one masked position so a generation is not the prompt
    // padded out to n.
    let cap = seq_len.saturating_sub(1).max(1);
    if ids.len() > cap {
        ids.truncate(cap);
    }
    Ok(ids)
}

fn decode_ids(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String, String> {
    tokenizer
        .decode(ids, false)
        .map_err(|e| format!("decoding: {e}"))
}

struct Shapes {
    cca: CcaConfig,
    backbone: BackboneConfig,
    encoder: NeighbourEncoderConfig,
    meta: CheckpointMeta,
}

fn shapes(seq_len: usize, vocab: usize) -> Result<Shapes, Box<dyn std::error::Error>> {
    if seq_len == 0 || !seq_len.is_multiple_of(CHUNK) {
        return Err(format!("--seq-len {seq_len} must be a positive multiple of {CHUNK}").into());
    }
    let cca = CcaConfig::new(
        seq_len,
        CHUNK,
        2,
        CHUNK * 2,
        HEADS as usize,
        HEADS as usize,
        HEAD_DIM as usize,
        2,
        1,
    )?;
    let backbone = BackboneConfig {
        vocab_size: vocab,
        num_layers: LAYERS,
        num_heads: HEADS,
        num_kv_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    };
    let encoder = NeighbourEncoderConfig {
        vocab_size: vocab,
        num_layers: ENCODER_LAYERS,
        num_heads: HEADS,
        num_kv_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        scope: EncoderScope::PerNeighbour,
    };
    let meta = CheckpointMeta {
        vocab_size: vocab as u32,
        seq_len: seq_len as u32,
        chunk: CHUNK as u32,
        num_layers: LAYERS as u32,
        num_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE as u32,
    };
    Ok(Shapes {
        cca,
        backbone,
        encoder,
        meta,
    })
}

fn load_docs(path: &Path, max: usize) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
    let mut shard = CorpusShard::read(path, None)?;
    shard.documents.truncate(max);
    Ok(shard.documents)
}

fn pretrain_backbone(
    args: &Args,
    backbone_cfg: BackboneConfig,
    docs: &[Document],
    mask: MaskToken,
    pad: u32,
) -> Result<NamedTensors, Box<dyn std::error::Error>> {
    let n = args.seq_len;
    let mut g = Graph::new();
    let backbone = Backbone::with_freezing(&mut g, "backbone", backbone_cfg, Freezing::Trainable);
    let token_ids = g.input_u32("token_ids", &[n]);
    let mut x = backbone.embed(&mut g, token_ids);
    for layer in 0..backbone_cfg.num_layers {
        x = backbone.layer(&mut g, x, layer);
    }
    let logits = backbone.head(&mut g, x);
    let labels = g.input("labels", &[n, backbone_cfg.vocab_size]);
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);

    let t0 = Instant::now();
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    eprintln!(
        "[pretrain] session built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    seed_parameters(&mut session, args.seed, None);
    configure_optimizer(&mut session, Optimizer::adam(args.lr));

    let seqs = training_sequences(docs, n, pad, CHUNK);
    eprintln!(
        "[pretrain] {} windows, {} steps",
        seqs.len(),
        args.pretrain_steps
    );
    if seqs.is_empty() {
        return Err("no training windows: lower --min-tokens or pass more --train-docs".into());
    }
    let objective = MaskedDiffusionLoss::new(backbone_cfg.vocab_size);
    let mut writer = SparseLabels::new(n, backbone_cfg.vocab_size);
    writer.zero(&mut session, "labels")?;
    let mut rng = Rng::new(0xDA7A);
    let mut window = Vec::new();
    let t0 = Instant::now();
    for i in 0..args.pretrain_steps {
        let seq = &seqs[i % seqs.len()];
        let t = anaphora::NoiseLevel::saturating(1.0 - rng.next_f32());
        let clean = CleanSequence::new(seq.tokens.clone());
        let draws: Vec<bool> = (0..seq.tokens.len())
            .map(|j| j < seq.content_len && rng.next_f32() < t.get())
            .collect();
        let view = clean.mask_with(t, mask, |j| draws[j]);
        let stats = writer.write(&mut session, "labels", objective, &view, &clean)?;
        if !stats.contributes() {
            continue;
        }
        session.set_input_u32("token_ids", view.tokens());
        session.step();
        session.wait();
        window.push(session.read_loss());
        if window.len() == 500 {
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            eprintln!(
                "[pretrain] step {i:>7}  loss {mean:.4}  {:.1} steps/s",
                (i + 1) as f32 / t0.elapsed().as_secs_f32()
            );
            window.clear();
        }
    }

    Ok(Backbone::param_names("backbone", backbone_cfg)
        .into_iter()
        .map(|name| {
            let len = session.param_size(&name).expect("declared");
            let mut values = vec![0.0f32; len];
            session.read_param(&name, &mut values);
            (name, values)
        })
        .collect())
}

fn train_retrofit(
    args: &Args,
    shape: &Shapes,
    docs: &[Document],
    index_docs: &[Document],
    pretrained: &NamedTensors,
    mask: MaskToken,
    pad: u32,
) -> Result<Checkpoint, Box<dyn std::error::Error>> {
    let mut embedder = HashedBagEmbedder::bigram(EMBED_DIM, mask.0);
    let corpus = build_corpus(index_docs, shape.cca, &mut embedder);
    let index = ExactIndex::build(&corpus);
    let guard = LeakageGuard::by_source_document();
    eprintln!(
        "[retrofit] index {} passages from {} documents",
        corpus.len(),
        index_docs.len()
    );

    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca: shape.cca,
            backbone: shape.backbone,
            encoder: shape.encoder,
            activation: GateActivation::Tanh,
            neighbours: NeighbourInput::Encoded,
        },
    );
    let labels = g.input("labels", &[args.seq_len, shape.backbone.vocab_size]);
    let loss = g.cross_entropy_loss(model.logits(), labels);
    g.set_outputs(vec![loss]);

    let t0 = Instant::now();
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    eprintln!(
        "[retrofit] session built in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let trainable = model.trainable_param_names();
    seed_parameters(&mut session, args.seed ^ 0xF17, Some(&trainable));
    for param in pretrained {
        session.set_parameter(&param.0, &param.1);
    }
    apply_zero_init(&mut session, &model);
    configure_optimizer(&mut session, Optimizer::adam(args.lr));

    let seqs = training_sequences(docs, args.seq_len, pad, CHUNK);
    let mut trainer = Trainer::new(
        TrainingConfig::new(shape.cca, shape.backbone.vocab_size, mask),
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
            eprintln!(
                "[retrofit] step {i:>7}  loss {mean:.4}  {:.1} steps/s",
                (i + 1) as f32 / t0.elapsed().as_secs_f32()
            );
            window.clear();
        }
    }
    Ok(Checkpoint::capture(&session, shape.meta))
}

fn train_checkpoint(
    args: &Args,
    shape: &Shapes,
    mask: MaskToken,
) -> Result<Checkpoint, Box<dyn std::error::Error>> {
    let pad = mask.0.wrapping_sub(1);
    let train_path = args.corpus.join("train.shard");
    let index_path = args.corpus.join("index.shard");
    let train_docs = load_docs(&train_path, args.train_docs)?;
    let index_docs = load_docs(&index_path, args.max_docs)?;
    eprintln!(
        "training on {} docs / retrieving from {} docs, vocab {}",
        train_docs.len(),
        index_docs.len(),
        shape.backbone.vocab_size
    );
    let pretrained = pretrain_backbone(args, shape.backbone, &train_docs, mask, pad)?;
    train_retrofit(
        args,
        shape,
        &train_docs,
        &index_docs,
        &pretrained,
        mask,
        pad,
    )
}

struct Engine {
    session: meganeura::runtime::Session,
    cca: CcaConfig,
    mask: MaskToken,
    vocab: usize,
    steps: usize,
    tokenizer: tokenizers::Tokenizer,
    index: ExactIndex,
    corpus: anaphora::retrieval::corpus::NeighbourCorpus,
    guard: LeakageGuard,
    embedder: HashedBagEmbedder,
}

impl Engine {
    fn generate_tokens(&mut self, prompt: &[u32]) -> Vec<u32> {
        let mut denoiser = SessionDenoiser::new(&mut self.session, self.cca, self.vocab, self.mask);
        let mut sampling = SamplingConfig::new(DocumentId(u64::MAX));
        sampling.steps = self.steps;
        let mut retrieval = RetrievalContext {
            index: &self.index,
            corpus: &self.corpus,
            guard: &self.guard,
            encoder: &mut self.embedder,
        };
        generate(
            prompt,
            self.mask,
            self.cca,
            &mut sampling,
            &mut retrieval,
            &mut denoiser,
        )
    }

    fn generate_text(&mut self, prompt_text: &str) -> Result<String, String> {
        let prompt = encode_prompt(&self.tokenizer, prompt_text, self.cca.seq_len())?;
        let tokens = self.generate_tokens(&prompt);
        decode_ids(&self.tokenizer, &tokens)
    }
}

fn load_index(
    dir: &Path,
    cca: CcaConfig,
    mask: MaskToken,
    max_docs: usize,
) -> (
    anaphora::retrieval::corpus::NeighbourCorpus,
    ExactIndex,
    HashedBagEmbedder,
) {
    let mut embedder = HashedBagEmbedder::bigram(EMBED_DIM, mask.0);
    let shard_path = dir.join("index.shard");
    if !shard_path.exists() {
        let corpus =
            anaphora::retrieval::corpus::NeighbourCorpus::new(cca.neighbour_len(), EMBED_DIM);
        let index = ExactIndex::build(&corpus);
        return (corpus, index, embedder);
    }
    match CorpusShard::read(&shard_path, None) {
        Ok(mut shard) => {
            shard.documents.truncate(max_docs);
            let corpus = build_corpus(&shard.documents, cca, &mut embedder);
            let index = ExactIndex::build(&corpus);
            (corpus, index, embedder)
        }
        Err(e) => {
            eprintln!(
                "warning: could not read {}: {e}; retrieval is empty",
                shard_path.display()
            );
            let corpus =
                anaphora::retrieval::corpus::NeighbourCorpus::new(cca.neighbour_len(), EMBED_DIM);
            let index = ExactIndex::build(&corpus);
            (corpus, index, embedder)
        }
    }
}

fn build_engine(args: &Args) -> Result<Engine, Box<dyn std::error::Error>> {
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("loading {}: {e}", args.tokenizer.display()))?;
    let vocab = tokenizer.get_vocab_size(true);
    let mask = MaskToken(mask_token_id(&tokenizer)?);
    if args.steps == 0 {
        return Err("--steps must be positive".into());
    }
    let shape = shapes(args.seq_len, vocab)?;

    let ckpt = if !args.train && args.checkpoint.exists() {
        eprintln!("loading {}", args.checkpoint.display());
        Checkpoint::load(&args.checkpoint)?
    } else {
        if !args.train && !args.checkpoint.exists() {
            eprintln!(
                "no checkpoint at {}; training on Simple English Wikipedia",
                args.checkpoint.display()
            );
        }
        let ckpt = train_checkpoint(args, &shape, mask)?;
        ckpt.save(&args.checkpoint)?;
        eprintln!("wrote {}", args.checkpoint.display());
        ckpt
    };

    let mut g = Graph::new();
    let model = CcaModel::build(
        &mut g,
        ModelConfig {
            cca: shape.cca,
            backbone: shape.backbone,
            encoder: shape.encoder,
            activation: GateActivation::Tanh,
            neighbours: NeighbourInput::Encoded,
        },
    );
    g.set_outputs(vec![model.logits()]);
    eprintln!(
        "building session (n={}, vocab={vocab}, d={})…",
        shape.cca.seq_len(),
        shape.cca.model_dim()
    );
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    ckpt.apply(&mut session, shape.meta)?;

    let (corpus, index, embedder) = load_index(&args.corpus, shape.cca, mask, args.max_docs);
    eprintln!(
        "index: {} passages; denoising steps {}",
        corpus.len(),
        args.steps
    );

    Ok(Engine {
        session,
        cca: shape.cca,
        mask,
        vocab,
        steps: args.steps,
        tokenizer,
        index,
        corpus,
        guard: LeakageGuard::disabled(),
        embedder,
    })
}

fn is_quit(line: &str) -> bool {
    matches!(line, "quit" | "exit" | ":q" | ":quit")
}

fn one_shot(engine: &mut Engine, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    let text = engine.generate_text(prompt)?;
    println!("{text}");
    Ok(())
}

fn repl(engine: &mut Engine) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("anaphora play — type a prompt, or quit / exit / :q");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("> ");
        let _ = io::stderr().flush();
        let Some(line) = lines.next() else {
            break;
        };
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_quit(trimmed) {
            break;
        }
        match engine.generate_text(trimmed) {
            Ok(text) => {
                writeln!(stdout, "{text}")?;
                stdout.flush()?;
            }
            Err(e) => eprintln!("error: {e}"),
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let mut engine = build_engine(&args)?;

    if let Some(prompt) = args.prompt.as_deref() {
        return one_shot(&mut engine, prompt);
    }
    if !io::stdin().is_terminal() {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        let prompt = buf.trim_end();
        if prompt.is_empty() {
            return Err("stdin was empty; pass --prompt TEXT or type a line".into());
        }
        return one_shot(&mut engine, prompt);
    }
    repl(&mut engine)
}

//! Turn a Wikipedia parquet dump into Anaphora corpus shards.
//!
//! Downloading is deliberately not this tool's job — it is one `curl` and a
//! job for a shell, not a reason to link an HTTP client:
//!
//! ```sh
//! curl -L -o simple.parquet \
//!   'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet'
//! curl -L -o tokenizer.json \
//!   'https://huggingface.co/GSAI-ML/LLaDA-8B-Base/resolve/main/tokenizer.json'
//! ```
//!
//! Then:
//!
//! ```sh
//! cargo run --release --features wikipedia --bin prepare_corpus -- \
//!   --parquet simple.parquet --tokenizer tokenizer.json --out corpus/
//! ```
//!
//! Produces `train.shard`, `index.shard`, and `eval.shard`. The split is by
//! article id, so it is identical on every machine and every rerun — which is
//! what lets a leakage audit be reproduced rather than merely repeated.

use anaphora::corpus::Document;
use anaphora::retrieval::corpus::DocumentId;
use anaphora::shard::{CorpusShard, Split, split_of};
use arrow_array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::path::PathBuf;

struct Args {
    parquet: PathBuf,
    tokenizer: PathBuf,
    out: PathBuf,
    train: u32,
    index: u32,
    eval: u32,
    limit: Option<usize>,
    min_tokens: usize,
}

fn usage() -> ! {
    eprintln!(
        "prepare_corpus --parquet FILE --tokenizer tokenizer.json --out DIR\n\
         \n\
         Options:\n  \
           --train N        split weight for the training shard  (default 8)\n  \
           --index N        split weight for the index shard     (default 1)\n  \
           --eval N         split weight for the eval shard      (default 1)\n  \
           --limit N        stop after N articles (for a smoke run)\n  \
           --min-tokens N   drop articles shorter than this      (default 128)"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut parquet = None;
    let mut tokenizer = None;
    let mut out = None;
    let (mut train, mut index, mut eval) = (8u32, 1u32, 1u32);
    let mut limit = None;
    let mut min_tokens = 128usize;

    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "--parquet" => parquet = Some(PathBuf::from(value())),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value())),
            "--out" => out = Some(PathBuf::from(value())),
            "--train" => train = value().parse().unwrap_or_else(|_| usage()),
            "--index" => index = value().parse().unwrap_or_else(|_| usage()),
            "--eval" => eval = value().parse().unwrap_or_else(|_| usage()),
            "--limit" => limit = Some(value().parse().unwrap_or_else(|_| usage())),
            "--min-tokens" => min_tokens = value().parse().unwrap_or_else(|_| usage()),
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }

    Args {
        parquet: parquet.unwrap_or_else(|| usage()),
        tokenizer: tokenizer.unwrap_or_else(|| usage()),
        out: out.unwrap_or_else(|| usage()),
        train,
        index,
        eval,
        limit,
        min_tokens,
    }
}

/// Wikipedia ships article ids as strings. They are numeric in practice, but
/// a non-numeric one must still get a stable id rather than abort the run or,
/// worse, collide with a real article.
fn document_id(raw: &str) -> DocumentId {
    if let Ok(n) = raw.parse::<u64>() {
        return DocumentId(n);
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in raw.bytes() {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // Set the top bit so a hashed id can never collide with a parsed one,
    // which are small and sequential.
    DocumentId(h | (1 << 63))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    std::fs::create_dir_all(&args.out)?;

    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| format!("loading {}: {e}", args.tokenizer.display()))?;
    let vocab_size = tokenizer.get_vocab_size(true) as u32;
    let mask_token = tokenizer
        .token_to_id("<|mdm_mask|>")
        .or_else(|| tokenizer.token_to_id("[MASK]"))
        .or_else(|| tokenizer.token_to_id("<mask>"))
        .ok_or("tokenizer has no recognisable mask token")?;
    println!("tokenizer: vocab {vocab_size}, mask token {mask_token}");

    let file = std::fs::File::open(&args.parquet)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    let mut shards: HashMap<Split, Vec<Document>> = HashMap::new();
    let (mut seen, mut kept, mut short) = (0usize, 0usize, 0usize);

    'outer: for batch in reader {
        let batch = batch?;
        let ids = batch
            .column_by_name("id")
            .ok_or("parquet has no `id` column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("`id` is not a string column")?;
        let texts = batch
            .column_by_name("text")
            .ok_or("parquet has no `text` column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("`text` is not a string column")?;

        for row in 0..batch.num_rows() {
            if ids.is_null(row) || texts.is_null(row) {
                continue;
            }
            seen += 1;
            if let Some(limit) = args.limit
                && seen > limit
            {
                break 'outer;
            }

            let encoding = tokenizer
                .encode(texts.value(row), false)
                .map_err(|e| format!("tokenizing article {}: {e}", ids.value(row)))?;
            let tokens = encoding.get_ids().to_vec();
            // A document shorter than a training window contributes one mostly
            // padded sequence, which costs a full forward pass to score a
            // handful of positions.
            if tokens.len() < args.min_tokens {
                short += 1;
                continue;
            }
            kept += 1;

            let id = document_id(ids.value(row));
            let split = split_of(id, args.train, args.index, args.eval);
            shards
                .entry(split)
                .or_default()
                .push(Document { id, tokens });
        }

        if seen % 20_000 < batch.num_rows() {
            println!("  {seen} articles read, {kept} kept");
        }
    }

    println!("read {seen} articles: {kept} kept, {short} below --min-tokens");

    for (split, name) in [
        (Split::Train, "train"),
        (Split::Index, "index"),
        (Split::Eval, "eval"),
    ] {
        let documents = shards.remove(&split).unwrap_or_default();
        let shard = CorpusShard {
            vocab_size,
            mask_token,
            documents,
        };
        let path = args.out.join(format!("{name}.shard"));
        shard.write(&path)?;
        println!(
            "{}: {} documents, {} tokens",
            path.display(),
            shard.documents.len(),
            shard.total_tokens()
        );
    }

    Ok(())
}

//! The corpus shard format and the split assignment.

use anaphora::corpus::Document;
use anaphora::retrieval::corpus::DocumentId;
use anaphora::shard::{CorpusShard, ShardError, Split, split_of};
use std::collections::HashMap;

fn scratch(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("anaphora-shard-test-{name}-{}", std::process::id()));
    p
}

fn shard() -> CorpusShard {
    CorpusShard {
        vocab_size: 126_464,
        mask_token: 126_336,
        documents: vec![
            Document {
                id: DocumentId(7),
                tokens: vec![1, 2, 3, 4, 5],
            },
            Document {
                id: DocumentId(1_000_000),
                tokens: (0..300).collect(),
            },
            Document {
                id: DocumentId(9),
                tokens: Vec::new(),
            },
        ],
    }
}

#[test]
fn a_shard_round_trips() {
    let path = scratch("roundtrip");
    let original = shard();
    original.write(&path).expect("write");
    let read = CorpusShard::read(&path, None).expect("read");
    assert_eq!(read, original);
    assert_eq!(read.total_tokens(), 305);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_shard_the_model_cannot_represent_is_rejected() {
    // Token ids past the embedding table would index out of range.
    let path = scratch("vocab");
    shard().write(&path).expect("write");
    assert!(matches!(
        CorpusShard::read(&path, Some(32_000)),
        Err(ShardError::VocabTooLarge {
            shard: 126_464,
            model: 32_000
        })
    ));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_model_wider_than_its_tokenizer_is_accepted() {
    // LLaDA-8B reports 126,464 embedding rows against a tokenizer of about
    // 126,346 tokens, the rest being padding for alignment. Demanding
    // equality would reject a shard that is perfectly usable.
    let path = scratch("wider");
    let mut s = shard();
    s.vocab_size = 126_346;
    s.write(&path).expect("write");
    assert!(CorpusShard::read(&path, Some(126_464)).is_ok());
    assert!(CorpusShard::read(&path, Some(126_346)).is_ok());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_foreign_file_is_not_mistaken_for_a_shard() {
    let path = scratch("foreign");
    std::fs::write(&path, b"this is not a shard, it is a text file").expect("write");
    assert!(matches!(
        CorpusShard::read(&path, None),
        Err(ShardError::NotAShard)
    ));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_truncated_shard_fails_instead_of_allocating() {
    // A corrupt length field must not turn into a multi-gigabyte allocation
    // before the read fails.
    let path = scratch("truncated");
    shard().write(&path).expect("write");
    let mut bytes = std::fs::read(&path).expect("read back");
    bytes.truncate(40);
    std::fs::write(&path, &bytes).expect("rewrite");
    assert!(CorpusShard::read(&path, None).is_err());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn splits_are_deterministic_and_roughly_proportional() {
    let assign = |id: u64| split_of(DocumentId(id), 8, 1, 1);
    // Same article, same split, on every machine and every rerun — which is
    // what makes a leakage audit reproducible.
    for id in 0..100u64 {
        assert_eq!(assign(id), assign(id));
    }

    let mut counts: HashMap<Split, usize> = HashMap::new();
    for id in 0..10_000u64 {
        *counts.entry(assign(id)).or_default() += 1;
    }
    let train = counts[&Split::Train];
    let index = counts[&Split::Index];
    let eval = counts[&Split::Eval];
    assert_eq!(train + index + eval, 10_000);
    // 80/10/10, give or take hashing noise.
    assert!((7_600..8_400).contains(&train), "train was {train}");
    assert!((800..1_200).contains(&index), "index was {index}");
    assert!((800..1_200).contains(&eval), "eval was {eval}");
}

#[test]
fn splits_do_not_inherit_the_id_numbering() {
    // Wikipedia allocates article ids chronologically, so `id % 3` would
    // correlate the splits with article age. Consecutive ids must scatter.
    let runs = (0..60u64)
        .map(|id| split_of(DocumentId(id), 1, 1, 1))
        .collect::<Vec<_>>();
    let alternating = runs.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        alternating > 20,
        "consecutive ids landed in blocks, not scattered: {alternating} changes"
    );
}

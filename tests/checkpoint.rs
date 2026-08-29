//! The play checkpoint: round-trip of named tensors, and a shape mismatch.

use anaphora::{Checkpoint, CheckpointError, CheckpointMeta};
use std::io::Cursor;

fn meta() -> CheckpointMeta {
    CheckpointMeta {
        vocab_size: 48,
        seq_len: 16,
        chunk: 4,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        intermediate_size: 32,
    }
}

fn sample() -> Checkpoint {
    Checkpoint {
        meta: meta(),
        params: vec![
            ("backbone.embed".into(), vec![0.25, -0.5, 1.0, 0.0]),
            ("cca.1.gate.w2".into(), vec![0.0, 0.0]),
        ],
    }
}

#[test]
fn roundtrip_preserves_names_and_values() {
    let ckpt = sample();
    let mut buf = Vec::new();
    ckpt.write_to(&mut buf).expect("write");
    let loaded = Checkpoint::read_from(&mut Cursor::new(&buf)).expect("read");
    assert_eq!(loaded.meta, ckpt.meta);
    assert_eq!(loaded.params, ckpt.params);
}

#[test]
fn check_shape_rejects_a_mismatched_seq_len() {
    let ckpt = sample();
    let mut other = meta();
    other.seq_len = 32;
    let err = ckpt.check_shape(other).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("n=16"), "{text}");
    assert!(text.contains("n=32"), "{text}");
}

#[test]
fn rejects_the_wrong_magic() {
    let mut buf = b"NOTCKP!!".to_vec();
    buf.extend_from_slice(&[0u8; 16]);
    let err = Checkpoint::read_from(&mut Cursor::new(&buf)).unwrap_err();
    assert!(matches!(err, CheckpointError::NotACheckpoint));
}

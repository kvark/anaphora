//! The tokenized corpus interchange format.
//!
//! Ingest is a one-time job that needs a parquet reader and a tokenizer, both
//! of which are large dependencies — `arrow` for one, a C regex library for
//! the other. Training needs neither: it works on `u32` tokens and document
//! ids. A shard is the seam between them, so the heavy dependencies stay in
//! `prepare_corpus` and nothing that links this library pays for them.
//!
//! The format is deliberately dull: a header, then length-prefixed documents,
//! little-endian throughout.
//!
//! ```text
//! magic       8 bytes   b"ANAPHRA1"
//! version     u32       1
//! vocab_size  u32       the tokenizer's vocabulary
//! mask_token  u32       the [MASK] id, so a shard is self-describing
//! doc_count   u64
//! per document:
//!   id        u64       the source article id
//!   len       u32       token count
//!   tokens    u32 * len
//! ```
//!
//! `vocab_size` and `mask_token` travel with the data because a shard read
//! back against the wrong tokenizer produces token ids that are silently
//! valid and completely wrong. The reader checks them.

use crate::corpus::Document;
use crate::retrieval::corpus::DocumentId;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"ANAPHRA1";
const VERSION: u32 = 1;

/// Why a shard could not be read or written.
#[derive(Debug)]
pub enum ShardError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The file does not begin with the shard magic.
    NotAShard,
    /// The file's format version is not understood.
    Version(u32),
    /// The shard holds token ids the model cannot represent.
    VocabTooLarge {
        /// The shard's vocabulary bound.
        shard: u32,
        /// What the model's embedding table can index.
        model: u32,
    },
    /// A length field exceeds what the file can hold.
    ///
    /// Checked rather than trusted: a corrupt or truncated shard would
    /// otherwise ask for a multi-gigabyte allocation before failing.
    Truncated { wanted: usize, available: usize },
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Io(ref e) => write!(f, "{e}"),
            Self::NotAShard => write!(f, "not an Anaphora corpus shard"),
            Self::Version(v) => write!(f, "unsupported shard version {v}"),
            Self::VocabTooLarge { shard, model } => write!(
                f,
                "shard was tokenized against a {shard}-token vocabulary but the model's \
                 embedding table holds {model}; token ids beyond it would index out of range"
            ),
            Self::Truncated { wanted, available } => write!(
                f,
                "shard claims {wanted} more bytes but only {available} remain"
            ),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A tokenized document collection on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusShard {
    /// An upper bound on the token ids present.
    ///
    /// The tokenizer's reported vocabulary, which a model's embedding table
    /// must be at least as wide as. Not necessarily equal to it — see
    /// [`CorpusShard::read`].
    pub vocab_size: u32,
    /// The `[MASK]` token id.
    pub mask_token: u32,
    /// The documents.
    pub documents: Vec<Document>,
}

impl CorpusShard {
    /// Total tokens across every document.
    pub fn total_tokens(&self) -> usize {
        self.documents.iter().map(|d| d.tokens.len()).sum()
    }

    /// Write to `path`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), ShardError> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&self.vocab_size.to_le_bytes())?;
        w.write_all(&self.mask_token.to_le_bytes())?;
        w.write_all(&(self.documents.len() as u64).to_le_bytes())?;
        for doc in &self.documents {
            w.write_all(&doc.id.0.to_le_bytes())?;
            w.write_all(&(doc.tokens.len() as u32).to_le_bytes())?;
            for &token in &doc.tokens {
                w.write_all(&token.to_le_bytes())?;
            }
        }
        w.flush()?;
        Ok(())
    }

    /// Read from `path`, rejecting a shard the model cannot represent.
    ///
    /// `model_vocab` is the width of the model's embedding table, and the
    /// check is `shard.vocab_size <= model_vocab` rather than equality.
    /// The two legitimately differ: LLaDA-8B reports 126,464 embedding rows
    /// against a tokenizer of roughly 126,346 tokens, the remainder being
    /// padding for alignment. Demanding equality would reject a shard that is
    /// perfectly usable; demanding nothing would let a shard from a larger
    /// tokenizer index past the end of the table.
    pub fn read(path: impl AsRef<Path>, model_vocab: Option<u32>) -> Result<Self, ShardError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let mut r = BufReader::new(file);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(ShardError::NotAShard);
        }
        let version = read_u32(&mut r)?;
        if version != VERSION {
            return Err(ShardError::Version(version));
        }
        let vocab_size = read_u32(&mut r)?;
        let mask_token = read_u32(&mut r)?;
        if let Some(model) = model_vocab
            && vocab_size > model
        {
            return Err(ShardError::VocabTooLarge {
                shard: vocab_size,
                model,
            });
        }
        let doc_count = read_u64(&mut r)? as usize;

        // Every document costs at least 12 bytes of header, so a plausible
        // count is bounded by the file size. This is the cheap guard against a
        // corrupt length turning into a huge allocation.
        let mut remaining = file_len.saturating_sub(24);
        if doc_count.saturating_mul(12) > remaining {
            return Err(ShardError::Truncated {
                wanted: doc_count * 12,
                available: remaining,
            });
        }

        let mut documents = Vec::with_capacity(doc_count);
        for _ in 0..doc_count {
            let id = read_u64(&mut r)?;
            let len = read_u32(&mut r)? as usize;
            remaining = remaining.saturating_sub(12);
            let wanted = len * 4;
            if wanted > remaining {
                return Err(ShardError::Truncated {
                    wanted,
                    available: remaining,
                });
            }
            remaining -= wanted;
            let mut bytes = vec![0u8; wanted];
            r.read_exact(&mut bytes)?;
            let tokens = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            documents.push(Document {
                id: DocumentId(id),
                tokens,
            });
        }

        Ok(Self {
            vocab_size,
            mask_token,
            documents,
        })
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Which shard a document belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// The retrofit trains on these.
    Train,
    /// These are what retrieval can find.
    Index,
    /// Held out for the evaluation protocol.
    Eval,
}

/// Assign a document to a split by its id.
///
/// Deterministic and content-independent: the same article lands in the same
/// split on every machine and every rerun, which is what makes a leakage
/// audit reproducible. Hashing the id rather than taking `id % 3` avoids
/// inheriting whatever structure the source's numbering has — Wikipedia
/// article ids are allocated chronologically, so a modulus would correlate
/// splits with article age.
pub fn split_of(id: DocumentId, train: u32, index: u32, eval: u32) -> Split {
    let total = train + index + eval;
    assert!(total > 0, "split weights must not all be zero");
    let mut h = id.0 ^ 0x9E37_79B9_7F4A_7C15;
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    let bucket = (h % u64::from(total)) as u32;
    if bucket < train {
        Split::Train
    } else if bucket < train + index {
        Split::Index
    } else {
        Split::Eval
    }
}

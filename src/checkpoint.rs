//! A dull parameter dump so a trained play model does not have to start from
//! random weights every launch.
//!
//! This is not a production checkpoint format (see `docs/v0-plan.md`). It
//! exists so the interactive binary can save the tensors a training session
//! just wrote and reload them into an inference session of the same shape.

use meganeura::runtime::Session;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"ANAPCKP1";
const VERSION: u32 = 1;

/// Shape the dumped parameters were trained at.
///
/// Load rejects a checkpoint whose shape does not match the session it would
/// be written into: a `d = 64` dump in a `d = 128` table is silently valid
/// ids and completely wrong weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointMeta {
    /// Tokenizer / embedding table width.
    pub vocab_size: u32,
    /// Sequence length `n`.
    pub seq_len: u32,
    /// Chunk size `m`.
    pub chunk: u32,
    /// Backbone depth.
    pub num_layers: u32,
    /// Query heads.
    pub num_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// MLP inner width.
    pub intermediate_size: u32,
}

/// Named parameter tensors plus the shape they belong to.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// Shape tag.
    pub meta: CheckpointMeta,
    /// `(name, values)` in session order at save time.
    pub params: Vec<(String, Vec<f32>)>,
}

/// Why a checkpoint could not be read, written, or applied.
#[derive(Debug)]
pub enum CheckpointError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// The file does not begin with the checkpoint magic.
    NotACheckpoint,
    /// The file's format version is not understood.
    Version(u32),
    /// The dump was trained at a different shape than the live model.
    ShapeMismatch {
        /// What the file contains.
        file: CheckpointMeta,
        /// What the live model expects.
        model: CheckpointMeta,
    },
    /// Applying onto a session that lacks a dumped parameter, or disagrees
    /// on its length.
    ParamMismatch {
        /// Parameter name.
        name: String,
        /// Length in the dump, if the session has the name.
        dump: usize,
        /// Length on the session, if it declares the name.
        session: Option<usize>,
    },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Io(ref e) => write!(f, "{e}"),
            Self::NotACheckpoint => write!(f, "not an Anaphora parameter checkpoint"),
            Self::Version(v) => write!(f, "unsupported checkpoint version {v}"),
            Self::ShapeMismatch { file, model } => write!(
                f,
                "checkpoint shape vocab={} n={} m={} layers={} heads={} head_dim={} intermediate={} \
                 does not match model vocab={} n={} m={} layers={} heads={} head_dim={} intermediate={}",
                file.vocab_size,
                file.seq_len,
                file.chunk,
                file.num_layers,
                file.num_heads,
                file.head_dim,
                file.intermediate_size,
                model.vocab_size,
                model.seq_len,
                model.chunk,
                model.num_layers,
                model.num_heads,
                model.head_dim,
                model.intermediate_size,
            ),
            Self::ParamMismatch {
                ref name,
                dump,
                session,
            } => match session {
                None => write!(f, "checkpoint has {name} but the session does not"),
                Some(n) => write!(f, "checkpoint {name} has {dump} values, session has {n}"),
            },
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<std::io::Error> for CheckpointError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

impl Checkpoint {
    /// Snapshot every parameter currently on `session`.
    pub fn capture(session: &Session, meta: CheckpointMeta) -> Self {
        let params = session
            .param_names()
            .into_iter()
            .map(|name| {
                let len = session.param_size(name).expect("declared");
                let mut values = vec![0.0f32; len];
                session.read_param(name, &mut values);
                (name.to_owned(), values)
            })
            .collect();
        Self { meta, params }
    }

    /// Reject a dump whose shape does not match `expected`.
    pub fn check_shape(&self, expected: CheckpointMeta) -> Result<(), CheckpointError> {
        if self.meta != expected {
            return Err(CheckpointError::ShapeMismatch {
                file: self.meta,
                model: expected,
            });
        }
        Ok(())
    }

    /// Write `self`'s tensors onto `session`.
    ///
    /// Extra names on the session (for example a freshly declared encoder
    /// that the dump predates) are left untouched. Missing names and length
    /// mismatches are errors — applying a partial dump is how a run looks
    /// trained and is not.
    pub fn apply(
        &self,
        session: &mut Session,
        expected: CheckpointMeta,
    ) -> Result<(), CheckpointError> {
        self.check_shape(expected)?;
        for param in &self.params {
            let name = &param.0;
            let values = &param.1;
            let Some(len) = session.param_size(name) else {
                return Err(CheckpointError::ParamMismatch {
                    name: name.clone(),
                    dump: values.len(),
                    session: None,
                });
            };
            if len != values.len() {
                return Err(CheckpointError::ParamMismatch {
                    name: name.clone(),
                    dump: values.len(),
                    session: Some(len),
                });
            }
            session.set_parameter(name, values);
        }
        Ok(())
    }

    /// Serialize to any writer.
    pub fn write_to(&self, w: &mut impl Write) -> Result<(), CheckpointError> {
        w.write_all(MAGIC)?;
        write_u32(w, VERSION)?;
        write_u32(w, self.meta.vocab_size)?;
        write_u32(w, self.meta.seq_len)?;
        write_u32(w, self.meta.chunk)?;
        write_u32(w, self.meta.num_layers)?;
        write_u32(w, self.meta.num_heads)?;
        write_u32(w, self.meta.head_dim)?;
        write_u32(w, self.meta.intermediate_size)?;
        write_u32(w, self.params.len() as u32)?;
        for param in &self.params {
            let bytes = param.0.as_bytes();
            write_u32(w, bytes.len() as u32)?;
            w.write_all(bytes)?;
            write_u32(w, param.1.len() as u32)?;
            for &v in &param.1 {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()?;
        Ok(())
    }

    /// Deserialize from any reader.
    pub fn read_from(r: &mut impl Read) -> Result<Self, CheckpointError> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(CheckpointError::NotACheckpoint);
        }
        let version = read_u32(r)?;
        if version != VERSION {
            return Err(CheckpointError::Version(version));
        }
        let meta = CheckpointMeta {
            vocab_size: read_u32(r)?,
            seq_len: read_u32(r)?,
            chunk: read_u32(r)?,
            num_layers: read_u32(r)?,
            num_heads: read_u32(r)?,
            head_dim: read_u32(r)?,
            intermediate_size: read_u32(r)?,
        };
        let n_params = read_u32(r)? as usize;
        let mut params = Vec::with_capacity(n_params);
        for _ in 0..n_params {
            let name_len = read_u32(r)? as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf).map_err(|_| CheckpointError::NotACheckpoint)?;
            let n = read_u32(r)? as usize;
            let mut values = Vec::with_capacity(n);
            let mut buf = [0u8; 4];
            for _ in 0..n {
                r.read_exact(&mut buf)?;
                values.push(f32::from_le_bytes(buf));
            }
            params.push((name, values));
        }
        Ok(Self { meta, params })
    }

    /// Write to `path`, creating parent directories.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CheckpointError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Read from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let mut file = std::fs::File::open(path)?;
        Self::read_from(&mut file)
    }
}

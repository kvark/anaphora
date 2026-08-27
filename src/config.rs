//! Shape and hyper-parameter configuration for chunked cross-attention.
//!
//! The names follow the design sketch and RETRO before it:
//!
//! | symbol | meaning                | RETRO default |
//! |--------|------------------------|---------------|
//! | `n`    | sequence length        | 2048          |
//! | `m`    | chunk size             | 64            |
//! | `l`    | chunk count, `n / m`   | 32            |
//! | `k`    | neighbours per chunk   | 2             |
//! | `r`    | neighbour length       | `2 * m`       |
//! | `d`    | model dim              | backbone      |
//!
//! There is no `B` (batch). Meganeura's attention operators are
//! two-dimensional — `[seq, num_heads * head_dim]` — so one graph describes
//! one sequence, exactly as [`meganeura::models::smollm2`] does. Batching is
//! a matter of running the session repeatedly, not of a leading tensor axis.

/// Why a configuration was rejected.
///
/// Every variant is a shape contradiction that would otherwise surface far
/// downstream as a Meganeura shape panic inside an attention operator, with
/// no indication of which knob was wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// A dimension that must be positive was zero.
    Zero(&'static str),
    /// `n` is not an exact multiple of `m`, so the sequence does not divide
    /// into whole chunks.
    SequenceNotChunkAligned { n: usize, m: usize },
    /// `d` is not `num_heads * head_dim`.
    ModelDimMismatch {
        d: usize,
        num_heads: usize,
        head_dim: usize,
    },
    /// `num_heads` is not a whole multiple of `num_kv_heads`, so grouped-query
    /// attention cannot map query heads onto key/value heads.
    HeadGroupingMismatch {
        num_heads: usize,
        num_kv_heads: usize,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero(field) => write!(f, "`{field}` must be non-zero"),
            Self::SequenceNotChunkAligned { n, m } => write!(
                f,
                "sequence length n={n} is not a multiple of chunk size m={m}"
            ),
            Self::ModelDimMismatch {
                d,
                num_heads,
                head_dim,
            } => write!(
                f,
                "model dim d={d} != num_heads={num_heads} * head_dim={head_dim}"
            ),
            Self::HeadGroupingMismatch {
                num_heads,
                num_kv_heads,
            } => write!(
                f,
                "num_heads={num_heads} is not a multiple of num_kv_heads={num_kv_heads}"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Validated shapes for the retrieval path.
///
/// Construct with [`CcaConfig::new`], which rejects every shape combination
/// the graph builder cannot express.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcaConfig {
    n: usize,
    m: usize,
    k: usize,
    r: usize,
    d: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    insert_every: usize,
    first_cca_layer: usize,
}

impl CcaConfig {
    /// Validate a configuration.
    ///
    /// `insert_every` is RETRO's `P` — a CCA block goes in after every `P`th
    /// backbone layer, starting at `first_cca_layer`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        m: usize,
        k: usize,
        r: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        insert_every: usize,
        first_cca_layer: usize,
    ) -> Result<Self, ConfigError> {
        for (value, name) in [
            (n, "n"),
            (m, "m"),
            (k, "k"),
            (r, "r"),
            (num_heads, "num_heads"),
            (num_kv_heads, "num_kv_heads"),
            (head_dim, "head_dim"),
            (insert_every, "insert_every"),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero(name));
            }
        }
        if !n.is_multiple_of(m) {
            return Err(ConfigError::SequenceNotChunkAligned { n, m });
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(ConfigError::HeadGroupingMismatch {
                num_heads,
                num_kv_heads,
            });
        }
        Ok(Self {
            n,
            m,
            k,
            r,
            d: num_heads * head_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            insert_every,
            first_cca_layer,
        })
    }

    /// RETRO's published shape, adapted: `n=2048`, `m=64`, `k=2`, `r=2m`.
    ///
    /// The head geometry is the caller's, since it has to match the frozen
    /// backbone's residual width.
    pub fn retro_like(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ConfigError> {
        Self::new(2048, 64, 2, 128, num_heads, num_kv_heads, head_dim, 3, 6)
    }

    /// Sequence length `n`.
    pub fn seq_len(self) -> usize {
        self.n
    }

    /// Chunk size `m`.
    pub fn chunk_size(self) -> usize {
        self.m
    }

    /// Chunk count `l = n / m`.
    pub fn num_chunks(self) -> usize {
        self.n / self.m
    }

    /// Neighbours retrieved per chunk, `k`.
    pub fn neighbours_per_chunk(self) -> usize {
        self.k
    }

    /// Token length of one retrieved neighbour, `r`.
    ///
    /// RETRO used `2m`: the matched chunk plus its continuation. The
    /// continuation is the half that carries new information, since the
    /// matched half is by construction similar to what the model already has.
    pub fn neighbour_len(self) -> usize {
        self.r
    }

    /// Key/value rows one chunk's cross-attention sees, `k * r`.
    pub fn neighbour_kv_rows(self) -> usize {
        self.k * self.r
    }

    /// Model dim `d`, equal to `num_heads * head_dim`.
    pub fn model_dim(self) -> usize {
        self.d
    }

    /// Query head count.
    pub fn num_heads(self) -> usize {
        self.num_heads
    }

    /// Key/value head count (grouped-query attention).
    pub fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    /// Per-head width.
    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    /// Key/value width, `num_kv_heads * head_dim`.
    pub fn kv_dim(self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Whether a CCA block is inserted after backbone layer `layer`.
    pub fn is_cca_layer(self, layer: usize) -> bool {
        layer >= self.first_cca_layer
            && (layer - self.first_cca_layer).is_multiple_of(self.insert_every)
    }

    /// The backbone layer indices that receive a CCA block, given a backbone
    /// of `num_layers` layers.
    pub fn cca_layers(self, num_layers: usize) -> Vec<usize> {
        (0..num_layers).filter(|&i| self.is_cca_layer(i)).collect()
    }
}

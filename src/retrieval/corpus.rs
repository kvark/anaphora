//! The neighbour corpus: what retrieval returns and where it came from.

/// Identifies a source document in the corpus.
///
/// Provenance is not bookkeeping here — it is the primary leakage defense.
/// The training document is itself in the index, and the cheapest exact way
/// to avoid retrieving it is to know which document a neighbour came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocumentId(pub u64);

/// Identifies one retrievable neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NeighbourId(pub u32);

/// A stored neighbour: `r` tokens plus provenance.
///
/// RETRO's `r = 2m` is the matched chunk followed by its continuation. The
/// continuation is the half that carries information the model does not
/// already have — the matched half is by construction similar to the query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighbourRecord {
    /// Source document.
    pub document: DocumentId,
    /// Token offset of the matched chunk within the source document.
    pub offset: usize,
    /// `r` tokens: the matched chunk and its continuation.
    pub tokens: Vec<u32>,
}

/// Token storage for retrievable neighbours, and their index embeddings.
///
/// Flat and contiguous on purpose: the design sketch puts this on NVMe or in
/// host RAM at a scale where per-neighbour allocations are the wrong shape.
#[derive(Debug, Clone, Default)]
pub struct NeighbourCorpus {
    neighbour_len: usize,
    embed_dim: usize,
    /// `count * neighbour_len` token ids.
    tokens: Vec<u32>,
    /// `count * embed_dim` embedding values.
    embeddings: Vec<f32>,
    documents: Vec<DocumentId>,
    offsets: Vec<usize>,
}

/// Why a neighbour could not be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusError {
    /// The record's token count does not match the corpus's `r`.
    TokenLenMismatch { got: usize, expected: usize },
    /// The embedding width does not match the corpus's `d_r`.
    EmbedDimMismatch { got: usize, expected: usize },
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenLenMismatch { got, expected } => {
                write!(f, "neighbour has {got} tokens, corpus expects r={expected}")
            }
            Self::EmbedDimMismatch { got, expected } => write!(
                f,
                "embedding has width {got}, corpus expects d_r={expected}"
            ),
        }
    }
}

impl std::error::Error for CorpusError {}

impl NeighbourCorpus {
    /// An empty corpus storing `neighbour_len`-token neighbours indexed by
    /// `embed_dim`-wide embeddings.
    pub fn new(neighbour_len: usize, embed_dim: usize) -> Self {
        Self {
            neighbour_len,
            embed_dim,
            ..Default::default()
        }
    }

    /// Add a neighbour and its index embedding.
    pub fn push(
        &mut self,
        record: &NeighbourRecord,
        embedding: &[f32],
    ) -> Result<NeighbourId, CorpusError> {
        if record.tokens.len() != self.neighbour_len {
            return Err(CorpusError::TokenLenMismatch {
                got: record.tokens.len(),
                expected: self.neighbour_len,
            });
        }
        if embedding.len() != self.embed_dim {
            return Err(CorpusError::EmbedDimMismatch {
                got: embedding.len(),
                expected: self.embed_dim,
            });
        }
        let id = NeighbourId(self.documents.len() as u32);
        self.tokens.extend_from_slice(&record.tokens);
        self.embeddings.extend_from_slice(embedding);
        self.documents.push(record.document);
        self.offsets.push(record.offset);
        Ok(id)
    }

    /// Number of stored neighbours.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the corpus is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Neighbour token length, `r`.
    pub fn neighbour_len(&self) -> usize {
        self.neighbour_len
    }

    /// Index embedding width, `d_r`.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// The `r` tokens of neighbour `id`.
    pub fn tokens(&self, id: NeighbourId) -> Option<&[u32]> {
        let start = (id.0 as usize).checked_mul(self.neighbour_len)?;
        self.tokens.get(start..start + self.neighbour_len)
    }

    /// The index embedding of neighbour `id`.
    pub fn embedding(&self, id: NeighbourId) -> Option<&[f32]> {
        let start = (id.0 as usize).checked_mul(self.embed_dim)?;
        self.embeddings.get(start..start + self.embed_dim)
    }

    /// The source document of neighbour `id`.
    pub fn document(&self, id: NeighbourId) -> Option<DocumentId> {
        self.documents.get(id.0 as usize).copied()
    }

    /// The in-document offset of neighbour `id`.
    pub fn offset(&self, id: NeighbourId) -> Option<usize> {
        self.offsets.get(id.0 as usize).copied()
    }

    /// All embeddings, laid out `count * embed_dim`.
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }
}

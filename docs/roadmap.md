# Anaphora roadmap

What exists, what is deliberately deferred, and what is still an open
research question. Ordered roughly by what blocks what.

## Open research question: encoding a `[MASK]`-bearing query

This is the piece the design sketch marks unsolved, and it is still unsolved.

RETRO used a frozen BERT over clean text. Here the retriever's input is
partly `[MASK]`, a distribution a clean-text encoder has never seen. The
sketch lists three ways out, in increasing order of cost:

* **(a) Restrict retrieval to low-mask steps.** Implemented:
  `schedule::retrieve_now` gates on the global `t`, and
  `chunk::ChunkAdmission` gates per chunk on its own mask rate. Usable with a
  frozen encoder today. The ceiling is that it simply declines to retrieve
  wherever the query is degraded, which is a large part of the trajectory.
* **(b) Fine-tune the retriever on masked inputs, with `t` as a conditioning
  input.** `chunk::RetrieverEncode` already receives `t` for this. Needs a
  training objective for the retriever that is not the denoising loss —
  contrastive against clean-text embeddings of the same chunk is the obvious
  candidate.
* **(c) Train it jointly against the denoising loss.** The most expensive and
  the most likely to work. Also the most dangerous: a retriever trained
  against the denoising loss has a direct incentive to find the leak, so (c)
  raises the stakes on everything in `retrieval::leakage` and on the
  `ViewId` checks. Do not attempt (c) before the evaluation protocol below
  exists.

The seam is `chunk::RetrieverEncode`. Nothing above it needs to change to
swap strategies.

## Evaluation protocol for the silent failure

The structural guards in `view` and `retrieval::leakage` prevent the leaks we
know how to name. They cannot prove absence. Before any result is believed,
there needs to be an empirical check that a perplexity improvement is not
copy-from-neighbour:

* Train with retrieval and evaluate with neighbours replaced by random
  corpus entries. A model that has learned to *use* retrieval degrades
  gracefully; a model that has learned to *copy* falls off a cliff.
* Report the low-`t` loss separately. That is the band where copying pays
  off, and where the training gate is closed for exactly that reason.
* Track the gate values by `t`. A gate that saturates open at low `t` during
  training is the signature of the leak.

## Deferred: a fused chunked cross-attention operator

`model::cca` emits `l` independent cross-attentions per block, which is the
right arithmetic (see the README) but costs `l` dispatches. At RETRO's shape
that is 32 dispatches per CCA block, times the number of CCA blocks.

A `ChunkedCrossAttention` operator in Meganeura — one dispatch, chunk index
as a workgroup dimension — would fold those into one. It needs a new `Op`, a
WGSL generator, and an autodiff rule, so it is worth doing only once
profiling shows dispatch overhead actually dominates. Measure before
building: `MEGANEURA_GPU_TIMING=1` and the `profile_session` example give the
per-dispatch breakdown.

## Deferred: an approximate index

`retrieval::index::ExactIndex` is exhaustive and linear in corpus size. That
is correct, and it is the right baseline to measure recall against, but it is
not what the design sketch describes — an ANN index over NVMe or host RAM at
a scale where exhaustive search is not an option.

`retrieval::index::NeighbourIndex` is the seam. An IVF or HNSW backend
changes nothing above it. Whatever lands must keep the `accept` callback
semantics: exclusions apply *during* the search, so a chunk still receives
`k` neighbours when its nearest match is excluded. An implementation that
filters afterwards silently changes the experiment.

## Deferred: backbone weight loading

`model::backbone` declares a LLaDA-shaped bidirectional transformer with
frozen parameters, but nothing loads real weights into it yet. Meganeura's
`SafeTensorsModel` is the path; the work is the name mapping and the
transposition conventions, the same shape of work as
`meganeura::models::smollm2::weight_names`.

Until then the backbone is exercised with deterministic synthetic weights,
which is enough for the shape, freezing, and identity-at-init tests but
nothing more.

## Deferred: batching

Meganeura's attention operators are two-dimensional — `[seq, heads·head_dim]`
— so one graph describes one sequence and there is no leading batch axis.
This matches how `meganeura::models::smollm2` is built. Batching today means
running the session repeatedly.

If throughput needs it, the options are a batch axis in Meganeura's attention
operators (large, invasive) or folding the batch into the sequence with a
block-diagonal mask (which Anaphora already does for chunks, so the machinery
is familiar). Neither is worth doing before there is a training run to
measure.

## Deferred: session-level neighbour KV caching

`model::NeighbourInput::Cached` lets the host feed encoded neighbours as a
graph input, which is what makes the refresh schedule expressible: encode
once per refresh threshold, reuse across the steps in between.

What is missing is the host-side glue that owns that buffer across a
trajectory and writes it only on refresh. `sample::sample` currently caches
the `Neighbours` (token ids) rather than the encoded keys/values, so the
saving it demonstrates is index traffic, not encoder compute. Closing this
needs the encoder to run as its own small session whose output feeds the
denoiser's `cca.neighbour_kv` input — ideally through
`Session::bind_external_buffer`, to keep the keys/values on the device
between the two.

## Committed tokens cannot be revised, and that bounds the Phase 3 claim

Masked diffusion's forward process never re-masks an unmasked position, so
the model is never trained to correct itself: once the sampler commits a
token, it stays. This is a known limitation of the formulation rather than
anything specific to Anaphora, but it interacts with the project's central
claim in a way worth stating.

Re-querying on a sharpening sketch improves the *query*. It cannot repair a
token the sampler already committed on the strength of a worse one. So the
value of refreshing is bounded by how much of the sequence is still open when
the better neighbours arrive — which argues for refresh thresholds early
enough to matter, and against reading a late refresh as if it could rescue an
early mistake.

Two ways out, both of which change the backbone rather than the retrieval
path, and neither of which is on V0's route:

* a remasking-capable variant, so low-confidence commits can be reconsidered;
* uniform-state (rather than absorbing-state) diffusion, where every position
  stays revisable throughout.

If the Phase 3 ablation shows refreshing buys little, this is the first thing
to check before concluding that re-querying does not help: the retrieval may
be improving while the sampler is unable to act on it.

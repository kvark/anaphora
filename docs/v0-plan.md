# Anaphora: the plan to V0

## What V0 is

**V0 is the smallest run that gives a trustworthy answer to one question: does
retrieval help a masked diffusion LM, and is the gain real?**

Not "a working retrieval-augmented 8B model." The distinction matters because
this project's characteristic failure is silent *and flattering* — a leaked
query improves perplexity. A V0 that produces a number nobody can trust is
worse than no V0, because it costs the same and then has to be re-run.

Everything below is ordered by that definition.

## Phase 1 status

The machinery is complete and runs on real data. What is missing is a GPU.

**Built and tested:** the masked-diffusion objective, the corpus and index
pipeline, the host training loop, the evaluation protocol, the leak
calibration harness, and the ingest path from a Wikipedia parquet dump to
training shards.

**The corpus exists.** `prepare_corpus` was run against `20231101.simple`:
241,787 articles, 131,624 kept above 128 tokens.

| shard | documents | tokens |
|---|---:|---:|
| train | 105,185 | 55.1M |
| index | 13,202 | 7.3M |
| eval | 13,237 | 7.2M |

`examples/phase1.rs` runs end to end against it — pretrain, freeze, retrofit,
protocol — on a software Vulkan device.

**What has not happened is a real run.** Lavapipe takes roughly twenty
seconds per step at this vocabulary, so the runs done here are single-digit
step counts, and at those the zero-init gate has barely opened: every
evaluation condition returns the same number. That is the identity property
holding, not a result, and the protocol now declines to report a copy ratio
in that regime rather than dividing one near-zero difference by another.

So the calibration verdict comes from hardware, not from here. On a real GPU:

```sh
cargo run --release --example phase1 -- --corpus corpus/
cargo run --release --features leak-harness --example leak_calibration
```

The first produces the protocol table. The second must flag its leaked arm
before the first table means anything.


## Where the code stands

Built and tested: the CCA block, the gates, the frozen-backbone wiring, the
chunking and view-identity guards, the schedule, the leakage guards, an exact
index, and the sampling loop's control flow.

Not built: **everything that turns it into a run.** Specifically —

| Gap | Size | Blocks |
|---|---|---|
| Host training loop | done | — |
| Masked-diffusion loss construction | done | — |
| Corpus + index build pipeline | done | — |
| Evaluation harness | done | — |
| Backbone weight loading | medium | Phase 2 only |
| Retriever encoder for masked queries | **open research** | quality, not correctness |

### The loss needs no new operator

Meganeura's `cross_entropy_loss` computes `L = -Σ labels·log_softmax(logits)`
with gradient `softmax·S − labels` where `S = Σ labels` — it generalizes to
arbitrary per-class label *weights*, not just probability distributions. The
shader was written for advantage-scaled policy gradients, but the masked
diffusion objective is the same shape:

```
L = -1/(t·L) · Σ_{i masked} log p(x₀ⁱ | x_t)
```

is exactly what the existing kernel computes if the host builds labels as:

* masked position `i` → `onehot(x₀ⁱ) / t`
* unmasked position → **all zeros**

A zero label row contributes zero loss and, since `S = 0`, zero gradient. The
kernel's own `1/batch` division over the `n` rows supplies the `1/L`. So the
objective is a host-side label-construction detail, not a framework change.

### The dense label cost turned out not to bind

The operator's signature is a dense `[n, vocab]` tensor, which at LLaDA's
vocabulary is 259 MB at `n = 512` and over a gigabyte at `n = 2048` — to
carry at most `n` non-zero values. That looked like a reason to give Phase 1
a smaller tokenizer.

It is not, because it is not a per-step cost. Meganeura's input buffers are
pinned by the memory plan — never aliased — and allocated `Memory::Shared`,
so they are host-coherent and their contents survive between steps.
`SparseLabels` keeps the buffer zeroed and each step clears only what the
previous step wrote: about 2 KB of traffic instead of 259 MB. Only the
allocation remains, and an indexed-label cross-entropy in Meganeura would
remove that too — worth doing at `n = 2048`, not needed before.

So the original call stands: reuse LLaDA's tokenizer in both phases, and the
corpus pipeline carries into Phase 2 unchanged.

### The training loop cannot reuse `Trainer`

`meganeura::DataLoader` streams `Vec<f32>` only, and this graph takes two U32
inputs (`token_ids`, `cca.neighbour_tokens`) alongside its f32 ones. So the
host loop drives `Session` directly: sample `t`, mask, build labels, run
retrieval, `set_input_u32` / `set_input`, `step`. Labels go through
`SparseLabels` rather than `set_input`, for the reason above.

## Datasets

**`wikimedia/wikipedia`, config `20231101.simple` for Phase 1 and
`20231101.en` for Phase 2.** (cc-by-sa-3.0 / GFDL.)

| Config | Articles | Parquet |
|---|---:|---:|
| `20231101.simple` | 241.8K | 156.9 MB |
| `20231101.en` | 6.4M | 11.6 GB |

Chosen for three properties, in order of importance:

1. **Real document boundaries with stable ids.** The schema is
   `(id, url, title, text)`, one row per article. `DocumentId` maps onto `id`
   directly, which is what makes the provenance-based leakage guard exact
   rather than heuristic. A corpus shipped as an undifferentiated token
   stream — most web dumps — cannot support that guard at all.
2. **Factual, self-contained prose.** Retrieval has something to contribute.
   On a corpus where neighbours add only style, the experiment measures
   nothing whichever way it comes out.
3. **Simple English is V0-sized.** 157 MB indexes exhaustively with the
   `ExactIndex` already in the tree, so Phase 1 needs no ANN work, and exact
   search is the recall baseline an approximate backend gets measured against
   later.

### Splitting

Partition by article id into **train / index / eval**, and run the index side
two ways, because the difference between them is itself a measurement:

* **(i) Disjoint index** — index shard only. No leak is possible. Understates
  what retrieval is worth, and is the honest floor.
* **(ii) Overlapping index** — index contains the training articles too, with
  `LeakageGuard::by_source_document` plus the offline n-gram audit doing the
  excluding. This is RETRO's actual setting, and it exercises the guard.

If (ii) beats (i) by much more than the retrieval-quality difference explains,
the guard is leaking. That comparison is free and should run every time.

### Tokenizer

Reuse LLaDA's tokenizer (vocab 126464, `mask_token_id` 126336) in **both**
phases, even though a 16K BPE trained on Simple English would give Phase 1 a
much smaller embedding table. The corpus pipeline — chunking, continuation
pairing, embedding, indexing, the n-gram audit — then carries from Phase 1 to
Phase 2 unchanged, and that pipeline is a bigger investment than the extra
embedding parameters cost to train.

## Phases

### Phase 1 — calibrate the instrument

**A small backbone trained from scratch, on Simple English Wikipedia.**

Roughly `d=512` (8 heads × 64), 8 layers, `intermediate=1408`, `n=512`,
`m=64` → `l=8`, `k=2`, `r=128`, CCA every 2 layers from layer 2 (3 blocks).
About 155M parameters, most of it the vocab-sized embedding.

Training our own backbone rather than loading LLaDA first is deliberate:

* no dependency on the weight-loading work,
* total control over what is in the index versus the training set,
* iteration in minutes,
* and the reason that matters most below.

**Exit criterion — and this is the whole point of Phase 1:**

> Deliberately build the leak, and confirm the evaluation protocol catches it.

Run a second training job through a test-only harness that constructs queries
from `x₀` instead of `x_t` — the exact thing `view` exists to prevent. Its
perplexity should look *better*. Then check that every metric in the protocol
below flags it.

If the protocol does not flag the leaked run, the protocol is not ready, and
no number Phase 2 produces can be trusted. Calibrate the instrument before
measuring anything with it.

### Phase 2 — does it help a real model

**LLaDA-8B-Base, frozen.**

Its config maps onto `BackboneConfig` with nothing left over:

| LLaDA-8B-Base | value | `BackboneConfig` |
|---|---|---|
| `n_heads` / `n_kv_heads` | 32 / 32 | `num_heads` / `num_kv_heads` |
| `d_model` | 4096 | `num_heads * head_dim` (32 × 128) |
| `mlp_hidden_size` | 12288 | `intermediate_size` |
| `n_layers` | 32 | `num_layers` |
| `vocab_size` | 126464 | `vocab_size` |
| `rope_theta` | 500000 | `rope_theta` |
| `rms_norm_eps` | 1e-5 | `rms_norm_eps` |
| `mask_token_id` | 126336 | `MaskToken` |

`block_type` is `llama` and `weight_tying` is false, so the backbone module
already has the right shape. What is missing is the safetensors name mapping
and the transpose conventions — the same work as
`meganeura::models::smollm2::weight_names`.

Note the LLaDA 2.x line (`inclusionAI/LLaDA2.1-mini`, `LLaDA2.2-flash`) is
**MoE** — 256 experts, 8 per token. Despite the name, "mini" is not a smaller
dense model and is not a drop-in for this backbone. LLaDA-8B-Base is the
target.

#### The memory problem is the trained parameters, not the backbone

This is counterintuitive and worth getting right before committing to a
machine.

The frozen backbone is cheap: no gradients, no optimizer state, and it can go
through `parameter_q4` / `parameter_q8`, which Meganeura dequantizes in the
matmul shader. Roughly 3.8 GB at Q4 for the 32 layers, plus ~2 GB keeping the
embedding and LM head at f16 (there is no `embedding_q4` — the gather path is
f32/f16 only). Call it **6 GB**.

The *trained* side is where it goes wrong. Nine CCA blocks at `d=4096` is
about 750M parameters, and with f32 master weights plus Adam's two moments
that is roughly **9 GB** — more than the frozen 8B model it is bolted onto.

Three ways down, in increasing order of how much they change the experiment:

1. **Insert less often.** `P=6` instead of RETRO's `P=3` halves it.
2. **Bottleneck the CCA projections.** Project `d → d_cca` (say 1024),
   cross-attend there, project back. At `d_cca=1024` the block drops roughly
   4×. RETRO had no reason to do this — it trained the whole model — but for
   a retrofit the CCA width is a free parameter, and there is no reason it
   must equal the residual width.
3. **Share the frozen embedding with the neighbour encoder** instead of
   giving the encoder its own table.

Measure this before choosing hardware. It is arithmetic, not a mystery, but
it inverts the intuition that freezing the backbone makes memory a non-issue.

### Phase 3 — the claim that is actually new

Everything above would be true of RETRO. The diffusion-specific claim is
**re-querying on a sharpening sketch**, and V0 is not complete without
testing it:

* retrieve once at the first admitted `t`, versus
* refresh at thresholds (`RefreshSchedule` already implements the control).

If refreshing does not beat retrieving once, the project is "RETRO, ported"
and should say so. That ablation is cheap — the machinery is written — and it
is the one result that justifies the premise.

## Evaluation protocol

Seven measurements. All of them must flag the Phase 1 leaked run.

1. **Random-neighbour ablation** *(primary copy detector)*. Evaluate the
   trained model with neighbours replaced by random corpus entries. A model
   that learned to *use* retrieval degrades gracefully; one that learned to
   *copy* falls off a cliff. The gap between real and random is the headline
   number, more than perplexity itself.
2. **Gate-zero ablation.** Same weights, `retrieval_mask` forced to 0.
   Isolates what the CCA blocks add over the frozen backbone.
3. **Oracle neighbour.** Feed the true continuation as a neighbour. Upper
   bound; also a sanity check that the CCA path can transmit information at
   all.
4. **Loss by `t` band.** Report in ~5 bands rather than one number. Copying
   concentrates at low `t`, which is exactly where the training gate is
   closed and where a single averaged number hides it.
5. **Gate magnitude by `t`.** Plot `|g|` against `t`. A gate saturating open
   at low `t` during training is the leak's signature.
6. **Index-membership A/B.** Disjoint index (i) versus overlapping index with
   guards (ii), per the split section above.
7. **Deduplicated eval.** N-gram overlap between each eval document and the
   index; report metrics separately on the low-overlap subset. Retrieval
   papers leak here routinely and it is cheap to rule out.

### Task metrics

* **Perplexity** on the held-out shard, by `t` band.
* **Entity infilling** on held-out articles: mask entity-bearing spans and
  measure exact-match recovery. This is the right retrieval-sensitive task
  for a *base* model — no instruction tuning needed — and it is diffusion-
  native, since infilling is what a masked diffusion model does and what an
  autoregressive model cannot do directly.
* **TriviaQA** (`mandarjoshi/trivia_qa`) is the conventional
  retrieval-heavy benchmark, but it wants an instruction-tuned model. Defer
  to after V0, or run against `LLaDA-8B-Instruct`.

## Workstream order

Dependency-ordered. Items 1–4 are Phase 1's critical path.

1. ~~**`MaskedDiffusionLoss`**~~ — done. Label construction, `1/t` weighting,
   zero rows for unmasked positions, checked against a CPU reference and
   against the kernel's gradient.
2. ~~**Corpus pipeline**~~ — done. `prepare_corpus` and the shard format;
   `build_corpus` chunks at `m` with `r`-token continuations and the audit
   runs offline.
3. ~~**Host training loop**~~ — done. Checkpointing is still just
   `Session::read_param` at the call site; there is no checkpoint format.
4. **Phase 1 backbone + the leaked-run calibration.** Built; the verdict
   needs a GPU. This is the remaining item.
5. ~~**Evaluation harness**~~ — done, as `eval::Evaluator`. Entity infilling
   is not implemented; perplexity by band is.
6. **Backbone weight loading** — safetensors name mapping for LLaDA-8B-Base.
7. **Quantized frozen backbone** — `parameter_q4`/`q8`, plus the CCA width
   decision from the memory section.
8. **Neighbour KV caching across denoising steps** — see `roadmap.md`; needed
   for Phase 3's refresh ablation to be affordable, not for correctness.

The retriever-encoder research question (`roadmap.md`) sits alongside all of
this. Phase 1 runs on option (a) — the hard gate plus per-chunk admission,
already implemented — which is enough to get a trustworthy number and not
enough to get a good one.

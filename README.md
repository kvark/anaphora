# anaphora

**Diffusion Reasoner** — retrieval-augmented masked-diffusion language
modeling in pure Rust, on [Meganeura](https://github.com/kvark/meganeura).

Chunked cross-attention (CCA), as RETRO defined it, under a masked diffusion
LM instead of an autoregressive one. The shape is a retrofit: freeze a
pretrained diffusion backbone (LLaDA-8B, Dream-7B) and train only the
neighbour encoder, the CCA blocks, and the gates.

> **Status:** early. The crate builds, trains, and samples, and the retrieval
> path is covered by tests on a software Vulkan device. The neighbour
> encoder's treatment of `[MASK]`-bearing queries is still the open research
> question — see [`docs/roadmap.md`](docs/roadmap.md).
>
> The route to a first trustworthy result — training workflow, datasets,
> and the evaluation protocol — is in [`docs/v0-plan.md`](docs/v0-plan.md).

## Why diffusion changes the retrieval story

RETRO retrieves once, before generation, because an autoregressive model's
context only grows. A diffusion model's context **sharpens**: every denoising
step yields a cleaner view of the whole sequence. So retrieval moves *inside*
the denoising loop, and the model re-queries on a progressively better
sketch — early steps on a rough semantic gist, later steps on something close
to the final text. That is the capability autoregressive RETRO does not have.

It also removes RETRO's chunk offset. `C_u+` existed so chunk `u` would not
attend to neighbours retrieved using chunk `u`'s own tokens; diffusion has no
ordering to exploit for that. The principle it protected survives in a nastier
form, and it is what this crate is most careful about.

## The failure this crate is built to prevent

> The retriever may only see what the denoiser sees.

Build retrieval queries from `x_0` during training and the neighbours
correlate with exactly the tokens that were masked. The loss collapses into
copy-from-neighbour and the experiment measures nothing. **The failure is
silent — perplexity improves.**

Three mechanisms make it hard to write by accident:

1. **`CleanSequence` is inert.** It exposes no accessor that reaches the
   retrieval path, and `chunk_queries` accepts only a `NoisedView`.
2. **Views have identity.** Every `NoisedView` carries a `ViewId`, and
   everything derived from it carries that id forward, so retrieving against
   a *different, cleaner* view of the same sequence is caught too — which
   inertness alone does not catch.
3. **Leakage control is split in two.** Provenance exclusion runs at query
   time; the n-gram audit runs offline at corpus preparation time, against
   the clean document. An inline n-gram filter against `x_t` — the design
   sketch's form — is blindest exactly where the leak lives, because masked
   positions cannot match anything.

## Layout

| module | design sketch |
|---|---|
| `view`, `chunk` | §1 query construction |
| `model::cca`, `model::gate` | §2 the CCA block |
| `schedule` | §3 the hard gate |
| `sample` | §4 inference inside the denoising loop |
| `retrieval` | index, corpus, leakage control |

## Two corrections to the design sketch

**The zero-init gate was not the identity.** The sketch zero-initialises the
gate MLP's last layer and applies a sigmoid, with the comment *"at step 0 the
block is the identity, so the frozen backbone is undisturbed"*. But
`sigmoid(0) = 0.5`, so the block returned `h + 0.5·ctx` and injected half an
untrained cross-attention output into the frozen residual stream on the first
forward pass. `model::gate` offers two activations that are exactly zero at
zero pre-activation *and* have a healthy derivative there — `Tanh` and
`ScaledSigmoid` (sigmoid times a zero-init learned scalar). Both are covered
by a test that compares logits against a bare backbone and requires an exact
match.

**Inline n-gram filtering against `x_t` cannot see the leak.** See mechanism 3
above; `retrieval::leakage` documents the reasoning in full.

## Block-diagonal attention without a mask

Chunk `u`'s queries may attend only to its own `k · r` retrieved key/value
rows. With a batch axis and an additive mask this is one `[n, l·k·r]`
attention with a block-diagonal mask, and most of the score matrix is thrown
away.

Meganeura's attention operators are two-dimensional and take no mask, so
Anaphora writes the block-diagonal form directly: `l` independent `[m, k·r]`
cross-attentions. This is not a workaround — the masked form computes
`n·l·k·r` scores and discards all but `1/l` of them, while the explicit form
computes exactly the `l·m·k·r` that survive. What it costs instead is `l`
dispatches per block rather than one.

## Build

```sh
cargo test
```

The GPU tests need a Vulkan or Metal device. On a headless Linux box, Mesa's
software Vulkan device works:

```sh
sudo apt-get install -y mesa-vulkan-drivers
```

## Play

Interactive generation (type a line, get a decoded trajectory; `quit` exits). Weights load from `runs/play.ckpt` when present. `--train` (or a missing dump) pretrains on Simple English Wikipedia and writes that file. Turns do not rebuild the graph.

```sh
cargo run --release --features text --bin play -- --train
cargo run --release --features text --bin play
cargo run --release --features text --bin play -- --prompt "The cat sat"
```

Needs `corpus/tokenizer.json` and the `train.shard` / `index.shard` produced by `prepare_corpus`. Pin the discrete GPU with `MEGANEURA_DEVICE_ID=0x744c`.

## License

MIT.

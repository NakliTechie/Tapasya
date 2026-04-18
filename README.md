# Tapasya

Train a language model in your browser — three stages, no server, no account, no data leaving your device.

**[→ Try it live](https://naklitechie.github.io/Tapasya/)**

## What it does

Tapasya walks you through three progressively sophisticated training regimes on your own text corpus, using only forward passes — no backprop, no PyTorch, no GPU driver installation.

| Stage | Tokenizer | Steps | What you see |
|---|---|---|---|
| 1 | Byte-level (vocab 256) | 500 | A 130K-parameter transformer learning your corpus's byte statistics |
| 2 | Byte-level | 2,000 | Same architecture, more compute — the compute lever |
| 3 | BPE (vocab 2,000) | 2,000 | Learned sub-word tokens — the vocabulary lever |

After any two stages complete, **Compare mode** lets you generate text from all three models side-by-side with the same seed.

## The algorithm

**EGGROLL** — [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652) (arXiv:2511.16652, 2025). Forward-only training: a population of perturbed model copies are evaluated, fitness is z-score normalised, and a gradient estimate is computed by resampling the same perturbations. No autograd. No backprop.

Tapasya uses standard isotropic perturbations (scalar σ applied to i.i.d. Gaussian noise). The paper's key innovation is replacing these with low-rank matrix products, which compresses the per-parameter cost from O(mn) to O(r(m+n)) and makes ES viable for billion-parameter models running at 91% of H100 inference throughput. At 130K parameters, isotropic perturbations are trivially fast and the extra complexity isn't warranted.

## How to run

```bash
cd Tapasya
python3 -m http.server 7771
# open http://localhost:7771
```

No build step. No npm. No dependencies.

For WebGPU acceleration, use Chrome or Edge on a machine with a GPU. Falls back to CPU automatically — training just takes longer.

---

## Tech

- **ES (EGGROLL-style)** — isotropic-perturbation evolution strategies, ~400 lines of JS; see [arXiv:2511.16652](https://arxiv.org/abs/2511.16652)
- **WebGPU** — GPU-accelerated forward pass + cross-entropy via custom WGSL kernels
- **BPE tokenizer** — hand-rolled `trainBPE` / `bpeEncode` / `bpeDecode`, ~60 lines
- **OPFS** — checkpoint save/restore so training survives a page reload
- Zero external dependencies. Four files (`index.html`, `main.js`, `worker.js`, `shaders.wgsl`).

Part of the [NakliTechie](https://naklitechie.github.io/) browser-native series.

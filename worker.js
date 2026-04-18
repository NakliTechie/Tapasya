'use strict';

// ============================================================
// COUNTER-BASED RNG  (no mutable global state)
// ============================================================

// SplitMix32 — one step of the mixing function
function sm32(h) {
  h = (Math.imul((h >>> 0) + 0x9e3779b9, 1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function hash2(a, b) { return sm32((sm32(a >>> 0) + (b >>> 0)) >>> 0); }

// Gaussian for weight init — seeded by (seed, tensorId, elementIdx)
function rngInitG(seed, tid, idx) {
  const h = hash2(hash2(seed >>> 0, tid >>> 0), idx >>> 0);
  const u1 = hash2(h, idx * 2 + 0) / 4294967296 + 1e-10;
  const u2 = hash2(h, idx * 2 + 1) / 4294967296;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Gaussian for ES perturbations — seeded by (seed, stage, step, popIdx, tensorId, elementIdx)
function rngEps(seed, stage, step, pop, tid, idx) {
  const h1 = hash2(hash2(seed >>> 0, (stage * 0x9e3779b9) >>> 0), (step * 0x6c62272e) >>> 0);
  const h2 = hash2(hash2(h1, pop >>> 0), tid >>> 0);
  const u1 = hash2(h2, idx * 2 + 0) / 4294967296 + 1e-10;
  const u2 = hash2(h2, idx * 2 + 1) / 4294967296;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Stateful LCG — only for batch sampling and sample generation (not training)
function makeLcg(seed, stage, step) {
  let s = sm32(sm32(seed >>> 0) ^ sm32((stage * 997 + step * 31 + 42) >>> 0));
  return () => { s = sm32(s + 0x6c62272e); return s / 4294967296; };
}

// ============================================================
// MODEL  (byte-level and BPE-tokenized — same transformer shape)
// ============================================================

function xavierFill(arr, fanIn, fanOut, seed, tid) {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  for (let i = 0; i < arr.length; i++) arr[i] = rngInitG(seed, tid, i) * std;
}

function initModel(cfg, seed) {
  const { vocab_size: V, d_model: D, n_heads: H, n_layers: L, d_ff: F } = cfg;
  let tid = 0;

  const embed     = new Float32Array(V * D); xavierFill(embed, V, D, seed, tid++);
  const pos_embed = new Float32Array(cfg.ctx_len * D); xavierFill(pos_embed, cfg.ctx_len, D, seed, tid++);

  const blocks = [];
  for (let l = 0; l < L; l++) {
    const wq = new Float32Array(D * D); xavierFill(wq, D, D, seed, tid++);
    const wk = new Float32Array(D * D); xavierFill(wk, D, D, seed, tid++);
    const wv = new Float32Array(D * D); xavierFill(wv, D, D, seed, tid++);
    const wo = new Float32Array(D * D); xavierFill(wo, D, D, seed, tid++);
    const ff1 = new Float32Array(D * F); xavierFill(ff1, D, F, seed, tid++);
    const ff2 = new Float32Array(F * D); xavierFill(ff2, F, D, seed, tid++);
    blocks.push({
      ln1_g: new Float32Array(D).fill(1), ln1_b: new Float32Array(D),
      wq, wk, wv, wo,
      ln2_g: new Float32Array(D).fill(1), ln2_b: new Float32Array(D),
      ff1, ff2
    });
  }

  return { cfg, embed, pos_embed, blocks,
           ln_f_g: new Float32Array(D).fill(1), ln_f_b: new Float32Array(D) };
}

// Ordered flat tensor list — index = tensor ID used by rngEps
function modelTensors(m) {
  const t = [m.embed, m.pos_embed];
  for (const b of m.blocks)
    t.push(b.wq, b.wk, b.wv, b.wo, b.ff1, b.ff2, b.ln1_g, b.ln1_b, b.ln2_g, b.ln2_b);
  t.push(m.ln_f_g, m.ln_f_b);
  return t;
}

// ============================================================
// FORWARD PASS  (pure CPU, returns logits [T × V])
// ============================================================

// A[T,M] × B[M,N] → C[T,N]  (cache-friendly loop order)
function matmul(A, B, T, M, N) {
  const C = new Float32Array(T * N);
  for (let t = 0; t < T; t++) {
    const rA = t * M, rC = t * N;
    for (let m = 0; m < M; m++) {
      const a = A[rA + m], rB = m * N;
      for (let n = 0; n < N; n++) C[rC + n] += a * B[rB + n];
    }
  }
  return C;
}

function layerNorm(src, off, g, b, D, dst, dOff) {
  let mean = 0;
  for (let i = 0; i < D; i++) mean += src[off + i];
  mean /= D;
  let v = 0;
  for (let i = 0; i < D; i++) v += (src[off + i] - mean) ** 2;
  const std = Math.sqrt(v / D + 1e-5);
  for (let i = 0; i < D; i++) dst[dOff + i] = g[i] * (src[off + i] - mean) / std + b[i];
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

// Causal multi-head self-attention
function attention(x, wq, wk, wv, wo, T, D, H) {
  const dh = D / H, scale = 1 / Math.sqrt(dh);
  const Q = matmul(x, wq, T, D, D);
  const K = matmul(x, wk, T, D, D);
  const V = matmul(x, wv, T, D, D);
  const out = new Float32Array(T * D);

  for (let h = 0; h < H; h++) {
    const off = h * dh;
    for (let i = 0; i < T; i++) {
      const scores = new Float32Array(i + 1);
      let mx = -Infinity;
      for (let j = 0; j <= i; j++) {
        let s = 0;
        for (let d = 0; d < dh; d++) s += Q[i * D + off + d] * K[j * D + off + d];
        scores[j] = s * scale;
        if (scores[j] > mx) mx = scores[j];
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
      for (let j = 0; j <= i; j++) scores[j] /= sum;

      for (let d = 0; d < dh; d++) {
        let val = 0;
        for (let j = 0; j <= i; j++) val += scores[j] * V[j * D + off + d];
        out[i * D + off + d] = val;
      }
    }
  }
  return matmul(out, wo, T, D, D);
}

function forward(model, tokens) {
  const { cfg, embed, pos_embed, blocks, ln_f_g, ln_f_b } = model;
  const { vocab_size: V, d_model: D, n_heads: H, n_layers: L, d_ff: F } = cfg;
  const T = tokens.length;

  // Token + position embeddings
  const x = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let d = 0; d < D; d++)
      x[t * D + d] = embed[tokens[t] * D + d] + pos_embed[t * D + d];

  const buf = new Float32Array(T * D);
  for (const { ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, ff1, ff2 } of blocks) {
    // Attention sublayer
    for (let t = 0; t < T; t++) layerNorm(x, t * D, ln1_g, ln1_b, D, buf, t * D);
    const attn = attention(buf, wq, wk, wv, wo, T, D, H);
    for (let i = 0; i < T * D; i++) x[i] += attn[i];

    // FFN sublayer
    for (let t = 0; t < T; t++) layerNorm(x, t * D, ln2_g, ln2_b, D, buf, t * D);
    const ff = matmul(buf, ff1, T, D, F);
    for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
    const ff2out = matmul(ff, ff2, T, F, D);
    for (let i = 0; i < T * D; i++) x[i] += ff2out[i];
  }

  // Final LN + logits (weight-tied to embed)
  const last = new Float32Array(D);
  layerNorm(x, (T - 1) * D, ln_f_g, ln_f_b, D, last, 0);
  const logits = new Float32Array(V);
  for (let v = 0; v < V; v++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += last[d] * embed[v * D + d];
    logits[v] = s;
  }
  return logits; // [V]
}

// All-positions forward (for training loss)
function forwardAll(model, tokens) {
  const { cfg, embed, pos_embed, blocks, ln_f_g, ln_f_b } = model;
  const { vocab_size: V, d_model: D, n_heads: H, n_layers: L, d_ff: F } = cfg;
  const T = tokens.length;

  const x = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let d = 0; d < D; d++)
      x[t * D + d] = embed[tokens[t] * D + d] + pos_embed[t * D + d];

  const buf = new Float32Array(T * D);
  for (const { ln1_g, ln1_b, wq, wk, wv, wo, ln2_g, ln2_b, ff1, ff2 } of blocks) {
    for (let t = 0; t < T; t++) layerNorm(x, t * D, ln1_g, ln1_b, D, buf, t * D);
    const attn = attention(buf, wq, wk, wv, wo, T, D, H);
    for (let i = 0; i < T * D; i++) x[i] += attn[i];
    for (let t = 0; t < T; t++) layerNorm(x, t * D, ln2_g, ln2_b, D, buf, t * D);
    const ff = matmul(buf, ff1, T, D, F);
    for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
    const ff2out = matmul(ff, ff2, T, F, D);
    for (let i = 0; i < T * D; i++) x[i] += ff2out[i];
  }

  // Apply final LN to all positions, then logits via weight tying: h · embed[v]
  const finalH = new Float32Array(T * D);
  for (let t = 0; t < T; t++) layerNorm(x, t * D, ln_f_g, ln_f_b, D, finalH, t * D);

  // embed is [V, D]; logits[t, v] = Σ_d finalH[t, d] * embed[v, d]  (not a plain matmul)
  const logits = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    const hOff = t * D, lOff = t * V;
    for (let v = 0; v < V; v++) {
      let s = 0;
      const eOff = v * D;
      for (let d = 0; d < D; d++) s += finalH[hOff + d] * embed[eOff + d];
      logits[lOff + v] = s;
    }
  }
  return logits; // [T, V]
}

// ============================================================
// LOSS
// ============================================================

// logits: [T, V]  targets: array-like [T]  returns scalar
function crossEntropy(logits, targets, T, V) {
  let loss = 0;
  for (let t = 0; t < T; t++) {
    const off = t * V;
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[off + v] > mx) mx = logits[off + v];
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[off + v] - mx);
    loss += -(logits[off + targets[t]] - mx - Math.log(sum));
  }
  return loss / T;
}

function batchLoss(model, bx, by, B, SL) {
  const V = model.cfg.vocab_size;
  let total = 0;
  for (let b = 0; b < B; b++) {
    const x = bx.subarray(b * SL, (b + 1) * SL);
    const y = by.subarray(b * SL, (b + 1) * SL);
    const logits = forwardAll(model, x);
    total += crossEntropy(logits, y, SL, V);
  }
  return total / B;
}

// ============================================================
// BATCH SAMPLING
// ============================================================

function sampleBatch(tokens, B, SL, rng) {
  const n = tokens.length;
  const bx = new Uint32Array(B * SL);
  const by = new Uint32Array(B * SL);
  const maxStart = Math.max(1, n - SL);
  for (let b = 0; b < B; b++) {
    const start = Math.floor(rng() * maxStart);
    for (let t = 0; t < SL; t++) {
      bx[b * SL + t] = tokens[(start + t) % n];
      by[b * SL + t] = tokens[(start + t + 1) % n];
    }
  }
  return { bx, by };
}

// ============================================================
// EGGROLL  (evolution strategies, forward-only)
// ============================================================

function eggrollStep(model, bx, by, cfg, step, stage) {
  const { pop_size: P, sigma, lr, batch_size: B, seq_len: SL, seed } = cfg;
  const tensors = modelTensors(model);
  const rawFit  = new Float32Array(P); // fitness = -loss

  // Evaluate population (antithetic pairs)
  for (let p = 0; p < P / 2; p++) {
    for (let sign = 1; sign >= -1; sign -= 2) {
      const popIdx = p * 2 + (sign === 1 ? 0 : 1);

      // Apply perturbation in-place
      for (let tid = 0; tid < tensors.length; tid++) {
        const W = tensors[tid];
        for (let j = 0; j < W.length; j++) W[j] += sign * sigma * rngEps(seed, stage, step, p, tid, j);
      }

      rawFit[popIdx] = -batchLoss(model, bx, by, B, SL);

      // Restore
      for (let tid = 0; tid < tensors.length; tid++) {
        const W = tensors[tid];
        for (let j = 0; j < W.length; j++) W[j] -= sign * sigma * rngEps(seed, stage, step, p, tid, j);
      }
    }
  }

  // Z-score normalize fitnesses
  let mean = 0;
  for (let i = 0; i < P; i++) mean += rawFit[i];
  mean /= P;
  let std = 0;
  for (let i = 0; i < P; i++) std += (rawFit[i] - mean) ** 2;
  std = Math.sqrt(std / P);
  const norm = new Float32Array(P);
  for (let i = 0; i < P; i++) norm[i] = (rawFit[i] - mean) / (std + 1e-8);

  // Weight update (regenerate eps — same RNG key → same values)
  for (let tid = 0; tid < tensors.length; tid++) {
    const W = tensors[tid];
    for (let j = 0; j < W.length; j++) {
      let g = 0;
      for (let i = 0; i < P; i++) {
        const p = Math.floor(i / 2);
        const s = (i % 2 === 0) ? 1 : -1;
        g += norm[i] * s * rngEps(seed, stage, step, p, tid, j);
      }
      W[j] += lr * g / (P * sigma);
    }
  }

  return rawFit; // caller computes train_loss estimate
}

// ============================================================
// SAMPLE GENERATION
// ============================================================

function generateSample(model, seedText, maxTokens, rng, temperature = 0.8, bpe = null) {
  const V   = model.cfg.vocab_size;
  const ctx = model.cfg.ctx_len;

  // Encode seed
  const buf = new Uint32Array(ctx);
  let len = 0;
  if (bpe) {
    const enc = bpeEncode(seedText, bpe.merges);
    for (let i = 0; i < enc.length && len < ctx; i++) buf[len++] = enc[i];
  } else {
    for (let i = 0; i < seedText.length && len < ctx; i++) buf[len++] = seedText.charCodeAt(i) & 0xff;
  }
  if (len === 0) buf[len++] = 10; // fallback: newline

  const out = [];
  for (let _ = 0; _ < maxTokens; _++) {
    const ctxSlice = buf.subarray(0, len);
    const logits   = forward(model, ctxSlice);

    let tokenId;
    if (temperature === 0) {
      let best = 0;
      for (let v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
      tokenId = best;
    } else {
      // Temperature-scaled softmax sampling
      let mx = -Infinity;
      for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
      let sum = 0;
      const probs = new Float32Array(V);
      for (let v = 0; v < V; v++) { probs[v] = Math.exp((logits[v] - mx) / temperature); sum += probs[v]; }
      const u = rng() * sum;
      let cum = 0;
      tokenId = V - 1;
      for (let v = 0; v < V; v++) { cum += probs[v]; if (u < cum) { tokenId = v; break; } }
    }

    out.push(tokenId);
    // Slide context window
    if (len < ctx) buf[len++] = tokenId;
    else {
      for (let i = 0; i < ctx - 1; i++) buf[i] = buf[i + 1];
      buf[ctx - 1] = tokenId;
    }
  }

  if (bpe) return bpeDecode(out, bpe.vocab);
  try { return new TextDecoder('utf-8', { fatal: false }).decode(new Uint8Array(out)); }
  catch { return out.map(c => String.fromCharCode(c)).join(''); }
}

// ============================================================
// CORPUS HANDLING
// ============================================================

function tokenizeBytes(text) {
  const enc = new TextEncoder();
  const bytes = enc.encode(text);
  const tokens = new Uint32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) tokens[i] = bytes[i];
  return tokens;
}

// ============================================================
// BPE TOKENIZER  (Stage 3 — hand-rolled, no external deps)
// ============================================================

function trainBPE(text, targetVocabSize) {
  // Initialise vocab with all 256 bytes
  const vocab = [];
  for (let i = 0; i < 256; i++) vocab.push(String.fromCharCode(i));

  // Byte-encode the full corpus
  const bytes = new TextEncoder().encode(text);
  let tokens  = Array.from(bytes);

  const merges    = [];
  const numMerges = targetVocabSize - 256;

  for (let m = 0; m < numMerges; m++) {
    // Count all consecutive pairs
    const counts = new Map();
    for (let i = 0; i < tokens.length - 1; i++) {
      const k = tokens[i] * 65536 + tokens[i + 1];
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    if (counts.size === 0) break;

    // Best pair by frequency (ties broken by key value for determinism)
    let bestKey = -1, bestCnt = 1; // require ≥2 occurrences
    for (const [k, v] of counts) {
      if (v > bestCnt || (v === bestCnt && (bestKey === -1 || k < bestKey))) {
        bestCnt = v; bestKey = k;
      }
    }
    if (bestKey === -1) break; // nothing appears ≥2 times

    const a = (bestKey / 65536) | 0;
    const b =  bestKey % 65536;
    const newId = 256 + m;
    merges.push([a, b]);
    vocab.push(vocab[a] + vocab[b]);

    // Apply merge (single left-to-right pass)
    const next = [];
    let i = 0;
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
        next.push(newId); i += 2;
      } else {
        next.push(tokens[i]); i++;
      }
    }
    tokens = next;
  }

  return { merges, vocab };
}

// Encode a string to BPE token ids (apply merges in training order)
function bpeEncode(text, merges) {
  const bytes = new TextEncoder().encode(text);
  let tokens  = Array.from(bytes);

  for (let i = 0; i < merges.length; i++) {
    const [a, b] = merges[i];
    const newId  = 256 + i;
    const next   = [];
    let j = 0;
    while (j < tokens.length) {
      if (j < tokens.length - 1 && tokens[j] === a && tokens[j + 1] === b) {
        next.push(newId); j += 2;
      } else {
        next.push(tokens[j]); j++;
      }
    }
    tokens = next;
  }

  return new Uint32Array(tokens);
}

function bpeDecode(tokens, vocab) {
  return tokens.map(t => vocab[t] ?? '?').join('');
}

async function hashCorpus(text) {
  try {
    const bytes = new TextEncoder().encode(text);
    const buf   = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(buf).slice(0, 8)).map(b => b.toString(16).padStart(2, '0')).join('');
  } catch {
    // FNV-1a fallback (Node.js testing)
    let h = 2166136261 >>> 0;
    for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    return h.toString(16).padStart(8, '0');
  }
}

// ============================================================
// MODEL SERIALIZATION
// ============================================================

function serializeModel(m) {
  return {
    config: m.cfg,
    embed:     Array.from(m.embed),
    pos_embed: Array.from(m.pos_embed),
    blocks: m.blocks.map(b => ({
      wq: Array.from(b.wq), wk: Array.from(b.wk), wv: Array.from(b.wv), wo: Array.from(b.wo),
      ff1: Array.from(b.ff1), ff2: Array.from(b.ff2),
      ln1_g: Array.from(b.ln1_g), ln1_b: Array.from(b.ln1_b),
      ln2_g: Array.from(b.ln2_g), ln2_b: Array.from(b.ln2_b)
    })),
    ln_f_g: Array.from(m.ln_f_g), ln_f_b: Array.from(m.ln_f_b)
  };
}

function deserializeModel(d) {
  return {
    cfg:       d.config,
    embed:     new Float32Array(d.embed),
    pos_embed: new Float32Array(d.pos_embed),
    blocks: d.blocks.map(b => ({
      wq: new Float32Array(b.wq), wk: new Float32Array(b.wk),
      wv: new Float32Array(b.wv), wo: new Float32Array(b.wo),
      ff1: new Float32Array(b.ff1), ff2: new Float32Array(b.ff2),
      ln1_g: new Float32Array(b.ln1_g), ln1_b: new Float32Array(b.ln1_b),
      ln2_g: new Float32Array(b.ln2_g), ln2_b: new Float32Array(b.ln2_b)
    })),
    ln_f_g: new Float32Array(d.ln_f_g), ln_f_b: new Float32Array(d.ln_f_b)
  };
}

// ============================================================
// OPFS CHECKPOINT
// ============================================================

async function opfsDir() {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle('tapasya', { create: true });
}

async function saveCheckpointOPFS(stage, st) {
  try {
    const dir = await opfsDir();
    const fh  = await dir.getFileHandle(`stage-${stage}.json`, { create: true });
    const w   = await fh.createWritable();
    const payload = {
      stage, step: st.step,
      model: serializeModel(st.model),
      config: st.cfg,
      history: st.history,
      sample_history: st.sampleHistory,
      rng_state: 0, // counter-based RNG needs no state
      corpus_hash: st.corpusHash,
      bpe: st.bpe || null,
    };
    await w.write(JSON.stringify(payload));
    await w.close();
  } catch (e) {
    console.warn('[tapasya] OPFS save failed:', e);
  }
}

async function loadCheckpointOPFS(stage) {
  try {
    const dir = await opfsDir();
    const fh  = await dir.getFileHandle(`stage-${stage}.json`);
    const f   = await fh.getFile();
    return JSON.parse(await f.text());
  } catch { return null; }
}

async function deleteCheckpointOPFS(stage) {
  try {
    const dir = await opfsDir();
    await dir.removeEntry(`stage-${stage}.json`);
  } catch {}
}

// ============================================================
// PER-STAGE STATE  (worker owns all three)
// ============================================================

const DEFAULT_CFG = {
  pop_size:   16,
  sigma:      0.05,
  lr:         0.003,
  batch_size: 8,
  seq_len:    128,
  seed:       42,
  val_split:  0.1
};

const STAGE_STEPS = { 1: 500, 2: 2000, 3: 2000 };

function makeModelCfg(stage) {
  return { vocab_size: stage === 3 ? 2000 : 256, d_model: 64, n_heads: 4, n_layers: 4, ctx_len: 128, d_ff: 256 };
}

// stages[1|2|3] → { model, trainTokens, valTokens, cfg, step, running, history, sampleHistory, corpusHash }
const stages = {};

// Active training loop handle
let activeStage = null;
let pauseFlag   = false;

// ============================================================
// GPU BACKEND  (WebGPU accelerated forward pass — CPU fallback if unavailable)
// ============================================================

let gpuCtx       = undefined; // undefined=not tried, null=failed/no WebGPU, obj=ready
let gpuModelBufs = null;      // model weights uploaded to GPU for the active stage

async function initWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device  = await adapter.requestDevice();

    const resp = await fetch('./shaders.wgsl').catch(() => null);
    if (!resp?.ok) return null;
    const mod = device.createShaderModule({ code: await resp.text() });
    const inf  = await mod.getCompilationInfo();
    if (inf.messages.some(m => m.type === 'error')) {
      console.error('[GPU] shader errors:', inf.messages.filter(m => m.type === 'error'));
      return null;
    }

    const bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]});
    const plo  = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const mkPL = ep => device.createComputePipeline({
      layout: plo, compute: { module: mod, entryPoint: ep }
    });
    const pl = {
      embed_add:    mkPL('embed_add'),
      matmul:       mkPL('matmul'),
      matmul_bt:    mkPL('matmul_bt'),
      layer_norm:   mkPL('layer_norm'),
      residual_add: mkPL('residual_add'),
      gelu_inplace: mkPL('gelu_inplace'),
      causal_attn:  mkPL('causal_attn'),
      cross_entropy: mkPL('cross_entropy'),
    };

    const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const df = device.createBuffer({ size: 4, usage: SC });
    const du = device.createBuffer({ size: 4, usage: SC });

    // Activation buffers sized for worst-case (Stage 3: V=2000, ctx=128, D=64, F=256)
    const mkA = n => device.createBuffer({ size: Math.max(4, n * 4), usage: SC });
    const act = {
      x0: mkA(128*64), x1: mkA(128*64), norm: mkA(128*64),
      Q:  mkA(128*64), K:  mkA(128*64), Vb:  mkA(128*64),
      attn: mkA(128*64), proj: mkA(128*64), ff_h: mkA(128*256),
      logits: mkA(128*2000),
      perTokenLoss: mkA(128),
      lossStage: device.createBuffer({
        size: 128 * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    };

    device.lost.then(() => { gpuCtx = null; gpuModelBufs = null; });
    post({ type: 'gpu_ready' });
    return { device, bgl, pl, df, du, act };
  } catch (e) {
    console.warn('[GPU] init failed:', e.message);
    return null;
  }
}

function gpuUploadModel(device, model) {
  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const up = a => {
    const b = device.createBuffer({ size: Math.max(4, a.byteLength), usage: SC, mappedAtCreation: true });
    new Float32Array(b.getMappedRange()).set(a);
    b.unmap();
    return b;
  };
  return {
    embed: up(model.embed), pos_embed: up(model.pos_embed),
    blocks: model.blocks.map(b => ({
      wq: up(b.wq), wk: up(b.wk), wv: up(b.wv), wo: up(b.wo),
      ff1: up(b.ff1), ff2: up(b.ff2),
      ln1_g: up(b.ln1_g), ln1_b: up(b.ln1_b),
      ln2_g: up(b.ln2_g), ln2_b: up(b.ln2_b),
    })),
    ln_f_g: up(model.ln_f_g), ln_f_b: up(model.ln_f_b),
  };
}

function gpuSyncModel(device, gpuM, model) {
  const wr = (buf, data) => device.queue.writeBuffer(buf, 0, data);
  wr(gpuM.embed, model.embed); wr(gpuM.pos_embed, model.pos_embed);
  for (let l = 0; l < model.blocks.length; l++) {
    const b = model.blocks[l], g = gpuM.blocks[l];
    wr(g.wq, b.wq); wr(g.wk, b.wk); wr(g.wv, b.wv); wr(g.wo, b.wo);
    wr(g.ff1, b.ff1); wr(g.ff2, b.ff2);
    wr(g.ln1_g, b.ln1_g); wr(g.ln1_b, b.ln1_b);
    wr(g.ln2_g, b.ln2_g); wr(g.ln2_b, b.ln2_b);
  }
  wr(gpuM.ln_f_g, model.ln_f_g); wr(gpuM.ln_f_b, model.ln_f_b);
}

// Records one compute dispatch into enc (creates a small per-dispatch uniform buffer)
function gpuPass(enc, device, bgl, df, du, pl, cfgArr, a, b, c, out, u32, dx, dy = 1) {
  const cfgBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(cfgBuf.getMappedRange()).set(
    [cfgArr[0] || 0, cfgArr[1] || 0, cfgArr[2] || 0, cfgArr[3] || 0]
  );
  cfgBuf.unmap();
  const bg = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: cfgBuf } },
      { binding: 1, resource: { buffer: a   || df } },
      { binding: 2, resource: { buffer: b   || df } },
      { binding: 3, resource: { buffer: c   || df } },
      { binding: 4, resource: { buffer: out        } },
      { binding: 5, resource: { buffer: u32 || du  } },
    ]
  });
  const pass = enc.beginComputePass();
  pass.setPipeline(pl);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(dx, dy);
  pass.end();
}

// Full forward pass + cross-entropy on GPU; reads back only T floats (512 bytes).
async function forwardLossGPU(model, tokens, targets) {
  const { device, bgl, pl: PL, act, df, du } = gpuCtx;
  const gpuM = gpuModelBufs;
  const T    = tokens.length;
  const { vocab_size: V, d_model: D, n_heads: H, n_layers: L, d_ff: F } = model.cfg;

  const SC = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const mkU32 = arr => {
    const b = device.createBuffer({ size: Math.max(4, arr.length * 4), usage: SC, mappedAtCreation: true });
    new Uint32Array(b.getMappedRange()).set(arr);
    b.unmap();
    return b;
  };
  const tokBuf = mkU32(tokens);
  const tgtBuf = mkU32(targets);

  const enc = device.createCommandEncoder();
  const c8  = n => Math.ceil(n / 8);
  const c64 = n => Math.ceil(n / 64);
  const go  = (pl, cfg, a, b, c, out, u32, dx, dy = 1) =>
    gpuPass(enc, device, bgl, df, du, pl, cfg, a, b, c, out, u32, dx, dy);

  let curX = act.x0, tmpX = act.x1;
  go(PL.embed_add, [T, D], gpuM.embed, gpuM.pos_embed, null, curX, tokBuf, c64(T * D));
  for (let l = 0; l < L; l++) {
    const blk = gpuM.blocks[l];
    go(PL.layer_norm,   [T, D],    curX,     blk.ln1_g, blk.ln1_b, act.norm, null, T);
    go(PL.matmul,       [T, D, D], act.norm, blk.wq,    null,      act.Q,    null, c8(D), c8(T));
    go(PL.matmul,       [T, D, D], act.norm, blk.wk,    null,      act.K,    null, c8(D), c8(T));
    go(PL.matmul,       [T, D, D], act.norm, blk.wv,    null,      act.Vb,   null, c8(D), c8(T));
    go(PL.causal_attn,  [T, D, H], act.Q,    act.K,     act.Vb,    act.attn, null, 1,     T);
    go(PL.matmul,       [T, D, D], act.attn, blk.wo,    null,      act.proj, null, c8(D), c8(T));
    go(PL.residual_add, [T * D],   curX,     act.proj,  null,      tmpX,     null, c64(T * D));
    [curX, tmpX] = [tmpX, curX];
    go(PL.layer_norm,   [T, D],    curX,     blk.ln2_g, blk.ln2_b, act.norm,  null, T);
    go(PL.matmul,       [T, D, F], act.norm, blk.ff1,   null,      act.ff_h,  null, c8(F), c8(T));
    go(PL.gelu_inplace, [T * F],   null,     null,      null,      act.ff_h,  null, c64(T * F));
    go(PL.matmul,       [T, F, D], act.ff_h, blk.ff2,   null,      act.proj,  null, c8(D), c8(T));
    go(PL.residual_add, [T * D],   curX,     act.proj,  null,      tmpX,      null, c64(T * D));
    [curX, tmpX] = [tmpX, curX];
  }
  go(PL.layer_norm,    [T, D],    curX,      gpuM.ln_f_g, gpuM.ln_f_b, act.norm,         null,   T);
  go(PL.matmul_bt,     [T, D, V], act.norm,  gpuM.embed,  null,        act.logits,        null,   c8(V), c8(T));
  go(PL.cross_entropy, [T, V],    act.logits, null,        null,        act.perTokenLoss,  tgtBuf, T);

  enc.copyBufferToBuffer(act.perTokenLoss, 0, act.lossStage, 0, T * 4);
  device.queue.submit([enc.finish()]);
  tokBuf.destroy();
  tgtBuf.destroy();

  await act.lossStage.mapAsync(GPUMapMode.READ, 0, T * 4);
  const losses = new Float32Array(act.lossStage.getMappedRange(0, T * 4).slice(0));
  act.lossStage.unmap();

  let sum = 0;
  for (const l of losses) sum += l;
  return sum / T;
}

async function batchLossGPU(model, bx, by, B, SL) {
  let total = 0;
  for (let b = 0; b < B; b++) {
    total += await forwardLossGPU(
      model,
      bx.subarray(b * SL, (b + 1) * SL),
      by.subarray(b * SL, (b + 1) * SL),
    );
  }
  return total / B;
}

// EGGROLL with GPU fitness evaluation; weight math stays on CPU
async function eggrollStepGPU(model, bx, by, cfg, step, stage) {
  const { pop_size: P, sigma, lr, batch_size: B, seq_len: SL, seed } = cfg;
  const tensors = modelTensors(model);
  const rawFit  = new Float32Array(P);

  for (let p = 0; p < P / 2; p++) {
    for (let sign = 1; sign >= -1; sign -= 2) {
      const popIdx = p * 2 + (sign === 1 ? 0 : 1);
      for (let tid = 0; tid < tensors.length; tid++) {
        const W = tensors[tid];
        for (let j = 0; j < W.length; j++) W[j] += sign * sigma * rngEps(seed, stage, step, p, tid, j);
      }
      gpuSyncModel(gpuCtx.device, gpuModelBufs, model);
      rawFit[popIdx] = -(await batchLossGPU(model, bx, by, B, SL));
      for (let tid = 0; tid < tensors.length; tid++) {
        const W = tensors[tid];
        for (let j = 0; j < W.length; j++) W[j] -= sign * sigma * rngEps(seed, stage, step, p, tid, j);
      }
    }
  }

  let mean = 0; for (const f of rawFit) mean += f; mean /= P;
  let std  = 0; for (const f of rawFit) std  += (f - mean) ** 2; std = Math.sqrt(std / P);
  const norm = rawFit.map(f => (f - mean) / (std + 1e-8));

  for (let tid = 0; tid < tensors.length; tid++) {
    const W = tensors[tid];
    for (let j = 0; j < W.length; j++) {
      let g = 0;
      for (let i = 0; i < P; i++) {
        const pp = Math.floor(i / 2), s = (i % 2 === 0) ? 1 : -1;
        g += norm[i] * s * rngEps(seed, stage, step, pp, tid, j);
      }
      W[j] += lr * g / (P * sigma);
    }
  }
  gpuSyncModel(gpuCtx.device, gpuModelBufs, model);
  return rawFit;
}

// ============================================================
// TRAINING LOOP
// ============================================================

async function runLoop(stageIdx) {
  const s    = stages[stageIdx];
  const { cfg, model, trainTokens, valTokens } = s;
  const { steps, batch_size: B, seq_len: SL } = cfg;
  const T0 = Date.now();

  while (!pauseFlag && s.step < steps) {
    s.step++;
    const wallStart = Date.now();

    // Sample training batch
    const trainRng    = makeLcg(cfg.seed, stageIdx, s.step);
    const { bx, by } = sampleBatch(trainTokens, B, SL, trainRng);

    // EGGROLL update (GPU when available, CPU fallback)
    const rawFit = gpuCtx
      ? await eggrollStepGPU(model, bx, by, cfg, s.step, stageIdx)
      : eggrollStep(model, bx, by, cfg, s.step, stageIdx);
    const trainLoss = -rawFit.reduce((a, b) => a + b, 0) / rawFit.length;

    // Report every 10 steps
    if (s.step % 10 === 0) {
      const valSL  = Math.min(SL, valTokens.length - 1);
      let valLoss  = NaN;
      if (valSL > 1) {
        const valRng    = makeLcg(cfg.seed + 1, stageIdx, s.step);
        const { bx: vx, by: vy } = sampleBatch(valTokens, Math.min(2, Math.floor(valTokens.length / valSL)), valSL, valRng);
        valLoss = batchLoss(model, vx, vy, Math.min(2, Math.floor(valTokens.length / valSL)), valSL);
      }

      const entry = { step: s.step, train_loss: trainLoss, val_loss: valLoss, wall_ms: Date.now() - T0 };
      s.history.push(entry);

      post({ type: 'progress', stage: stageIdx, step: s.step,
             train_loss: trainLoss, val_loss: valLoss, wall_ms: Date.now() - T0 });

      // Sample
      const genRng = makeLcg(cfg.seed + 99, stageIdx, s.step);
      const seed_text = s.bpe
        ? bpeDecode(Array.from(trainTokens.slice(0, 10)), s.bpe.vocab)
        : String.fromCharCode(...Array.from(trainTokens.slice(0, 10)));
      const text = generateSample(model, seed_text, 100, genRng, 0.8, s.bpe);
      const sEntry = { step: s.step, text };
      s.sampleHistory.push(sEntry);
      if (s.sampleHistory.length > 20) s.sampleHistory.shift();
      post({ type: 'sample', stage: stageIdx, step: s.step, text });
    }

    // Checkpoint every 50 steps
    if (s.step % 50 === 0) {
      await saveCheckpointOPFS(stageIdx, s);
      post({ type: 'checkpoint_saved', stage: stageIdx, step: s.step });
    }

    // Guided-mode unlock notifications
    if (stageIdx === 1 && s.step === 300) post({ type: 'unlock', unlocks: 2 });
    if (stageIdx === 2 && s.step === 1000) post({ type: 'unlock', unlocks: 3 });

    // Yield to prevent blocking the worker event loop
    await new Promise(r => setTimeout(r, 0));
  }

  if (pauseFlag) {
    pauseFlag = false;
    activeStage = null;
    post({ type: 'paused', stage: stageIdx, step: s.step });
    return;
  }

  // Done
  s.running = false;
  activeStage = null;
  const modelJson = serializeModel(model);
  await saveCheckpointOPFS(stageIdx, s);
  post({ type: 'done', stage: stageIdx, step: s.step, model_json: modelJson });

  // Compare-mode unlock if ≥2 stages done
  const doneCt = [1, 2, 3].filter(i => stages[i] && !stages[i].running && stages[i].step >= (stages[i].cfg.steps || STAGE_STEPS[i])).length;
  if (doneCt >= 2) post({ type: 'unlock', unlocks: 'compare' });
}

// ============================================================
// MESSAGE HANDLERS
// ============================================================

async function handleInit(msg) {
  const { stage, config, corpus, seed } = msg;

  const cfg       = { ...DEFAULT_CFG, steps: STAGE_STEPS[stage], ...config, seed: seed || DEFAULT_CFG.seed };
  const corpusHash = await hashCorpus(corpus);

  // Check for existing checkpoint
  const ckpt = await loadCheckpointOPFS(stage);
  if (ckpt && ckpt.corpus_hash === corpusHash && ckpt.step > 0) {
    post({ type: 'checkpoint_found', stage, step: ckpt.step, corpus_hash: corpusHash });
    // Store pending resume data
    stages[`pending_${stage}`] = { ckpt, corpus, cfg };
    return;
  }

  // Initialise WebGPU once per worker lifetime
  if (gpuCtx === undefined) gpuCtx = await initWebGPU();

  // Fresh init
  await initStage(stage, corpus, cfg, corpusHash);
  post({ type: 'ready', stage });
}

async function initStage(stage, corpus, cfg, corpusHash) {
  const splitIdx    = Math.floor(corpus.length * (1 - cfg.val_split));
  const trainCorpus = corpus.slice(0, splitIdx);
  const valCorpus   = corpus.slice(splitIdx);

  let bpe = null, trainTokens, valTokens;
  if (stage === 3) {
    post({ type: 'status', stage, message: 'Training BPE tokenizer…' });
    bpe         = trainBPE(corpus, makeModelCfg(3).vocab_size);
    trainTokens = bpeEncode(trainCorpus, bpe.merges);
    valTokens   = bpeEncode(valCorpus,   bpe.merges);
  } else {
    trainTokens = tokenizeBytes(trainCorpus);
    valTokens   = tokenizeBytes(valCorpus);
  }

  const modelCfg = makeModelCfg(stage);
  const model    = initModel(modelCfg, cfg.seed);

  stages[stage] = {
    model, trainTokens, valTokens, bpe,
    cfg: { ...cfg, steps: cfg.steps || STAGE_STEPS[stage] },
    step: 0, running: false,
    history: [], sampleHistory: [],
    corpusHash
  };

  if (gpuCtx) gpuModelBufs = gpuUploadModel(gpuCtx.device, model);
}

async function handleResume(msg) {
  const { stage } = msg;
  const pending = stages[`pending_${stage}`];
  if (!pending) { post({ type: 'error', stage, message: 'No pending resume.' }); return; }

  const { ckpt, corpus, cfg } = pending;
  delete stages[`pending_${stage}`];

  const corpusHash = await hashCorpus(corpus);
  const model      = deserializeModel(ckpt.model);
  const splitIdx   = Math.floor(corpus.length * (1 - cfg.val_split));

  const bpe = ckpt.bpe || null;
  let trainTokens, valTokens;
  if (stage === 3 && bpe) {
    trainTokens = bpeEncode(corpus.slice(0, splitIdx), bpe.merges);
    valTokens   = bpeEncode(corpus.slice(splitIdx),   bpe.merges);
  } else {
    trainTokens = tokenizeBytes(corpus.slice(0, splitIdx));
    valTokens   = tokenizeBytes(corpus.slice(splitIdx));
  }

  stages[stage] = {
    model, trainTokens, valTokens, bpe,
    cfg: { ...DEFAULT_CFG, ...ckpt.config, steps: cfg.steps || STAGE_STEPS[stage] },
    step: ckpt.step, running: false,
    history: ckpt.history || [], sampleHistory: ckpt.sample_history || [],
    corpusHash
  };

  if (gpuCtx) gpuModelBufs = gpuUploadModel(gpuCtx.device, model);
  post({ type: 'ready', stage, step: ckpt.step });
}

async function handleStart(msg) {
  const { stage } = msg;
  const s = stages[stage];
  if (!s) { post({ type: 'error', stage, message: 'Stage not initialised. Send init first.' }); return; }
  if (activeStage !== null && activeStage !== stage) {
    post({ type: 'error', stage, message: `Stage ${activeStage} is already running.` }); return;
  }

  pauseFlag   = false;
  s.running   = true;
  activeStage = stage;
  runLoop(stage); // fire-and-forget
}

function handlePause(msg) {
  pauseFlag = true;
}

async function handleReset(msg) {
  const { stage } = msg;
  if (activeStage === stage) { pauseFlag = true; }
  delete stages[stage];
  await deleteCheckpointOPFS(stage);
  post({ type: 'reset_done', stage });
}

async function handleGenerate(msg) {
  const { stage, seed_text, max_tokens } = msg;
  const s = stages[stage];
  if (!s) { post({ type: 'generated', stage, text: '(model not trained)' }); return; }
  const rng  = makeLcg(s.cfg.seed + 77, stage, s.step + 1000);
  const text = generateSample(s.model, seed_text || '\n', max_tokens || 200, rng, 0.8, s.bpe || null);
  post({ type: 'generated', stage, text });
}

// ============================================================
// WORKER MESSAGE ENTRY POINT
// ============================================================

function post(msg) {
  if (typeof self !== 'undefined') self.postMessage(msg);
}

if (typeof self !== 'undefined') {
  self.addEventListener('message', async (e) => {
    try {
      const m = e.data;
      if      (m.type === 'init')     await handleInit(m);
      else if (m.type === 'resume')   await handleResume(m);
      else if (m.type === 'start')    await handleStart(m);
      else if (m.type === 'pause')    handlePause(m);
      else if (m.type === 'reset')    await handleReset(m);
      else if (m.type === 'generate') await handleGenerate(m);
      else if (m.type === 'train_bpe') {
        const bpe = trainBPE(m.corpus, m.targetVocab || 2000);
        post({ type: 'bpe_ready', vocab_size: bpe.vocab.length });
      }
    } catch (err) {
      self.postMessage({ type: 'error', stage: e.data?.stage, message: err.message });
    }
  });
}

// ============================================================
// NODE.JS TEST EXPORTS
// ============================================================
if (typeof module !== 'undefined') {
  module.exports = {
    initModel, makeModelCfg, forward, forwardAll, crossEntropy, batchLoss,
    sampleBatch, eggrollStep, generateSample, tokenizeBytes, hashCorpus,
    makeLcg, STAGE_STEPS, DEFAULT_CFG
  };
}

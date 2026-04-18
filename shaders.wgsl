// Tapasya — WebGPU compute shaders
// Phase 2: forward pass kernels (matmul, attention, LayerNorm, GELU, embed, loss)
//
// All entry points share one bind group layout (group 0, bindings 0-5).
// The JS side always provides all 6 bindings; unused slots get a 4-byte dummy buffer.
//
//   binding 0 — uniform Cfg { x, y, z, w: u32 }  (dimensions, meaning varies per op)
//   binding 1 — storage read  f32_a               (primary float input)
//   binding 2 — storage read  f32_b               (second float input)
//   binding 3 — storage read  f32_c               (third float input)
//   binding 4 — storage r/w   f32_out             (output, also used for in-place ops)
//   binding 5 — storage read  u32_a               (u32 input: token ids / loss targets)

struct Cfg { x: u32, y: u32, z: u32, w: u32 }

@group(0) @binding(0) var<uniform>             cfg:     Cfg;
@group(0) @binding(1) var<storage, read>       f32_a:   array<f32>;
@group(0) @binding(2) var<storage, read>       f32_b:   array<f32>;
@group(0) @binding(3) var<storage, read>       f32_c:   array<f32>;
@group(0) @binding(4) var<storage, read_write> f32_out: array<f32>;
@group(0) @binding(5) var<storage, read>       u32_a:   array<u32>;

// ── 1. embed_add ─────────────────────────────────────────────────
// out[T,D] = embed[token[t], D] + pos_embed[t, D]
// cfg: x=T  y=D
// u32_a=tokens[T]  f32_a=embed[V*D]  f32_b=pos_embed[ctx*D]  f32_out=x[T*D]
@compute @workgroup_size(64)
fn embed_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let T = cfg.x; let D = cfg.y;
  if (i >= T * D) { return; }
  let t   = i / D;
  let d   = i % D;
  let tok = u32_a[t];
  f32_out[i] = f32_a[tok * D + d] + f32_b[t * D + d];
}

// ── 2. matmul ────────────────────────────────────────────────────
// out[M,N] = A[M,K] × B[K,N]
// cfg: x=M  y=K  z=N
// f32_a=A  f32_b=B  f32_out=C
@compute @workgroup_size(8, 8)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x; let m = gid.y;
  let M = cfg.x; let K = cfg.y; let N = cfg.z;
  if (m >= M || n >= N) { return; }
  var acc = 0.0f;
  for (var k = 0u; k < K; k++) {
    acc += f32_a[m * K + k] * f32_b[k * N + n];
  }
  f32_out[m * N + n] = acc;
}

// ── 3. matmul_bt ─────────────────────────────────────────────────
// out[M,N] = A[M,K] × B^T  where B is stored [N,K]
// Used for weight-tied logits: finalH[T,D] × embed^T  (embed stored [V,D])
// cfg: x=M  y=K  z=N
// f32_a=A[M*K]  f32_b=B[N*K]  f32_out=C[M*N]
@compute @workgroup_size(8, 8)
fn matmul_bt(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x; let m = gid.y;
  let M = cfg.x; let K = cfg.y; let N = cfg.z;
  if (m >= M || n >= N) { return; }
  var acc = 0.0f;
  for (var k = 0u; k < K; k++) {
    acc += f32_a[m * K + k] * f32_b[n * K + k];
  }
  f32_out[m * N + n] = acc;
}

// ── 4. layer_norm ─────────────────────────────────────────────────
// out[T,D] = LayerNorm(X[T,D], gain[D], bias[D])
// cfg: x=T  y=D
// f32_a=X[T*D]  f32_b=gain[D]  f32_c=bias[D]  f32_out=Y[T*D]
@compute @workgroup_size(1)
fn layer_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let T = cfg.x; let D = cfg.y;
  if (t >= T) { return; }
  let off = t * D;
  var mean = 0.0f;
  for (var d = 0u; d < D; d++) { mean += f32_a[off + d]; }
  mean /= f32(D);
  var v = 0.0f;
  for (var d = 0u; d < D; d++) {
    let diff = f32_a[off + d] - mean;
    v += diff * diff;
  }
  let inv_std = inverseSqrt(v / f32(D) + 1e-5);
  for (var d = 0u; d < D; d++) {
    f32_out[off + d] = f32_b[d] * (f32_a[off + d] - mean) * inv_std + f32_c[d];
  }
}

// ── 5. residual_add ───────────────────────────────────────────────
// out[N] = A[N] + B[N]
// cfg: x=N
// f32_a=A  f32_b=B  f32_out=C
@compute @workgroup_size(64)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.x) { return; }
  f32_out[i] = f32_a[i] + f32_b[i];
}

// ── 6. gelu_inplace ───────────────────────────────────────────────
// f32_out[N] = GELU(f32_out[N])  (read-modify-write)
// cfg: x=N
const SQRT_2_PI = 0.7978845608f;
@compute @workgroup_size(64)
fn gelu_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.x) { return; }
  let x = f32_out[i];
  f32_out[i] = 0.5f * x * (1.0f + tanh(SQRT_2_PI * (x + 0.044715f * x * x * x)));
}

// ── 7. causal_attn ────────────────────────────────────────────────
// out[T,D] = causal multi-head attention(Q[T,D], K[T,D], V[T,D])
// cfg: x=T  y=D  z=H
// f32_a=Q  f32_b=K  f32_c=V  f32_out=attn_out
// Dispatch: (1, T) workgroups with workgroup_size(H, 1)
// → gid.x=head  gid.y=query_pos
@compute @workgroup_size(4, 1)
fn causal_attn(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let i = gid.y;
  let T = cfg.x; let D = cfg.y; let H = cfg.z;
  if (h >= H || i >= T) { return; }
  let dh    = D / H;
  let scale = 1.0f / sqrt(f32(dh));
  let h_off = h * dh;

  // Compute attention scores for all past/current positions (causal mask)
  var scores: array<f32, 128>;   // 128 = ctx_len (compile-time constant)
  var mx = -1e9f;
  for (var j = 0u; j <= i; j++) {
    var s = 0.0f;
    for (var d = 0u; d < dh; d++) {
      s += f32_a[i * D + h_off + d] * f32_b[j * D + h_off + d];
    }
    s *= scale;
    scores[j] = s;
    if (s > mx) { mx = s; }
  }

  // Numerically stable softmax
  var sum_exp = 0.0f;
  for (var j = 0u; j <= i; j++) {
    scores[j] = exp(scores[j] - mx);
    sum_exp  += scores[j];
  }
  for (var j = 0u; j <= i; j++) { scores[j] /= sum_exp; }

  // Weighted sum of V
  for (var d = 0u; d < dh; d++) {
    var val = 0.0f;
    for (var j = 0u; j <= i; j++) {
      val += scores[j] * f32_c[j * D + h_off + d];
    }
    f32_out[i * D + h_off + d] = val;
  }
}

// ── 8. cross_entropy ─────────────────────────────────────────────
// out[T] = per-position cross-entropy loss
// cfg: x=T  y=V
// f32_a=logits[T*V]  u32_a=targets[T]  f32_out=losses[T]
@compute @workgroup_size(1)
fn cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let V = cfg.y;
  if (t >= cfg.x) { return; }
  let off = t * V;
  var mx = -1e30f;
  for (var v = 0u; v < V; v++) {
    if (f32_a[off + v] > mx) { mx = f32_a[off + v]; }
  }
  var sum_exp = 0.0f;
  for (var v = 0u; v < V; v++) { sum_exp += exp(f32_a[off + v] - mx); }
  let tgt    = u32_a[t];
  f32_out[t] = -(f32_a[off + tgt] - mx - log(sum_exp));
}

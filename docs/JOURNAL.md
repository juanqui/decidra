# Work journal

Autonomous overnight session. Newest entry first. Numbers are GPU 1 (RTX 4090,
sm_89), cache disabled for unique-prompt benchmarks.

## 2026-10-20 (night)

### Final state (overnight)

Shipped default is now **mixed W4A8-INT8 with per-module scale granularity**:

- per-token INT8 (fast int32 epilogue): `mlp.gate_proj`
- group-64 INT8: `linear_attn.in_proj_z/b/a`, `mlp.up_proj`, `mlp.down_proj`,
  `linear_attn.in_proj_qkv`, `linear_attn.out_proj`, `self_attn.q_proj`, plus all linears
  in layers 56-63 (`last:8`)
- W4A16 outside those layers: `self_attn.o_proj`, `self_attn.v_proj`,
  `self_attn.k_proj`

`JEV_NATIVE_INT8_GROUP_SCALE=true`, `JEV_NATIVE_INT8_GROUP_EXCLUDE=mlp.gate_proj`,
`JEV_NATIVE_INT8_EXCLUDE=self_attn.o_proj,self_attn.v_proj,self_attn.k_proj`.

Quality (`eval/suite.py` vs CPU AWQ golden): **28/28, KL 3.095e-5, MAE 8.642e-4,
max 7.511e-3** — all strict gates pass.

Engine cold/warm single request (cache off): 73 tok **~56 ms mean / 37.6 ms min**,
4K **924 ms** (905 min), 16K **3900 ms** (3876 min).

Server (one resident process, 16 unique prompts/level, cache off):

| level | req/s | p50 | GPU util | power |
|---|---:|---:|---:|---:|
| c1 | 25.08 | 40.2 ms | 97% | 376 W |
| c2 | 38.95 | 51.9 ms | 93% | 349 W |
| c4 | 50.18 | 79.2 ms | 93% | 328 W |
| c8 | 53.20 | 148.4 ms | 96% | 334 W |
| c16 | 53.44 | 240.5 ms | 95% | 353 W |

Long single request: **4K 0.913 s (4,477 tok/s, 100% GPU, 384-428 W)**,
**8K 1.871 s (4,371 tok/s, 100%, 422 W)**,
**16K 4.051 s (4,045 tok/s, 100%, 430 W)**.

Resident/peak memory: **18.83 GiB** short, **20.17 GiB during 16K**, flat across
repeated 16K requests. The 2.37 GiB embedding table is resident and rows are gathered
on-device; the low-memory pool threshold is 12,288 tokens.

LiteLLM verified end-to-end against the live server (non-streaming + SSE streaming,
usage included): `/tmp/litellm-venv`.

Corrected baseline note: the earlier 758 ms 4K number was **all-linear per-token INT8**
(quality-failing). The shipped selective W4A16 baseline is 4K ~1231 ms / 16K ~5174 ms;
the mixed config is **~25% faster** than that baseline while passing the gate.

**Superseded analysis (corrected).** An earlier draft compared the expanded group-64
config against an all-linear per-token run and concluded INT8 expansion did not help. The
correct A/B is against the **W4A16** (all-excluded) footprint. At 4K the same module is
2.1-2.4x faster on INT8:

| 4K per-op | W4A16 | per-token INT8 | group-64 INT8 |
|---|---:|---:|---:|
| up | 3.811 ms | 1.606 ms | 2.380 ms |
| down | 3.808 ms | 1.776 ms | 2.548 ms |
| in_qkv | 2.248 ms | 0.946 ms | 1.415 ms |
| out | 1.365 ms | 0.633 ms | 0.946 ms |
| q | 2.690 ms | 1.121 ms | 1.684 ms |
| gate | — | 1.575 ms | 2.362 ms |

Group scales cost ~35-50% versus the per-token int32 epilogue on the same module, but
still beat W4A16. Keeping only `mlp.gate_proj` on per-token saves ~50 ms at 4K; that is
the shipped default.

### What landed

- **Marlin W4A8 per-K-group INT8 activation scales** (group 64), including the split-K
  group-offset fix (`a8_group_base = slice_row * thread_k_blocks / group_blocks`). The
  first version ignored `slice_row`, which corrupted every split-K partial; row 0 looked
  correct, the suite failed at MAE 0.33. Fixed and verified against W4A16 at 64, 73 and
  4090 rows.
- **INT8 activation scratch reuse**: one `xq`/scales buffer pair per engine instead of two
  `cuMemAllocAsync` + memset per INT8 GEMM (400 GEMMs/forward at short).
- **Marlin launch-path caching**: device attributes and `cudaFuncSetAttribute` are queried
  once per thread instead of every GEMM; `JEV_MARLIN_CHUNK`, `JEV_GEMM_DEBUG*` and
  `JEV_MARLIN_DEBUG` are cached (`LazyLock`) instead of read per GEMM.
- **Resident GPU embedding table** (2.37 GiB): rows are gathered on-device with a
  16-byte-vector kernel, bit-identical to the host gather. 4K 938.8 -> 915.8 ms
  (-2.5%), 16K 4096 -> 4004 ms (-2.2%), short min 40.8 -> 37.2 ms. The host gather
  (parallel, up to 8 threads) remains the fallback when disabled.
- **CLI benchmark parity**: `jev-native layer/batch` now enables chunked+TC GDN by
  default (the server already did); `--legacy-gdn` disables all chunked paths. Earlier CLI
  numbers measured the scalar f32 GDN by mistake.
- **OpenAI-compatible decision routes** for LiteLLM: `/v1/chat/completions` and
  `/v1/completions` parse a JSON decision request from the user content/prompt and return
  the compact decision JSON (`chat.completion` / `text_completion`), with SSE streaming
  and `stream_options.include_usage`. Non-JSON text is 400. Tests added (47 unit + 10
  integration green).
- Diagnostics: `JEV_CPU_PROFILE` (host enqueue/alloc timing), `JEV_GDN_PROF` (chunked GDN
  phase clocks), and `JEV_INT8_GROUP_UNIFORM` (forced per-row scales; the group kernel
  must then match the per-token kernel — it does).

### Experiments (results)

- **BN=128 Marlin tiles** (thread_n_blocks=8, thread_k_blocks=4, tmb 2..4) for small M:
  after fixing the group-scale application to run at the last k of the stage
  (`k == b_sh_wr_iters - 1`, not hard-coded `k == 1`), the kernel is bit-identical to
  BN=256 but **neutral** at c1/c4/c16 and long (c1 23.75 vs 24.65, within run noise).
  The small-M path is weight-bound, not block-bound. Reverted; the instantiations and
  selection code were removed.
- **Per-token INT8 for the expanded set with clipping** (clip 0.995/0.99/0.98):
  fails the gate (MAE 2.5-3.7e-3, max up to 0.052). Group-64 scales are required for
  `up/down/in_qkv/out/q`; clipping is not a substitute.
- **All-linear group-64** (includes o/v/k): passes (28/28, MAE 9.692e-4, max 8.469e-3)
  but trades away most of the MAE margin for ~1.4% long-context speed. Kept opt-in via
  `JEV_NATIVE_INT8_EXCLUDE=none`; the default keeps o/v/k on W4A16.
- **GPU-resident embedding table** (2.37 GiB, device row gather): bit-identical logits,
  -2.5% at 4K and -2.2% at 16K; memory 16.85/18.10 -> 18.83/20.17 GiB. Upload first:
  a 2.4 GiB contiguous block can fail after 13.1 GiB of repacked weights fragment the
  pool.

### Measured saturation (expanded group config, server)

- Short, 16 unique 73-token requests, cache off: c1 24.5 req/s / p50 41.0 ms /
  util 51-95%; c2 37.4 / 54.0 ms / 84-91%; c4 47.8 / 83.2 ms / 86-92%; c8 50.3-53.8 /
  125-156 ms / 94-95%; **c16 49.8-50.5 / 240-271 ms / 94-99%, 351-361 W**.
- Long single request: 4K 0.985-0.989 s / 4.14-4.15k tok/s / 86-100% util / 389-428 W;
  8K 2.011-2.018 s / 4.05-4.07k / 91-100% / 420-427 W; **16K 4.195-4.201 s / 3.90k /
  96-100% / 423-427 W**.
- Peak memory grows across repeated 16K requests: 22.8 -> 23.2 -> 23.9 GiB. Needs the
  low-memory pool threshold lowered or a post-forward trim (open).
- Quality of this expanded config: suite28 28/28, MAE 8.11e-4, max delta 8.42e-3.

### Open work / next

1. Revert the default to the old selective per-token set and re-benchmark the server
   (expected 4K ~0.76 s, 16K ~3.3 s with the current binary).
2. Bound 16K memory growth (trim pools above ~12K tokens).
3. Short-context throughput: Marlin BN=128 instantiations (more N tiles, all 128 SMs;
   68 -> 136 blocks for N=17408) for M <= ~512.
4. Re-check why the server's long path previously differed from the CLI; current binary
   shows single ≈ batch (4K 1.005 vs 1.015 s, 16K 4.17 vs 4.23 s).
5. LiteLLM end-to-end with a real proxy/server (SDK installed in `/tmp/litellm-venv`).
6. Update RESULTS/VLLM-COMPARISON/KERNELS docs and commit.

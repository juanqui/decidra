

# Work journal

> Kernel-level CUTLASS / warp-specialization work is tracked separately in
> `docs/CUTLASS.md` (measurements, design, checklist).

## 2026-10-20 — W4A8 card example: artifact math vs vLLM weight-only

- Card example (billing / shipping / tech support), same W4A8 artifact + adapter:
  provider card 0.9963; local vLLM 0.31.0 0.9588; engine INT8 0.9606; CPU
  compressed-tensors decompress + LoRA + configured per-token INT8 0.9623; CPU
  weight-only 0.9582; engine weight-only 0.9588.
- The engine INT8 is 0.0017 from the CPU artifact-math reference; the local vLLM is
  0.0035 away. Across the 28-row suite the engine weight-only vs vLLM is 28/28,
  max 9.77e-3, MAE 1.23e-3, i.e. vLLM's W4A8 path behaves as weight-only here.
- The card's 0.9963 is not reproducible with the downloaded artifact under any local
  implementation; all cluster at 0.958-0.962, and the card number is close to the
  merged-BF16 teacher (0.9969 / 0.9976).
- The engine default INT8 is the faithful configured W4A8;
  `DECIDRA_NATIVE_INT8=false` reproduces the vLLM / weight-only behavior. The 28-row
  CPU artifact golden was paused to free RAM for the AutoTrust quant.

## 2026-10-20 — AutoTrust-JEV-27B-v1.1 INT4/INT8 quantization

- Model `autotrust/AutoTrust-JEV-27B-v1.1`: Qwen3.5 hybrid text stack whose
  `text_config` is byte-identical to JEV-27B-VL; separate BF16 `lm_head` with the
  255-code `jev-code-readout-v1`; serving needs StartLux bidirectional prompt
  attention plus the chat-template code prompt.
- Engine gap: the native backend is causal and uses the JEV verbalizer head. Faithful
  serving needs a bidirectional full-attention prefill and the code readout (the
  engine's `decision_logits` already returns arbitrary lm_head rows, so the readout
  itself is available). The quantization is otherwise portable.
- Calibration: 1024 kind-balanced rows (60/40 choice/noul) rendered with the model's
  `jev_format` chat template and 255 codes (113-563 tokens); 256 disjoint held-out.
- Quant: AWQ smoothing + GPTQ, W4A16 asymmetric group 64, actorder=weight, ignore
  lm_head/visual/mtp, max_seq 1024, output
  `/srv/models/autotrust-jev-27b-v1.1-awq-g64` (496 modules, 2 h).
- Post: vision spliced (333 tensors), lm_head BF16 indexed, engine decision head with
  the 255 readout ids / zero bias / T=1.0; `native/tests/autotrust_probe.rs` compiles.
- Validation (16 held-out rows, causal forward):
  - HF transformers on the quantized dir vs BF16: top-1 15/16, max 0.0587,
    MAE 1.97e-2, KL 2.59e-3 -> the AWQ+GPTQ g64 quant is good.
  - Native engine on the same quantized dir vs BF16: top-1 10/16, max 0.392,
    KL 6.92e-2; vs the HF quantized dir: 11/16, max 0.386, KL 6.65e-2.
  - Conclusion: the quantization is sound; the native engine's forward for the
    Qwen3.8 weights (the AutoTrust base differs from JEV-27B-VL despite the
    byte-identical text_config) is the next issue. The native `Engine::new` probe
    measures the W4A16 weight-only path; the INT8 activation setters are applied
    server-side, so the INT8 path needs a server-side or explicit-setter test.
  - The AutoTrust also requires StartLux bidirectional prompt attention and the
    chat-template code readout; both are separate engine-port items.

## 2026-10-20 — W4A8 group-128 checkpoint support

- Loader accepts group 128 with the compressed-tensors offset zero point 8
  (symmetric checkpoints omit the tensor); Marlin group-8 instantiations for the
  per-token INT8 path and the W4A16 fallback.
- PEFT rank-32 LoRA attached to every target linear and `lm_head`; adapter store,
  `DECIDRA_LORA_ADAPTER`, and `(x A^T) B^T` deltas. `adapter_config.json`
  writes `lora_alpha` as a float (64.0); parsing it as an integer silently made
  the scale 1.0 instead of 2.0 — fixed.
- Engine vs vLLM 0.31.0 with the same W4A8 artifact + adapter: top-1 28/28,
  max delta 0.0304, MAE 0.00266, KL 2.20e-4.
- Speed: short 2.2–2.6x vLLM; long 4K/8K/16K 0.903/1.846/4.159 s vs
  1.386/2.308/4.813. Peak 21.2 GiB with the host embedding table (21.0 GiB
  vLLM); the GPU table variant is 23.3 GiB and ~3% faster.
- Docker image built from the same commit, run on GPU 1 with the W4A8 artifact:
  healthy, suite 28/28, same engine-vs-vLLM deltas.

## 2026-10-20 — FMA group epilogue shipped

The group epilogue now uses `fmaf(c0, bs0*a_s, acc)` instead of
`acc += c0*bs0*a_s`. Measured on the shipped binary: suite **28/28, KL 3.269e-5,
MAE 7.166e-4, max 1.027e-2** (gate 0.011); batched c8 max 1.305e-2 / MAE
7.23e-4, c16 max 1.251e-2 / MAE 1.115e-3, top-1 28/28. Server 4K/8K/16K
**0.833 / 1.711 / 3.740 s** (from 0.854/1.757/3.838, ~2.5%); short c1-16
**25.78 / 40.73 / 53.10 / 56.97 / 56.93 req/s** (c4/c8 ~9% better). The exact
`+= c0*bs0*a_s` order was 1.9% slower and is the documented fallback. The
pipelined scale-fetch and double-buffered-accumulator designs were attempted and
rejected; see `docs/CUTLASS.md`.

## 2026-10-20 — post-goal experiment sweep

All A/Bs in-session on the shipped kernel set (clock drift makes cross-session
4K numbers range 812-854 ms; compare only within a session).

- **Group epilogue FMA**: two reassociations tested. `fmaf(c0, bs0*a_s, acc)`
  is 1.9% faster (16K 3773 -> 3703 ms, up 2.227 -> 2.122 ms/op) and passes at
  max delta **1.027e-2** / MAE 7.17e-4, but batched c8 rises to **1.30e-2**.
  `fmaf(c0*bs0, a_s, acc)` is 2.3% faster yet **fails** (max 1.731e-2, MAE
  1.69e-3). Reverted: the 1.9% does not justify a 6.6% gate margin.
- **All-linear INT8** (include o/v/k): 4K 823.2 vs 831.4 ms, 16K 3629.3 vs
  3670.2 ms (**1.0-1.1%**). Gate passes but MAE margin is thinner; remains the
  documented opt-in `DECIDRA_NATIVE_INT8_EXCLUDE=none`.
- **GDN phase profile (4K, per chunk cycles)**: solve 36%, state 13%, vnew 12%,
  QS 12%, QK+mask 10%, out 4%, loads 3%. The chunk-parallel FLA split must add
  the 128x128 transition `decay*I - K^T W` per chunk (~12 GFLOP/layer) plus
  ~1.5 GB/layer of state traffic, which cancels the 1.33x occupancy gain;
  the existing fused-serial design is near the practical limit. A rewrite is
  only justified with a warp-specialized kernel, not a naive split.
- **Launch gap**: 73 tokens, event sum 38.3 ms vs forward min 41.5 ms -> ~3.2 ms
  (8%) of host/launch time; at 4K ~9 ms (1.1%). CUDA graphs could recover most
  of it, but require pinned per-graph buffer arenas (the pool reuses addresses)
  and capture-safe FlashInfer/Marlin; a whole-forward static path, not a flag.
- **Blocked-transposed group scales** (`[tile][group][64 rows]`, so each stage
  loads 64 contiguous rows): bit-identical logits at 4K and 16K, but **15%
  slower** in-session (up 2.577 vs 2.135 ms/op, down 2.522 vs 2.126, in_qkv
  1.436 vs 1.259). A temporary skip-quant probe (removed) showed the quantizer
  costs <0.01 ms/op (full vs skip identical), so the regression is inside the
  Marlin kernel (register/occupancy from the extra address math), not the load
  pattern. Reverted; the strided `[row][group]` staging remains.
- Methodology note: GPU clocks drift 5-7% across sessions (same config 4K
  812-854 ms). Same-binary/same-session A/Bs are the only trustworthy deltas;
  the all-linear 1% and the blocked-layout 15% were measured that way.
- **Remaining real bets**: a CUTLASS-style group-INT8 GEMM (target 400+ TOPS vs
  330; 8-12% end-to-end) and rotation/SmoothQuant-style activation transforms
  so per-token INT8 (463 TOPS) can replace group-64 (up to 15-25% GEMM time,
  but a checkpoint transformation). Both are multi-day projects.


## 2026-10-20 — session end

Commits (oldest first): `eab402c` mixed per-module W4A8-INT8 with group-64 scales,
`e7fb6e8` OpenAI-compatible decision routes for LiteLLM, `f0fb849` resident GPU
embedding table, `94dfcb9` shared-memory group-scale staging, `2cfdb86` fused
SiLU→down quantizer, `b42fbcc` lone-request batch window.

Final verification on the shipped binary: suite **28/28, KL 3.095e-5, MAE
8.642e-4, max 7.511e-3**; LiteLLM non-streaming + SSE streaming OK; 16K probe
**3.843 s at 100% GPU / 435 W / 20.26 GiB**; `cargo test` 47 unit + 10
integration; clippy `-D warnings` clean. GPU 1 left idle.

Final server medians (3 runs, cache off): short c1/c2/c4/c8/c16 **25.62 /
40.33 / 48.79 / 52.13 / 55.94 req/s**; long 4K/8K/16K **0.854 / 1.757 /
3.838 s**. Versus vLLM 0.31.0: **1.36-1.99x** short, **1.44-1.52x** cold long;
memory 18.83 GiB resident / 20.11 GiB peak vs 21.9 GiB; startup 9.2 s vs 145 s.

Open leads for the next session: GDN chunk-parallel decomposition (14% of 4K),
prefix caching for shared long prefixes (2x on a 4K shared prefix), CUDA graphs
for sub-1K shapes (~5% of 4K wall is inter-kernel gaps), and a larger-M Marlin
tile if the group epilogue can be pushed closer to the per-token rate.


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

`DECIDRA_NATIVE_INT8_GROUP_SCALE=true`, `DECIDRA_NATIVE_INT8_GROUP_EXCLUDE=mlp.gate_proj`,
`DECIDRA_NATIVE_INT8_EXCLUDE=self_attn.o_proj,self_attn.v_proj,self_attn.k_proj`.

Quality (`eval/suite.py` vs CPU AWQ golden): **28/28, KL 3.095e-5, MAE 8.642e-4,
max 7.511e-3** — all strict gates pass.

Engine cold/warm single request (cache off): 73 tok **~56 ms mean / 37.6 ms min**,
4K **924 ms** (905 min), 16K **3900 ms** (3876 min).

Server (one resident process, 16 unique prompts/level, cache off):

| level | req/s | p50 | GPU util | power |
|---|---:|---:|---:|---:|
| c1 | 25.62 | 39.5 ms | 97% | 376 W |
| c2 | 40.33 | 49.6 ms | 96% | 349 W |
| c4 | 48.79 | 76.5 ms | 96% | 318 W |
| c8 | 52.13 | 132.2 ms | 97% | 330 W |
| c16 | 55.94 | 168.8 ms | 99% | 353 W |

Long single request: **4K 0.854 s (4,787 tok/s, 100% GPU, 384-428 W)**,
**8K 1.757 s (4,655 tok/s, 100%, 422 W)**,
**16K 3.838 s (4,269 tok/s, 100%, 434 W)**.

Resident/peak memory: **18.83 GiB** short, **20.51 GiB during 16K**, flat across
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
  once per thread instead of every GEMM; `DECIDRA_MARLIN_CHUNK`, `DECIDRA_GEMM_DEBUG*` and
  `DECIDRA_MARLIN_DEBUG` are cached (`LazyLock`) instead of read per GEMM.
- **Resident GPU embedding table** (2.37 GiB): rows are gathered on-device with a
  16-byte-vector kernel, bit-identical to the host gather. 4K 938.8 -> 915.8 ms
  (-2.5%), 16K 4096 -> 4004 ms (-2.2%), short min 40.8 -> 37.2 ms. The host gather
  (parallel, up to 8 threads) remains the fallback when disabled.
- **CLI benchmark parity**: `decidra-native layer/batch` now enables chunked+TC GDN by
  default (the server already did); `--legacy-gdn` disables all chunked paths. Earlier CLI
  numbers measured the scalar f32 GDN by mistake.
- **OpenAI-compatible decision routes** for LiteLLM: `/v1/chat/completions` and
  `/v1/completions` parse a JSON decision request from the user content/prompt and return
  the compact decision JSON (`chat.completion` / `text_completion`), with SSE streaming
  and `stream_options.include_usage`. Non-JSON text is 400. Tests added (47 unit + 10
  integration green).
- Diagnostics: `DECIDRA_CPU_PROFILE` (host enqueue/alloc timing), `DECIDRA_GDN_PROF` (chunked GDN
  phase clocks), and `DECIDRA_INT8_GROUP_UNIFORM` (forced per-row scales; the group kernel
  must then match the per-token kernel — it does).

### Experiments (results)

- **Lone-request batch window**: when the queue is empty at the first dequeue, cap
  the wait at 500 us instead of 2 ms; a queue with waiters keeps the full window.
  c1 24.6 -> 25.6 req/s (p50 41.0 -> 39.5 ms), c2 40.3, c4 48.8-52.1, c16 55.9
  req/s; no burst regression.
- **Fused SiLU into the down quantizer**: `silu(gate)*up` is computed in the INT8
  group quantizer for `down_proj`, removing the separate BF16 elementwise tensor and
  kernel. Bit-identical logits; down 2.390 -> 2.154 ms/op at 4K, engine 4K 879 ->
  854 ms, 16K 3.921 -> 3.777 s. Server 4K/8K/16K 0.879/1.805/3.921 ->
  0.857/1.758/3.833 s; suite unchanged (8.642e-4 / 7.511e-3).
- **Group-scale staging in shared memory**: each K stage stages the block's 64
  activation scales once (reusing the unused per-row scale reservation in the dynamic
  buffer) instead of 8 global loads per thread per group. Group GEMMs 5-6.5% faster
  (up 2.346 -> 2.224 ms, down 2.512 -> 2.390, in_qkv 1.394 -> 1.321 at 4K), logits
  bit-identical. Server 4K/8K/16K 0.913/1.871/4.051 -> 0.879/1.805/3.921 s; c16
  53.4 -> 55.9 req/s.

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
  `DECIDRA_NATIVE_INT8_EXCLUDE=none`; the default keeps o/v/k on W4A16.
- **GPU-resident embedding table** (2.37 GiB, device row gather): bit-identical logits,
  -2.5% at 4K and -2.2% at 16K; memory 16.85/18.10 -> 18.83/20.17 GiB. Upload first:
  a 2.4 GiB contiguous block can fail after 13.1 GiB of repacked weights fragment the
  pool.

### Batched quality and soak (final config)

- 28-row suite, concurrent requests (server batch path), vs CPU golden:
  **c8** top-1 28/28, MAE 9.85e-4, max 9.52e-3; **c16** top-1 28/28, MAE
  1.274e-3, max 1.098e-2. The strict gate is defined at batch=1 (8.64e-4 /
  7.51e-3); the batched deltas are the documented BF16/M-tiling composition noise
  (W4A16 alone is worse at 1.78e-2). Top-1 is stable at every concurrency.
- Mixed soak: 200 suite requests at c16 (39.3 req/s on the mixed suite workload,
  0 errors, 259/259 ok in metrics) followed by 4K/8K/16K probes (0.881/1.804/
  3.929 s, 100% GPU). GPU memory 16.5 -> 17.1 GiB after the short soak and
  21.1 GiB at the 8K/16K peak, flat afterwards; no leak.

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

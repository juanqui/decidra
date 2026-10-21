# H100 NVL: vLLM 0.31 baseline vs Decidra

Measured 2026-10-21 on a single **NVIDIA H100 NVL** (Hopper SM90, 94 GB,
400 W, 1.785 GHz boost, driver 580.82.09/CUDA 13.0), 16 vCPU, 96 GB RAM.
Same W4A8 artifact (`aioutfitters/JEV-27B-VL-AWQ-INT4-W4A8`) and decision
protocol for both stacks.

**Outcome:** vLLM is the faster server on this H100. It wins long prefill
(**1.5–1.7×**), the maximum short-concurrency sweep (**79.0 vs 57.7 req/s**),
and context capacity (**201K tokens** vs the engine's verified 122.5K). The
native engine wins low-concurrency latency (roughly 2× at c=1) and uses no
Python at serving time. The FA3 and Hopper cuBLAS paths are integrated and
measured, but they do not close the gap: the engine still lacks a fused
Machete-style W4A8 WGMMA/TMA GEMM and a parallel GDN scan.

## vLLM configuration (best measured)

`vllm/vllm-openai:v0.31.0` used directly as the Vast instance image (Ubuntu
24.04, glibc 2.39, nvcc 13.0, vLLM 0.31.0/Torch 2.13+cu130). The provider's
`serve_decide.py` shim adds `/v1/decide`.

```text
--model /workspace/models/jev-w4a8 --served-model-name autotrust/JEV-27B-VL
--enable-lora --max-lora-rank 32 --lora-modules jev-decision=.../adapter_vllm
--logprobs-mode processed_logprobs --max-logprobs 256
--max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95
--mamba-cache-mode align --max-num-seqs 32 --trust-request-chat-template
--limit-mm-per-prompt '{"image": 8}'
--enable-chunked-prefill --max-num-batched-tokens 16384
--performance-mode throughput --no-enable-prefix-caching
--no-enable-log-requests --disable-log-stats
```

Selected kernels from the startup log: **MacheteLinearKernel** for the
compressed-tensors W4A8/WNA8Int linears (some layers also log Marlin),
**FLASH_ATTN** for full attention, **FlashInfer GDN prefill** for the linear
attention layers. Startup was 3m30s cold (torch.compile 88 s); restarts reuse
the compile cache in ~2.5 min. The provider recommends `--max-num-seqs 8` for
LoRA correctness; 16 and 32 both passed this 28-row suite, but 8 remains the
safe setting for unseen prompts. Prefix caching was disabled for cold
measurements; that makes vLLM set mamba cache mode to `none` instead of
`align`.

## Results

### Short unique prompts (~71 tokens; 32 requests/level, median of 3)

| concurrency | vLLM max32 req/s | vLLM p50 | engine req/s | engine p50 |
|---:|---:|---:|---:|---:|
| 1 | 10.64 | 101 ms | ~18 | ~55 ms |
| 2 | 10.90 | 197 ms | — | — |
| 4 | 21.21 | 195 ms | — | — |
| 8 | 42.78 | 199 ms | — | — |
| 16 | 61.94 | 217 ms | 47.30 | 350 ms |
| 32 | **78.99** | 381 ms | 53.09 | 613 ms |
| 64 | — | — | 56.72 | 1113 ms |
| 128 | — | — | **57.74** | 2138 ms |

vLLM max16: 9.86 / 10.73 / 20.15 / 38.80 / 54.95 / 53.93 req/s.
vLLM max8 (cold-server median): 5.38 / 5.44 / 9.74 / 19.13 / 19.89; the third
run after full LoRA JIT warmup reached 9.4 req/s at c1, so the max8 medians
are cold-biased. The 32-request levels and the max32 run followed warmups.

Engine row is the Marlin W4A8-INT8 server, `MAX_BATCH=64`, 10 ms window,
persistent connections, 256 requests/level. The c1 engine figure is the H100
SXM Marlin result (18.8 req/s / 55 ms); the NVL sweep started at c16.

### Cold long prefill

| prompt tokens | vLLM latency | vLLM tok/s | engine latency | engine tok/s | vLLM/engine |
|---:|---:|---:|---:|---:|---:|
| 4,832 | **0.688 s** | **7,067** | 1.149 s | 4,207 | **1.68×** |
| 10,705 | **1.625 s** | **6,582** | 2.508 s | 4,261 | **1.54×** |
| 19,498 | **3.126 s** | **6,222** | 4.627 s | 4,205 | **1.48×** |

The engine numbers are the dequant→BF16+cuBLAS path; Marlin is 5.9% slower at
19,498 tokens (4.898 s with FA3). vLLM used a single cold request; prefix
caching off.

### Context

| prompt tokens | vLLM | engine |
|---:|---:|---:|
| 65,767 | 12.38 s, 5,298 tok/s | 17.79 s (SXM), 3,697 tok/s |
| 122,516 | 26.90 s, 4,549 tok/s | 39.98 s, 3,065 tok/s |
| 201,177 | 53.04 s, 3,790 tok/s | not supported (fails ~130K) |

The engine's SM90 illegal-address bug at ~130K tokens (from the earlier SXM
run) remains. vLLM served 201K tokens without error in 94 GB.

### Quality (28-row W4A8 suite vs CPU artifact golden)

| stack | top-1 | mean KL | mean MAE | max Δ |
|---|---:|---:|---:|---:|
| vLLM max32 | 28/28 | 3.361e-4 | 3.922e-3 | **2.669e-2** |
| vLLM max16 | 28/28 | 3.323e-4 | 3.849e-3 | 2.668e-2 |
| vLLM max8 | 28/28 | 3.073e-4 | 3.650e-3 | 2.667e-2 |
| engine Marlin INT8 (SXM) | 28/28 | 2.980e-4 | 3.485e-3 | 4.204e-2 |
| engine dequant weight-only (NVL) | 28/28 | 2.593e-4 | 3.490e-3 | 2.282e-2 |

vLLM's max-Δ gate passes at 0.02; the engine's Marlin path does not on the
same near-tie row. The dequant path is closer but still above 0.02. vLLM's
Machete path is weight-only (BF16 activations), matching the README note.

## Where the engine time goes

Profile at 19,504 tokens, one request:

| path | GDN recurrent | GEMMs | attention | total |
|---|---:|---:|---:|---:|
| Marlin W4A8-INT8 | 1.336 s | ~3.2 s (unlisted residual) | 0.148 s | 4.96 s |
| dequant + cuBLAS | 1.562 s | 2.43 s (cuBLAS) | 0.150 s | 4.75 s |

- **GDN recurrent is 1.34 s** (27–34%). The chunked TC kernel launches only
  96 CTAs on 132 SMs and walks 610 chunks sequentially per CTA. Splitting value
  columns 64→32 (192 CTAs) was slower (1.77 s), because it halves per-CTA
  efficiency without shortening the sequential chain. The legacy GDN path was
  8.65 s.
- **GEMMs** are the other 2.4–3.2 s. The cuBLAS dequant path uses Hopper
  WGMMA/TMA inside cuBLAS and is the faster long-context GEMM in this engine,
  but it re-dequantizes each module per forward (54 GB of BF16 traffic) and is
  much slower at short concurrency (5.3 req/s at c1).
- **Attention is not the bottleneck.** FA3 reduced the 16 full-attention layers
  from **275.6 ms to 147.2 ms (1.87×)** and total latency from 4.939 s to
  4.898 s.

## What was added

- **FlashInfer FA3 (SM90a) prefill**: `native/third_party/flashinfer/fi_prefill_fa3.cu`,
  compiled with CUTLASS 4.5/CuTe for `sm_90a`; embedded via
  `DECIDRA_FA3_SO` (or compiled with `DECIDRA_FA3=1` +
  `DECIDRA_CUTLASS_INCLUDE`). Runtime selection by compute capability and
  `DECIDRA_FLASHINFER_FA3`; FA2 remains the fallback and the default build
  (no FA3 library) is unchanged.
- **Architecture-parametric CUDA build**: `DECIDRA_CUDA_ARCH=sm_90`; cubin/SO
  names include the arch.
- **SM-aware Marlin launch** (from the earlier L40S work) and the existing
  **cuBLAS dequant path** (`MARLIN=false`, `FUSED_W4A16=false`) as the Hopper
  WGMMA/TMA GEMM experiment.
- `eval/saturate.py` for persistent-connection concurrency sweeps.

None of these is a fused W4A8 WGMMA/TMA GEMM. cuBLAS supplies WGMMA/TMA for
BF16, and FA3 supplies it for attention, but the quantized linear path still
pays dequantization or `mma.sync` Marlin costs.

## To beat this vLLM baseline

1. **Fused W4A8/W4A16 WGMMA/TMA GEMM** (Machete/CUTLASS mixed-input): int4
   weights stay int4; dequantize in registers inside the WGMMA mainloop. Target
   GEMM 2.4 s → ~1.2 s. This is the largest and most direct win.
2. **Parallel GDN chunk scan**: separate chunk-local W/U/decay from the state
   prefix scan and output pass, so chunks run in parallel. Target GDN
   1.34 s → ~0.4 s.
3. **CUDA graphs for the short path** to reduce the per-forward launch/queue
   cost at c1–c8.
4. Fix the SM90 >130K context illegal address (likely a sequence/workspace
   index limit) and raise the allocator ceiling.

With (1) and (2), the expected 19.5K time is ~2.6–2.9 s (6.7–7.5K tok/s),
competitive with vLLM's 3.13 s. With only one of them, vLLM still wins long.

## Cost

Instance: 0.637 h × $2.7467 GPU + $0.018 storage + $0.053 download =
**$1.822** (balance $30.3205 → $28.4409 = $1.8796 incl. finalization).
Destroyed; zero instances remain.

Raw artifacts: `eval/out/h100nvl/{vllm_suite,vllm_batch,vllm_long,vllm_ctx,
vllm16_suite.json,vllm16_batch,vllm32_suite.json,vllm32_batch,h100_dequant,
h100nvl_sat64,one-100000,fa3-0.json,fa3-1.json}/`.

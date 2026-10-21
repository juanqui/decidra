# Night run results — W4A8 decision-model support

Date: 2026-10-21. GPU: 1 (RTX 4090 24 GB). Host RAM: 122 GB.

## Checkpoints

All three decision models were quantized with the same recipe:

- weights: INT4, symmetric, group 128, `pack-quantized`
- `input_activations`: INT8, symmetric, per-token, dynamic
- AWQ + GPTQ, `actorder=weight`, 1,024 calibration prompts in the model's serving format
- readout/lm_head, embeddings, norms unquantized

| model | path | size | `weight_packed` | format |
|---|---|---|---|---|
| d3-turbo | `/srv/models/d3-turbo-w4a8-g128` | 17 GB | 496 | pack-quantized |
| pplx-decider-v1.2 | `/srv/models/pplx-decider-v1.2-w4a8-g128` | 18 GB | 496 | pack-quantized |
| AutoTrust-JEV-27B-v1.1 | `/srv/models/autotrust-jev-27b-v1.1-w4a8-g128` | 17 GB | 496 | pack-quantized |

## Engine results

### JEV W4A8 regression — PASS

`eval/suite.py` against the CPU artifact golden (`eval/data/w4a8-cpu-suite28.json`):

- 28/28 rows, top-1 28/28
- mean KL 2.33e-4, mean MAE 0.00328, max Δ 0.0176

### AutoTrust — layer parity PASS

`autotrust_parity` with the noncausal HF layer dump: layers 0, 1, 2, 3, 7, 15, 31, 47, 63
(634,880 values per layer) compared; test passed. Readout/final-norm comparison still open.

### pplx — engine marker extraction PASS

`pplx_eval` extracted the hidden state at the declared decision markers for all 32 rows
(`/tmp/pplx_engine_hidden.json`). HF reference comparison blocked: the source `Decider`
loads on CPU but the process is killed at 64% of shards (1183-shard layout), and the
quantized checkpoint has no `auto_map`/custom model class, so the native `Qwen3_5Model`
decompress hits a module-shape mismatch.

### d3-turbo — FAIL (unexplained)

Engine readout vs the BF16 `D3Model` reference, 32 held-out rows:

- top-1 8/32, max Δ 0.784, MAE 0.235, KL 0.91
- engine probabilities near-uniform (e.g. 0.33/0.18/0.32/0.17) vs sharp reference
  (0.01/0.12/0.73/0.13)
- the engine and the initial CPU golden disagree (top-1 0/32), but that golden ran with
  **0 INT8 activation hooks** (wrong module prefix), so it is not a valid comparison
- the corrected CPU golden path is blocked by the same class/decompress problem: the
  d3's custom `D3` loads on CUDA by default, and the plain `Qwen3_5` class does not match
  the checkpoint's `model.layers.*` naming

## Code changes

- `native/src/config.rs`: read `attention_mode` from `decision_config.json` when
  `config.json` does not declare it; accept `prefix_bidir` as noncausal full. This is the
  mask the d3/p l x checkpoints declare; the engine previously ran them causal.
  AutoTrust was unaffected because it declares `jev_prompt_attention` in `config.json`.
- commit `54f7376`.

## Open items

- d3-turbo readout/hidden root cause (engine vs same-checkpoint reference).
- pplx HF reference on the host (loader killed at 64% of shards).
- AutoTrust final-norm/readout comparison and CPU golden.
- Prefix-cache GPU phases (only the CPU core is implemented).
- Performance benchmarks.

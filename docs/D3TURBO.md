# d3-turbo state-once support

Status: engine, quantization, eval harness and the native `/v1/systemone` d3
adapter are in the tree. The server self-tests the exact state-once path at
startup. Remaining work is performance and a larger accuracy sample (§7).

## 1. What the model does

`vllm-sr/d3-turbo` is a 26.1B Qwen3.5 hybrid decision model: 48 Gated DeltaNet
layers and 16 full-attention layers (indices 3, 7, …, 63), hidden 5120, 24 Q /
4 KV heads at head_dim 256, 16/48 GDN K/V heads at 128, a separate F32
`[255,5120]` answer-code readout, and `attention_mode: prefix_bidir`.

`prefix_bidir` splits a prompt at the end of the state:

```
[ system + State: <state> ][ \n\nQuestion: ... options ... ][ readout ]
        state block S                     question block Q
```

- state queries see every state key, both directions;
- question queries see every state key **and every key in their own block**,
  both directions;
- **no state query sees a question key.**

The GDN layers stay causal and scan the state once. The state's hidden states,
full-attention K/V, final recurrent state and convolution tail are therefore
independent of every question. d3-turbo computes them once per request and
forks every question from that cache. `vllm-sr/d3` (Decision 3.0) is a
different mask (`noncausal_full_attention`); its state does see the questions,
so it cannot use this cache.

## 2. Engine API

```rust
let engine = Engine::new(model_dir)?;
engine.set_marlin(true);
engine.set_marlin_int8(true);
engine.set_int8_group_scale(true);   // group-64 W4A8
engine.set_gdn_chunked(true);
engine.set_gdn_chunked_tc(true);
engine.set_gdn_chunked_min_seq(16);
engine.preload_all()?;

let cache = engine.capture_state(&state_tokens)?;
let logits = engine.readout_state_once_batch(&cache, &question_blocks)?;
```

`capture_state` stores, per full layer, post-RoPE K/V `[S,4,256]`, and per GDN
layer the final f32 `[48,128,128]` recurrent state and the last three
pre-conv `in_proj_qkv` rows. `readout_state_once_batch` runs all question
blocks as one ragged batch: one GEMM per projection, one cascade attention
launch per row, a seeded GDN recurrence, and one `[B,255]` F32 readout.

Per-row reference for tests:

```rust
engine.decision_logits_split(&full_tokens, state_len, &codes)?;
```

`Engine::new` refuses a non-causal checkpoint without the vendored FlashInfer
library; `prefix_bidir` never falls through to the causal native kernels.

## 3. Correctness (BF16, reference `d3_runtime.py`)

| check | max abs Δp |
|---|---:|
| full noncausal per-row, engine vs HF | 0.0061 |
| per-row block cascade, engine vs HF | 0.0022 |
| state-once (state + question pass), vs HF block | 0.0035 |
| state-once with production chunked GDN, vs HF block | 0.0029 |
| state-once vs per-row block, 75-token state / 49-token question | 0.0052 |
| batch-of-4 vs single, after the F32 readout | 0.0056 |
| zero layer-0 GDN seed (negative control) | 0.0049 |
| zero layer-0 convolution tail (negative control) | 0.0047 |

The F32 readout matters: with the previous BF16 head, the batch-vs-single gap
was 0.014. `HeadMatrix` now stores the head in F32 and runs an F32 cuBLAS GEMM,
matching `hidden.float() @ readout.T`.

## 3.1 Serving

Build with `--features native` and start with:

```
CUDA_VISIBLE_DEVICES=1 DECIDRA_BIND_ADDR=127.0.0.1:8899 \
DECIDRA_MODEL=d3-turbo \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo-w4a8-g64-v4 \
DECIDRA_TOKENIZER=/srv/models/d3-turbo-w4a8-g64-v4/tokenizer.json \
DECIDRA_NATIVE_MARLIN=1 DECIDRA_NATIVE_INT8=1 \
DECIDRA_NATIVE_INT8_GROUP_SCALE=1 \
DECIDRA_NATIVE_INT8_EXCLUDE=self_attn.o_proj,self_attn.v_proj,self_attn.k_proj,mlp.gate_proj \
DECIDRA_NATIVE_GDN_CHUNKED=1 DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ=16 \
DECIDRA_NATIVE_GPU_EMBED=0 DECIDRA_D3=1 ./target/release/decidra
```

The server detects `decision_config.json` with `attention_mode: prefix_bidir`,
renders the checkpoint's prompt with `src/d3.rs`, splits each prompt by
tokenizer offsets, verifies every question produced the same state tokens,
captures the state once, runs all question blocks as one batch, and returns
the `/v1/systemone` answer shape (choice, noul and score). Startup runs an
8-token state + one-question self-test through the same path; a broken mask,
artifact or FlashInfer build refuses to serve. `DECIDRA_D3=1` makes a missing
`prefix_bidir` config a startup error instead of a silent fallback.

Text only: image/video states return an error until the vision tower is
ported. Per request: at most 64 questions; each question's state plus its own
block is at most 32K tokens; larger question sets are split into several forks
over the same state cache.

## 4. Quantization

The shipped recipe is **GPTQ-only, INT4 symmetric group 64, INT8 per-token
activations with per-64-group scales**, 1,024 block-masked calibration rows.
AWQ is not shipped: both AWQ+GPTQ test artifacts were far from BF16 (max Δp
0.50), while GPTQ-only stayed within 0.04 on the same row. The shipped v4
artifact quantizes `in_proj_a|b` as well; a v5 control kept them BF16 and was
slightly worse on the 32-row gate (30/32 vs 31/32), so v4 is the artifact.

The previous W4A8 artifact was quantized from **randomly initialized weights**
because `llmcompressor.oneshot(model=<path>)` loaded `AutoModelForCausalLM`
while the checkpoint uses `language_model.*` keys. The fixed pipeline loads
`Qwen3_5ForCausalLM` with
`key_mapping={r"^language_model\.": "model."}`, verifies a source tensor
bit-exactly, and uses the real block mask for calibration:

```
<safetensors>  state queries -> state keys only
               question queries -> state + own question block
```

The GDN layers receive the 2-D padding mask. The mask path is
`Qwen3_5TextModel.forward`'s `attention_mask` dict.

Artifact: `/srv/models/d3-turbo-w4a8-g64-v4` (15 GB after stripping the
CausalLM `lm_head`). Dequant gate: 496/496 modules pass cosine ≥ 0.95 against
the BF16 source; the v4 32-row eval passes:

| variant | top-1 | mean MAE | max Δp | flips |
|---|---:|---:|---:|---|
| W4A16 (dequant weights) | 31/32 | 0.0154 | 0.083 | 1 near-tie (BF16 margin 0.025) |
| W4A8, production exclusions + gate W4A16 | 31/32 | 0.0156 | 0.0942 | 1 near-tie |
| W4A8, k/v/o exclusions only | 31/32 | 0.0144 | 0.121 on a correct row | 1 near-tie |
| W4A16, same artifact | 31/32 | 0.0151 | 0.0875 | 1 near-tie |

`eval/d3_compare.py` gates on ≥95% top-1, mean MAE ≤0.02, max Δp ≤0.10 on
non-near-tie rows, and no flip whose BF16 margin is above 0.05.

Batch-shape checks: batch-of-one is bit-identical to the single path; four
questions sharing one valid state move by at most 0.037 (W4A8) or 0.009
(W4A16). Through the server, the same question in 1, 2, 4 and 8-question
requests moves by at most 0.006.

End-to-end on a 3-question request (choice + noul + score) through the server,
against the checkpoint's own `D3.system_one` on CPU BF16:

| answer | server | BF16 |
|---|---|---|
| route returns | 0.999938 | 0.999955 |
| receipt yes | 0.999875 | 0.999903 |
| urgency score | 1.5272 | 1.5284 |
| urgency p(Routine) | 0.0288 | 0.0147 |

Largest probability difference 0.0272; score difference 0.0012. The batched
W4A8 path is what the server serves; single-question parity (0.0056 max Δp
vs batch-of-one) is a separate check.

## 5. Performance on one RTX 4090

2,048-token state, 64 question blocks (3,236 question tokens), weights
preloaded, `gpu_embed` off:

| phase | time |
|---|---:|
| preload (15 GB + Marlin repack) | 6.9–9.6 s |
| state capture (default exclusions) | 465 ms |
| 64-question pass (default exclusions) | 734 ms |
| state capture (gate also W4A16) | 543 ms |
| 64-question pass (gate also W4A16) | 853 ms |
| 64 single-question passes | 1.97 s |
| resident device free | 9.26 GiB |

Question-pass profile before split-KV (sum 819 ms): INT8 GEMMs ~500 ms, GDN
recurrent 126 ms (chunked TC), attention 128 ms (64 rows × 16 layers).
Split-KV and the server's INT8 exclusions brought the measured pass to
734 ms. State capture is ~465 ms; the state GEMMs are the same Marlin path.

The 64-question request is ~1.20 s in the engine, 2.6× faster than running the
same questions one at a time.

**Same-machine baseline.** vLLM-SR `vllm-srun` 0.2.0 (native engine,
`--max-num-seqs 64 --enable-shared-context`) served the same 15 GB int4
artifact and the same 2,076-token state / 64-question request on the same
RTX 4090: **3.89 s** per request (three runs, 3.86–3.89 s). The native server
answered in **1.65 s** (three runs, 1.65–1.66 s), **2.36× faster**, with the
same top choice and a largest probability difference of 0.005 on the repeated
question. vLLM-SR loads the artifact as W4A16 (int4 unpacked to BF16); the
native path serves the same weights as W4A8 g64 with per-group INT8 scales.

## 6. Tests

```
# BF16 parity, 32 rows
CUDA_VISIBLE_DEVICES=1 DECIDRA_D3_MODEL=/srv/models/d3-turbo \
  DECIDRA_D3_PROBE=/tmp/d3_probe_split32.json \
  cargo test --release -p decidra-native --test d3_eval -- --ignored --nocapture

# state-once / batch parity and negative controls
CUDA_VISIBLE_DEVICES=1 DECIDRA_NATIVE_CACHE=0 DECIDRA_GDN_CHUNKED=1 \
  cargo test --release -p decidra-native --test d3_state_once_negative -- --ignored --nocapture

# 2K state, 64 questions
CUDA_VISIBLE_DEVICES=1 DECIDRA_MARLIN=1 DECIDRA_INT8=1 DECIDRA_INT8_GROUP=1 \
  DECIDRA_PRELOAD=1 DECIDRA_GDN_CHUNKED=1 DECIDRA_GDN_CHUNK_MIN=16 \
  DECIDRA_D3_MODEL=/srv/models/d3-turbo-w4a8-g64-v4 \
  DECIDRA_D3_PROBE=/tmp/d3_once_64_2k.json \
  cargo test --release -p decidra-native --test d3_perf -- --ignored --nocapture

# artifact dequant gate
/srv/models/jev-awq/venv/bin/python quant/d3turbo/verify_quant.py \
  --quant /srv/models/d3-turbo-w4a8-g64-v4 --group 64
```

## 7. Remaining work

1. **Per-row attention cost.** 1,024 FlashInfer launches per 64-question
   pass. A split-KV workspace was added to the cascade/ext shim; measure it
   at 2K. The next step is the cross/self/LSE-merge form or a batched ragged
   cascade, which should remove most launch and occupancy cost.
2. **Shared GDN seed.** 64 copies of a 3 MiB state per GDN layer (~9 GiB
   copied per pass) plus a final write-back the questions do not need. A
   read-only seed pointer on the batched recurrent kernels removes it.
3. **Marlin tuning.** ~200 TFLOPS of the 4090's 660 INT8 peak. Concatenating
   q/k/v and gate/up at load, larger row chunks, and the fp32 split-K reduce
   are the candidates; measure each.
4. **Larger accuracy set.** 32 rows is a smoke gate. Move to 256–1,024 rows
   with noul/score and ≥2K states before publishing a quality claim.
5. **State LRU.** The cache is per request today; a short-TTL exact-state
   LRU is the next win for repeated states.

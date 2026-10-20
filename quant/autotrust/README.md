# AutoTrust-JEV-27B-v1.1 INT4/INT8 quantization

Production-format quantization of
[`autotrust/AutoTrust-JEV-27B-v1.1`](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1)
for the Decidra native engine, plus the reference harness used to validate it.

## Model shape

- Qwen3.5 hybrid text stack (64 layers, 48 GDN + 16 full attention, hidden 5120,
  intermediate 17408, vocab 248320); its `text_config` is byte-identical to
  JEV-27B-VL.
- BF16 `lm_head` shipped separately; the decision readout is `jev-code-readout-v1`:
  a system prompt + chat template, one coded option line per option, and a softmax
  over the first `n` answer-code token logits at the last prompt position
  (`vllm_readout.json`, temperature 1.0, zero bias).
- Serving upstream requires StartLux bidirectional prompt attention; the HM
  transformers/native causal forward is not the model's intended attention.

## Recipe

- AWQ smoothing (duo scaling, 40-point grid) + GPTQ.
- W4A16, asymmetric INT4, group 64, `actorder=weight` — the engine's
  highest-quality supported weight layout (Marlin W4A16 + runtime INT8 path).
- Ignore `lm_head`, vision tower, and MTP.
- Calibration: 1024 kind-balanced rows (60/40 choice/noul) from the provider
  decision corpus, rendered with the model's own `jev_format` chat template and
  255 answer codes (113–563 tokens), max sequence 1024.
- 256 disjoint held-out rows for validation.

## Run

```bash
python prep_autotrust_calibration.py          # -> corpus/autotrust-calibration-v1.jsonl
python run_awq_autotrust.py                   # -> /srv/models/autotrust-jev-27b-v1.1-awq-g64
python prep_autotrust_probe.py --rows 16      # token ids + code ids
python eval_autotrust_bf16.py --rows 16       # BF16 readout (CPU)
python eval_autotrust_hf_quant.py --rows 16   # same checkpoint through HF
# native engine readout via native/tests/autotrust_probe.rs (decision_logits)
python compare_autotrust.py                   # BF16 / HF-quant / engine
```

The quantizer post-processing splices the BF16 vision tower, keeps `lm_head`
BF16 and indexed, copies the tokenizer/chat/readout files, and writes an engine
head (`decision_head/decision_head.json`, `calibration.json`) with the 255 code
ids, zero bias, and T=1.0.

## Validation

16 held-out rows, causal forward:

| implementation | top-1 | max Δ vs BF16 | MAE | KL |
|---|---:|---:|---:|---:|
| HF transformers on quantized dir | **15/16** | **0.0587** | **0.0197** | **0.0026** |
| Native engine on quantized dir | 10/16 | 0.392 | 0.0981 | 0.0692 |

The quantization is validated; the native engine's forward for the Qwen3.8
weights diverges from HF on the same checkpoint (engine vs HF-quant: 11/16,
max 0.386, KL 0.0665). The engine's `decision_logits` already returns the code
rows, but faithful AutoTrust serving additionally needs bidirectional prompt
attention and the chat-template prompt; both remain engine-port items. The
native `Engine::new` probe measures the W4A16 weight-only path (the INT8
activation setters are server-side).

## Hardware

GPU1 (24 GB) for the engine probe; host CPU (32 cores, 122 GB RAM) for the BF16
and HF-quant references. The quantized checkpoint is ~18 GB.

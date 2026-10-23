# GLiDE quantization

Source: [`fastino/GLiDE`](https://huggingface.co/fastino/GLiDE), a Fastino
decision model built on a Qwen3.5 27B language model plus a 0.92 GB vision
encoder and a 255-way decision readout. The BF16 package is 52.2 GB and needs
at least an 80 GB GPU for unquantized serving.

## Artifacts

| path | recipe | top-1 on 32 held-out rows | mean MAE | max Δp |
|---|---|---:|---:|---:|
| `/srv/models/GLiDE-w4a8-gptq` | GPTQ only | **31/32** | **0.00030** | **0.110** |
| `/srv/models/GLiDE-w4a8-awq-gptq` | AWQ + GPTQ | 23/32 | 0.0020 | 0.481 |

Both are **W4A8**: INT4 group-64 symmetric weights, INT8 dynamic per-token
activations, compressed-tensors `pack-quantized`. The vision encoder, LM head,
and 255-way readout remain BF16. Aux GLiDE files (`glide/`, tokenizer,
`decision_config.json`, `readout.safetensors`, preprocessor configs, licenses)
are copied into each package.

The quality gate is evaluated on 32 rows from `test.jsonl`, not used for
calibration. The metric reflects the saved quantized weights; activation
quantization is configured in the artifact but the in-process post-save eval
runs after compressed-tensors restores the packed weights, so it does not
exercise the runtime INT8 activation hooks. Treat it as a weight-quantization
gate, not an end-to-end W4A8 serving benchmark.

AWQ+GPTQ is retained because it was the requested recipe, but it fails the
90% top-1 gate on this model. GPTQ-only is the quality winner, consistent with
the earlier d3-turbo finding.

## Calibration

`quant/glide/prepare_calibration.py` samples 1,024 rows from the raw decision
training corpus (`calibration.jsonl`, 13,766 rows; stratified 534 `noul`, 236
`score`, 254 `choice`) and renders them with GLiDE's own `glide.Prompter`.
Token lengths are 91-687, under the 2,048 cutoff. The corpus hash and sampling
seed are deterministic.

```bash
python3 quant/glide/prepare_calibration.py \
  --model /srv/models/GLiDE \
  --input /srv/models/jev-awq/corpus/calibration.jsonl \
  --out /srv/models/glide-calibration-1024.jsonl \
  --samples 1024 --max-tokens 2048
```

Quantization was run with the **sequential** pipeline to fit the 52.2 GB model
through CPU offload; the basic pipeline OOM'd during GPTQ on an 80 GB A100.
The A100 instance was destroyed after the artifacts were copied locally.

## Engine status

GLiDE is **not yet supported by the Decidra native serving engine**. The engine
currently implements the d3-turbo `prefix_bidir` contract and its F32
code-readout; GLiDE uses non-causal full attention over each prompt, a 255-way
readout, and an optional vision encoder. Serving it requires a separate
GLiDE backend/prompt contract. The quantized artifacts are ready for that
integration; they are not a drop-in replacement for the d3-turbo checkpoint.

The source model is under the Fastino Community license; see the `LICENSE` and
`licenses/` files copied into each artifact. No model weights are committed to
this repository.

# GLiDE W4A8 quantization

INT4 group-64 weights with INT8 per-token activations, using **AWQ smoothing
followed by GPTQ**. Calibration is the raw decision training corpus
(`calibration.jsonl`), rendered with GLiDE's own `glide.Prompter`; it is not
generic web text. The vision encoder, the 255-way decision readout, and the LM
head stay BF16.

```bash
# 1. Prepare tokenized prompts from the raw decision corpus.
python3 quant/glide/prepare_calibration.py \
  --model /srv/models/GLiDE \
  --input /srv/models/jev-awq/corpus/calibration.jsonl \
  --out /srv/models/glide-calibration-1024.jsonl \
  --samples 1024 --max-tokens 2048

# 2. AWQ + GPTQ. On an 80 GB GPU pipeline=basic; for a 24 GB GPU use
#    --pipeline sequential --max-memory 20GiB and a large CPU offload budget.
python3 quant/glide/quantize.py \
  --model /srv/models/GLiDE \
  --calib /srv/models/glide-calibration-1024.jsonl \
  --out /srv/models/GLiDE-w4a8-awq-gptq \
  --samples 1024 --group 64 --pipeline basic --max-memory 74GiB \
  --eval /srv/models/jev-awq/corpus/test.jsonl \
  --eval-out /srv/models/GLiDE-eval.json
```

The script records BF16-versus-quantized probabilities on 32 held-out rows
before saving. On GLiDE (1,024 training-corpus rows, 32 held-out test rows):

| recipe | top-1 | mean MAE | max Δp |
|---|---:|---:|---:|
| AWQ + GPTQ | 23/32 | 0.0020 | 0.481 |
| GPTQ only | **31/32** | **0.00030** | **0.110** |

AWQ+GPTQ is retained because it is the requested recipe and the artifact is
complete, but it fails the 90% top-1 gate. GPTQ-only is the quality winner on
the same calibration data, consistent with the earlier d3-turbo result. Both
artifacts were produced with the sequential pipeline on an A100 80 GB; the
basic pipeline OOM'd because the full 52.2 GB model plus AWQ state exceeded
80 GB. The artifact includes GLiDE's `glide/`, `decision_config.json`,
`readout.safetensors`, tokenizer, license, and model config. `vision`,
`lm_head`, and `readout` are excluded from quantization.

Note: the d3-turbo W4A8 recipe found AWQ+GPTQ less accurate than GPTQ-only.
GLiDE is being quantized with AWQ+GPTQ as requested; the eval JSON is the
quality gate, not an assumption that the recipe is better.

# d3-turbo quantization

GPTQ W4A8 group-64 quantization for the SM89 package.

```bash
python3 quant/d3turbo/quantize.py --help
python3 quant/d3turbo/verify_quant.py --quant /srv/models/d3-turbo-w4a8-vision-v2 --group 64
```

`prep_calibration.py` builds calibration data; `strip_lm_head.py` removes the
unused LM head. The published v2 artifact is a 16.4 GB W4A8 language package
with the BF16 vision tower, d3-turbo readout, tokenizer, and `prefix_bidir`
decision config.

The shipped recipe runs GPTQ with the 256-row v4 block calibration, sequential
pipeline, `--max-seq 2048`, and `--max-memory 18GiB`; the quantizer's `post()`
step merges the source `visual.*` tensors in BF16. The resulting 32-row text
gate is 32/32 and the image eval matches BF16 13/13.

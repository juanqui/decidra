# d3-turbo quantization

GPTQ W4A8 group-64 quantization for the SM89 package.

```bash
python3 quant/d3turbo/quantize.py --help
python3 quant/d3turbo/verify_quant.py --quant /srv/models/d3-turbo-w4a8-g64-v4 --group 64
```

`prep_calibration.py` builds calibration data; `strip_lm_head.py` removes the
unused LM head. The published artifact is a 15 GB W4A8 g64 package with the
d3-turbo readout, tokenizer, and `prefix_bidir` decision config.

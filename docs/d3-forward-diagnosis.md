# d3-turbo W4A8 engine: forward is broken

The engine's d3-turbo final-norm hidden at the last token is **uncorrelated** with
the HF backbone hidden on the same tokens:

- hidden relL2 mean 1.458 (max 1.472)
- hidden cosine mean **0.0107** (min 0.0058)
- applying the same `readout.safetensors` and temperature to both: engine probabilities
  are near-uniform (0.31–0.37 / 0.14–0.20 / 0.29–0.32 / 0.14–0.18) and do not track the
  reference, while the HF hidden + the same readout does track the reference ordering.

Conclusion: the readout head is fine; the engine's d3 forward is wrong. The d3's
`prefix_bidir` path (`forward_split` / `BlockCascade`) and the plain noncausal path both
produce the same near-uniform result, so the mask alone is not the cause.

Candidates, in order:

1. The d3 checkpoint uses `model.layers.*` keys (AutoTrust uses
   `model.language_model.layers.*`). The engine's store prefix/layer mapping for the d3
   may not match, silently leaving layers at wrong/zero weights.
2. The d3's W4A8 `weight_packed` maybe loaded with the wrong per-module scheme (496 packed
   modules on all three, so structure matches AutoTrust, which passes parity).
3. The d3's GDN/full layer map or the split path's state/question block handling.

Evidence files: `/tmp/d3_engine_hidden.json`, `/tmp/d3_hf_hidden.json`,
`/tmp/d3_probe.json` (now carries `split`), `/tmp/d3_hf_eval.json`.

The next step is a per-layer hidden comparison (engine vs HF) to find the first layer
where the cosine collapses, then inspect that layer's weight load and mask.

## Per-layer teacher-forced result (first probe row)

Inputs are the HF backbone's layer inputs; the engine runs one layer with the split path.

| layer | relL2 vs HF next-layer input | cosine |
|---|---|---|
| 0 | 16.57 | 0.0015 |
| 1 | 18.30 | 0.0002 |
| 2 | 21.43 | 0.0003 |
| 3 | 14.16 | -0.0005 |
| 7 | 11.05 | -0.0001 |
| 15 | 11.77 | 0.0004 |
| 31 | 13.41 | 0.0014 |
| 47 | 5.01 | -0.0015 |

The collapse is at **layer 0**: the engine's first GDN layer output is already
uncorrelated with the HF's. AutoTrust's identical layer 0 passes at 1.5–4.5%, so this is
not the GDN kernel in general. The difference is the d3 checkpoint's key layout
(`model.layers.*` vs AutoTrust's `model.language_model.layers.*`) or the per-module W4A8
scheme attached to those names; both must be verified against the checkpoint.

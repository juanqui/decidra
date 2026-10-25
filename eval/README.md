# d3-turbo evaluation

- `d3_compare.py` compares engine logits with the checkpoint's BF16 reference
  on the 32-row probe (`data/d3_probe_split32.json`, `data/d3_hf_eval.json`).
- `d3_bench.py` measures `/v1/systemone` latency/throughput against Decidra or
  vLLM-SR engine mode using `data/d3_ticket_2k_64q.json`.
- `data/d3_once_*.json` are state-once probes for the native tests.

```bash
python3 eval/d3_bench.py --url http://127.0.0.1:8899 \
  --request eval/data/d3_ticket_2k_64q.json \
  --questions 1,8,64 --warmup 2 --runs 10 --concurrency 1,8
```

`requirements.txt` is intentionally empty: both tools use the standard library.

- `image_api.py` smoke-tests the live image API (PNG/JPEG/WebP, edge
  resolutions, RGBA/L/P, two images, malformed data, and image-cache identity).

# Decidra

**Read the state once. Answer every question.**

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-brightgreen.svg)](https://github.com/juanqui/decidra/releases)
[![Container](https://img.shields.io/badge/container-ghcr.io%2Fjuanqui%2Fdecidra-2496ED?logo=docker&logoColor=white)](https://github.com/juanqui/decidra/pkgs/container/decidra)

Decidra is a purpose-built CUDA inference engine for exactly one model:
**`vllm-sr/d3-turbo`**, a 26.1B Qwen3.5 hybrid decision model with the
`prefix_bidir` attention mask and a separate F32 `[255, 5120]` readout.

It is not a general model runner. The HTTP server and the CUDA engine are one
Rust process; there is no Python, PyTorch, vLLM, or external model server on the
runtime path. A state is encoded once, then every question forks from that
cache. On identical hardware and the same checkpoint, Decidra answers the
64-question benchmark **2.4–5.2× faster** than vLLM-SR, with up to **5.2×
higher 64-question throughput** at c8. The full side-by-side is below.

---

## Why state-once

`prefix_bidir` splits a rendered prompt into a state block and question blocks:

```text
[ system + State: … ][ Question: … options … ][ readout ]
        state S                 question Q
```

- every state query sees every state key, in both directions;
- every question query sees every state key and every key in its own block;
- **no state query sees a question key.**

The state’s hidden states, post-RoPE full-attention K/V, GDN recurrent state,
and convolution tail are therefore question-independent. Decidra captures them
once and serves every question with one ragged pass and one readout GEMM.

```text
state ──► capture once ──► StateCache ──┬─ full layers: post-RoPE K/V
                                        └─ GDN layers: f32 state + conv tail
questions ──► one ragged pass ──► [B, 255] readout ──► typed answers
```

## Quick start

### Requirements

- Linux x86_64, NVIDIA driver with CUDA 13 support.
- RTX 4090 / L40S (SM89), H100 / H200 (SM90), or RTX PRO 6000 Blackwell
  (SM120).
- The d3-turbo package mounted read-only. It must contain `config.json`,
  `decision_config.json` (`attention_mode: prefix_bidir`,
  `format_id: d3-code-readout-v1`), `tokenizer.json`, `model.safetensors`, and
  `readout.safetensors`. Image requests additionally require the BF16
  `visual.*` tensors, `preprocessor_config.json`, and the vision config in
  `config.json`; the published vision artifact includes all of them.
- 24 GB VRAM for the W4A8 vision package (W4A8 language + BF16 image tower)
  on a 4090; 49 GB BF16 on H100/H200; 96 GB on the RTX PRO 6000, which fits
  either BF16 or W4A8 plus state caches.

### Docker

Published images are built per GPU architecture. The binary embeds one cubin
and refuses to start on a different SM family, so select the tag for the target
GPU:

| GPU | moving tag | version tag |
|---|---|---|
| RTX 4090 / L40S (SM89) | `ghcr.io/juanqui/decidra:sm89-latest` | `ghcr.io/juanqui/decidra:sm89-0.2.0` |
| H100 / H200 (SM90) | `ghcr.io/juanqui/decidra:sm90-latest` | `ghcr.io/juanqui/decidra:sm90-0.2.0` |
| RTX PRO 6000 Blackwell (SM120) | `ghcr.io/juanqui/decidra:sm120-latest` | `ghcr.io/juanqui/decidra:sm120-0.2.0` |

The unsuffixed `latest`, `0.2.0`, and commit-sha tags remain SM89 aliases.
Verify a pulled image before deploying:

```bash
docker image inspect ghcr.io/juanqui/decidra:sm90-latest \
  --format '{{ index .Config.Labels "ai.decidra.cuda-arch" }}'
# sm_90
```

```bash
docker run --rm --gpus '"device=0"' \
  --user "$(id -u):$(id -g)" \
  -p 8080:8080 \
  -v /srv/models/d3-turbo-w4a8-vision-v2:/models:ro \
  ghcr.io/juanqui/decidra:sm89-latest
```

```bash
curl -fsS http://127.0.0.1:8080/health
# {"status":"ready","model":"vllm-sr/d3-turbo","head":"d3-code-readout"}
```

Build a Hopper or Blackwell image from source:

```bash
docker build --build-arg CUDA_ARCH=sm_90  -t decidra:sm90-0.2.0 .
docker build --build-arg CUDA_ARCH=sm_120 -t decidra:sm120-0.2.0 .
```

Or use the provided compose file:

```bash
DECIDRA_MODEL_DIR=/srv/models/d3-turbo-w4a8-vision-v2 \
DECIDRA_GPU=0 docker compose up -d --build
```

Compose defaults to the SM89 image. For another architecture, select both the
build arch and the matching tag:

```bash
DECIDRA_CUDA_ARCH=sm_90 DECIDRA_IMAGE_TAG=sm90-0.2.0 \
DECIDRA_MODEL_DIR=/srv/models/d3-turbo \
DECIDRA_GPU=0 docker compose up -d --build
```

### Native build

```bash
# RTX 4090 / L40S
cargo build --release --locked --features native

# H100 / H200
DECIDRA_CUDA_ARCH=sm_90 cargo build --release --locked --features native

# RTX PRO 6000 Blackwell (SM120)
DECIDRA_CUDA_ARCH=sm_120 cargo build --release --locked --features native

CUDA_VISIBLE_DEVICES=0 \
DECIDRA_CHECKPOINT=/srv/models/d3-turbo-w4a8-vision-v2 \
DECIDRA_NATIVE_MODEL=/srv/models/d3-turbo-w4a8-vision-v2 \
./target/release/decidra
```

Validate a package without a GPU:

```bash
./target/release/decidra --check-config
```

Run the state-once startup gate before serving:

```bash
./target/release/decidra --native-check
```

## API

`POST /v1/systemone` accepts a state string, a map of typed questions, and
images either as an optional top-level `images` list or as OpenRouter-style
`image_url` content parts inside a `state` array.

```bash
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "The order arrived damaged yesterday. The customer has a receipt.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "returns": "Refunds, replacements and damaged deliveries",
          "billing": "Payments, invoices and charges",
          "technical": "Product setup and faults"
        }
      },
      "receipt": {
        "type": "noul",
        "instructions": "Does the customer have a receipt?"
      }
    }
  }'
```

```json
{
  "object": "structured_decision",
  "model": "vllm-sr/d3-turbo",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "returns",
      "probabilities": { "returns": 0.99996, "billing": 0.000018, "technical": 0.000021 },
      "confidence": 0.99994
    },
    "receipt": { "type": "noul", "noul": 0.99981 }
  },
  "usage": { "input_tokens": 168, "output_tokens": 0 }
}
```

Question types: `choice`, `noul`, and `score` (2–10 ordered levels). Scores
return `score`, `probabilities`, `confidence`, and `legend`.

Images are shared by every question in the request. The direct form uses a
top-level list:

```json
{
  "state": "An image from a catalogue.",
  "questions": {
    "object": {
      "type": "choice",
      "instructions": "What is in the image?",
      "criteria": { "panda": "a panda", "box": "a cardboard box" }
    }
  },
  "images": ["data:image/png;base64,iVBORw0KGgo..."]
}
```

The controller/OpenRouter form puts the image in the state content array;
the engine hoists it, concatenates the text parts, and sends the image to the
vision tower:

```json
{
  "state": [
    { "type": "text", "text": "An image from a catalogue." },
    { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
  ],
  "questions": { "object": { "type": "noul", "instructions": "Is the object held?" } }
}
```

Both forms can be combined; top-level images precede state content images.
`video_url` content is rejected with HTTP 400.

Accepted formats are PNG, JPEG and WebP data URLs, up to 8 MB encoded, 16 MP
source pixels, aspect ratio 200, 16 images, and 8,192 expanded vision tokens
per request. Up to 64 image decode/tokenize tasks run concurrently, and up to
128 HTTP requests are admitted; excess image requests wait for a decode or CUDA
queue slot instead of being refused. 128 concurrent distinct image requests
were verified end-to-end. The runtime converts to RGB (alpha is
dropped, EXIF orientation is not applied), applies the Qwen2VL `smart_resize`
and bicubic resize, and uses the model's mRoPE image positions. Image states
are cached by the decoded image identity plus state tokens, so the same text
with a different image never reuses K/V state.

| endpoint | purpose |
|---|---|
| `POST /v1/systemone` | typed decisions |
| `GET /health` | readiness and model identity |
| `GET /v1/models` | model metadata |
| `GET /metrics` | Prometheus metrics |

Limits: 64 questions per request, 32,768 tokens per question’s state + block,
1 MiB rendered state and 1 MiB per question; 64 MiB request body. Image
requests add the expanded vision tokens to the state block.

## Performance: Decidra vs vLLM-SR

End-to-end `POST /v1/systemone`, warm weights, same checkpoint, same GPU, same
request: a 2,076-token state and 64 questions. The baseline is **vLLM-SR
`vllm-srun` 0.2.0 engine mode** (not the generic vLLM server) with
`--max-num-seqs 64 --enable-shared-context`. A `c8` row means 8 simultaneous
requests; `q/s` means questions per second, not requests per second.

| GPU | package | test | Decidra | vLLM-SR | advantage |
|---|---|---|---:|---:|---|
| RTX 4090 (450 W) | W4A8 g64 | 64q, c1 | **936 ms** | **3.89 s** | **4.2× lower latency** |
| RTX 4090 (600 W) | W4A8 vision | 80×60 PNG image, c1 | **88 ms cold / 34 ms cached** | — | — |
| H100 SXM 80 GB | BF16 | 64q, c1 | **824 ms** | **2.43 s** | **2.9× lower latency** |
| H100 SXM 80 GB | BF16 | 64q, c8 | **5.10 s / 100 q/s** | **13.66 s / 23 q/s** | **2.7× lower latency, 4.3× q/s** |
| RTX PRO 6000, 600 W | BF16 | 64q, c1 | **775 ms** | **3.23 s** | **4.2× lower latency** |
| RTX PRO 6000, 600 W | BF16 | 64q, c8 | **3.42 s / 84 q/s** | **14.57 s / 20 q/s** | **4.3× q/s** |
| RTX PRO 6000, 600 W | **FP8** | 64q, c1 | **623 ms** | **3.23 s** | **5.2× lower latency** |
| RTX PRO 6000, 600 W | **FP8** | 64q, c8 | **2.80 s / 103 q/s** | **14.57 s / 20 q/s** | **5.2× q/s** |
| H200 | BF16 | 64q, c1 | **800 ms** | **2.90 s** | **3.6× lower latency** |
| H200 | BF16 | 64q, c8 | **5.47 s / 94 q/s** | **13.22 s / 22 q/s** | **2.4× lower latency, 4.3× q/s** |
| H200 | BF16 | 16q, c8 | **1.32 s / 97 q/s** | **4.30 s / 17 q/s** | **3.3× lower latency, 5.7× q/s** |

The 4090 vLLM-SR comparison is c1 only; a c8 baseline was not measured, so no
4090 throughput speedup is claimed. On the same 4090 the two paths chose the
same top answer and differed by at most **0.005 probability** on the repeated
question. vLLM-SR loads the W4A8 artifact as dequantized W4A16; Decidra serves
it as W4A8 language weights with per-64-group INT8 activation scales; the
vision tower stays BF16.

### Decidra-only scaling

| GPU | 64q c1 | 64q c8 | 16q c8 |
|---|---:|---:|---:|
| RTX 4090, 450 W | **936 ms** | **4.18 s / 69 q/s** | **1.15 s / 69 q/s** |
| RTX PRO 6000, 600 W BF16 | **775 ms** | **3.42 s / 84 q/s** | **976 ms / 82 q/s** |
| RTX PRO 6000, 600 W FP8 | **623 ms** | **2.80 s / 103 q/s** | **754 ms / 106 q/s** |
| RTX PRO 6000 Max-Q, 300 W BF16 | **1.045 s** | **4.70 s / 61 q/s** | — |
| RTX PRO 6000 Max-Q, 300 W FP8 | **859 ms** | **3.84 s / 75 q/s** | **1.06 s / 75 q/s** |
| H100 SXM 80 GB | **824 ms** | **5.10 s / 100 q/s** | **1.28 s / 100 q/s** |
| H200 | **800 ms** | **5.47 s / 94 q/s** | **1.32 s / 97 q/s** |

RTX PRO 6000 defaults to BF16 because it has the tightest state-once parity
(**0.0102**). The opt-in SM120 FP8 path is **1.24× faster** again
(**623 ms / 103 q/s**) and answers the same request **5.2×** faster than the
vLLM-SR `vllm-srun` runtime on the same host. FP8 is approximate: the measured
single-vs-batch probability delta is **0.036**, and the top choice matched on
**64/64** probe rows. State capture stays on BF16, so FP8 only affects the question
GEMMs. On Max-Q (300 W) the same FP8 path is **859 ms / 74 q/s** versus
**61 q/s** BF16; the 4090 W4A8 reference is **933 ms / 69 q/s** at 450 W.
The 547 W WS result is not a hardware comparison: that instance ran at
1695 MHz and 376 W, while the 600 W host sustained 2445–2482 MHz and ~605 W.
On the 4090: 1q **43 ms** warm, 8K/64q **1.02 s**, 24K/64q **1.29 s**. The
16-entry pinned GPU state cache plus an 8 GiB host L2 serves **16 states at
69 q/s** and **32 states at 63 q/s**; long states are protected by a 2 GiB
scratch reserve. On H200, long-state 64q is **1.27 s at 8K** and **2.62 s at
24K**, versus **4.39 s** and **9.45 s** for vLLM-SR.

Full matrices, profiles, and the method are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The requested **127 q/s** target was
**not reached**; the best measured result is **100 q/s on H100 SXM** at c8.

## Configuration

The image and compose file carry production defaults. The most important knobs:

| variable | default | meaning |
|---|---|---|
| `DECIDRA_BIND_ADDR` | `0.0.0.0:8080` | HTTP bind |
| `DECIDRA_CHECKPOINT` | `/srv/models/d3-turbo-w4a8-vision-v2` | model package |
| `DECIDRA_IMAGE_DECODE_CONCURRENCY` | `64` | image decode slots; excess waits |
| in-flight request admission | `128` | bounded outer admission for text and images |
| `DECIDRA_NATIVE_MODEL` | `$DECIDRA_CHECKPOINT` | engine package |
| `DECIDRA_NATIVE_MARLIN` | `true` | Marlin W4A8 |
| `DECIDRA_NATIVE_INT8_EXCLUDE` | `self_attn.v_proj,self_attn.k_proj` | modules kept on W4A16 |
| `DECIDRA_STATE_CACHE_ENTRIES` | `16` | pinned exact-state entries |
| `DECIDRA_STATE_CACHE_BYTES` | `5368709120` | GPU state K/V/GDN budget |
| `DECIDRA_HOST_STATE_CACHE_BYTES` | `8589934592` | host L2 for overflow states |
| `DECIDRA_BATCH_ROWS` | `32` | question-row budget per pass |
| `DECIDRA_BATCH_WINDOW_US` | `2000` | request coalescing window |
| `DECIDRA_NATIVE_FP8` | off | SM120 approximate FP8 GEMMs |
| `DECIDRA_FP8_MODE` | `mlp` | `mlp`, `gate_up`, or `all_linear` |
| `DECIDRA_FP8_MIN_ROWS` | `512` | FP8 question-row threshold; capture stays BF16 |
| `DECIDRA_METRICS_ENABLED` | `true` | Prometheus `/metrics` |

See [docs/DEPLOY.md](docs/DEPLOY.md) for the full environment reference, H100
and RTX PRO 6000 settings, and operational notes.

## Tests

```bash
# CPU server tests
cargo test --locked -p decidra

# Formatting and build
cargo fmt --all -- --check
cargo check --features native

# Native state-once, parity, negative controls, perf
make native-check
```

The 32-row HF parity gate, state-once fork parity, batch-vs-single comparison,
and seed/tail negative controls are documented with exact commands in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#correctness-gates).

## Documentation

| document | contents |
|---|---|
| [docs/ENGINE.md](docs/ENGINE.md) | mask semantics, state cache, kernels, correctness |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | full matrices, profiles, correctness gates |
| [docs/DEPLOY.md](docs/DEPLOY.md) | build, Docker, operations, environment reference |
| [docs/README.md](docs/README.md) | documentation index |
| [native/README.md](native/README.md) | engine crate and execution contract |
| [eval/README.md](eval/README.md) | parity and benchmark tools |
| [quant/d3turbo/](quant/d3turbo/) | W4A8 g64 quantization pipeline |

## Repository layout

```text
src/             HTTP API, d3 prompt contract, state-once scheduler
native/          CUDA engine, Marlin, GDN, FlashInfer, F32 readout
quant/d3turbo/   GPTQ W4A8 group-64 quantization
eval/            HF comparison and serving benchmarks
docs/            architecture, benchmarks, deployment
scripts/         local smoke and launch helpers
```

## License

Decidra is dual-licensed under **MIT OR Apache-2.0**. See [`LICENSE`](LICENSE),
[`LICENSE-MIT`](LICENSE-MIT), and [`LICENSE-APACHE`](LICENSE-APACHE).

The `vllm-sr/d3-turbo` checkpoint is Apache-2.0. Vendored FlashInfer and Marlin
sources are Apache-2.0; vendored NVIDIA CCCL headers are Apache-2.0 with LLVM
exceptions. See [`NOTICE`](NOTICE) and `native/third_party/`.

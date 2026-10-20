# Decidra

Decidra is a native inference engine for decision models. It runs one model on
one NVIDIA GPU and serves typed decisions over HTTP. The HTTP server and the
GPU engine are one process. There is no Python at serving time.

- Native CUDA backend: Marlin W4A16 and W4A8-INT8, FlashInfer attention,
  chunked GDN prefill, device-resident embedding table.
- Typed decisions: `noul` (binary), `score` (ordinal), `choice` (multi-option).
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` for LiteLLM.
- Exact-decision cache, request batching, Prometheus metrics, optional request
  signing.
- Self-contained binaries: kernel cubins and the FlashInfer prefill library are
  embedded at build time.

## Supported models

Decidra is built for specific decision checkpoints. It is not a general-purpose
model runner. Two JEV-27B-VL quantized artifacts are supported:

| artifact | weights | activations | decision head | status |
|---|---|---|---|---|
| [`aioutfitters/JEV-27B-VL-AWQ-INT4-W4A8`](https://huggingface.co/aioutfitters/JEV-27B-VL-AWQ-INT4-W4A8) | INT4, group **128**, symmetric | INT8 **per-token dynamic** (`CompressedTensorsWNA8Int`) | rank-32 **LoRA** on all linears + `lm_head` | **supported** |
| `awq-w4a16-v2` build (local checkpoint) | INT4, group **64**, asymmetric | engine applies INT8 at runtime (group-64 / per-token) | **merged** decoder + BF16 head | **supported** |
| [`autotrust/JEV-27B-VL`](https://huggingface.co/autotrust/JEV-27B-VL) | BF16 base | — | LoRA / merged variants | source model; does not fit one 24 GB GPU |

### W4A8 checkpoint

This is the artifact at the link above. The engine loads the group-128
symmetric INT4 weights (Marlin group-8 kernels, offset zero point 8), quantizes
activations to INT8 per token, and applies the PEFT rank-32 LoRA from
`adapter_vllm/` to every target linear and `lm_head`. The LoRA scale is read
from `adapter_config.json` (`64/32 = 2.0`).

```bash
export DECIDRA_CHECKPOINT=/path/to/JEV-27B-VL-AWQ-INT4-W4A8
export DECIDRA_NATIVE_MODEL="$DECIDRA_CHECKPOINT"
export DECIDRA_TOKENIZER="$DECIDRA_CHECKPOINT/tokenizer.json"
export DECIDRA_LORA_ADAPTER="$DECIDRA_CHECKPOINT/adapter_vllm"
export DECIDRA_NATIVE_INT8_EXCLUDE=none
export DECIDRA_NATIVE_INT8_GROUP_SCALE=false
export DECIDRA_NATIVE_GPU_EMBED=false   # 21.2 GiB; drop for ~3% more speed (23.3 GiB)
./target/release/decidra
```

Required files for this artifact:

```
config.json          model architecture + group-128 symmetric W4A8 config
tokenizer.json       tokenizer
decision_head/       decision_head.json + calibration.json
model.safetensors    compressed-tensors W4A8 text weights
adapter_vllm/        adapter_config.json + adapter_model.safetensors (rank 32)
```

### W4A16 checkpoint

The local `awq-w4a16-v2` build. It uses the engine's original runtime INT8
activation path (group-64 / per-token); this is the artifact behind the
**W4A16** tables in Performance and Quality below.

```
config.json          model architecture + group-64 asymmetric config
tokenizer.json       tokenizer
decision_head/       merged decision_head.json + calibration.json
model*.safetensors   AWQ W4A16 weights
```

The native backend serves **text decision requests**; the checkpoint's vision
tower is not used. Other decision models need a backend and checkpoint port; the
engine does not load them by configuration alone.

## Hardware

Tested on one **RTX 4090 (Ada, SM89), 24 GB**, with a CUDA 13.x driver.

The server, API, decision head, and model contract are GPU-agnostic. The native
kernels are compiled for **SM89 only** today. **RTX Pro 6000 (Blackwell, SM120)
is not supported yet.** Supporting it is a backend port: the Marlin/INT8 GEMMs,
FlashInfer prefill, GDN kernels, and the cubin build need SM120 versions. The
current Docker image will not run on SM120.

## Performance

**Artifacts:** the W4A16 tables use the group-64 merged checkpoint with the
engine's runtime INT8 path. The W4A8 tables use the HF W4A8 checkpoint + LoRA.
Same GPU, same prompts, one resident server, exact-decision cache disabled,
median of 3 runs. vLLM 0.31.0 serves the same checkpoint in each comparison.

### W4A16 checkpoint

Short unique prompts (~73 tokens):

| concurrency | Decidra req/s | vLLM req/s | Decidra p50 / p95 | vLLM p50 / p95 |
|---:|---:|---:|---|---|
| 1 | **25.78** | 18.81 | **39.2 / 42.4 ms** | 53.06 / 54.08 ms |
| 2 | **40.73** | 20.24 | **49.1 / 51.8 ms** | 98.20 / 103.24 ms |
| 4 | **53.10** | 29.32 | **74.4 / 75.6 ms** | 135.10 / 138.67 ms |
| 8 | **56.97** | 34.52 | **123.2 / 168.0 ms** | 229.98 / 231.08 ms |
| 16 | **56.93** | 35.17 | **173.1 / 275.7 ms** | 304.75 / 454.23 ms |

Single cold long request:

| prompt | Decidra | vLLM | Decidra tok/s | vLLM tok/s |
|---|---:|---:|---:|---:|
| 4K (4,090 tokens) | **0.833 s** | 1.302 s | 4,908 | 3,142 |
| 8K (8,178 tokens) | **1.711 s** | 2.660 s | 4,779 | 3,075 |
| 16K (16,383 tokens) | **3.740 s** | 5.533 s | 4,380 | 2,962 |

Memory and startup:

| metric | Decidra | vLLM 0.31.0 |
|---|---:|---:|
| resident GPU memory | **18.83 GiB** (13.1 GiB weights + 2.37 GiB embedding + scratch) | 21,909 MiB |
| 16K prefill peak | **20.11 GiB** | ~21.9 GiB |
| startup to ready | **9.2 s** | 145 s |

### W4A8 checkpoint

Engine uses the host embedding table (21.2 GiB peak). vLLM serves the same W4A8
artifact and adapter with `--max-num-seqs 8` and fp8 KV.

Short unique prompts (~73 tokens):

| concurrency | Decidra req/s | vLLM req/s | Decidra p50 | vLLM p50 |
|---:|---:|---:|---:|---:|
| 1 | **24.2** | 11.05 | **41.6 ms** | 83.2 ms |
| 2 | **38.3** | 14.76 | **52.4 ms** | 159.7 ms |
| 4 | **53.6** | 22.50 | **73.9 ms** | 174.4 ms |
| 8 | **56.8** | 25.91 | **122.6 ms** | 273.4 ms |
| 16 | **56.6** | 22.82 | **188.0 ms** | 494.1 ms |

Single cold long request:

| prompt | Decidra | vLLM | speedup |
|---|---:|---:|---:|
| 4K (4,090 tokens) | **0.903 s** | 1.386 s | **1.53×** |
| 8K (8,178 tokens) | **1.846 s** | 2.308 s | **1.25×** |
| 16K (16,383 tokens) | **4.159 s** | 4.813 s | **1.16×** |

Memory: **21.2 GiB** peak for the engine vs **21.0 GiB** for vLLM. The GPU
embedding table variant is 23.3 GiB and about 3% faster on long prompts.

Long prefill runs at 100% GPU utilization and 420–430 W. Short batching reaches
90–96% utilization. The GPU is the limit at every measured point.

Full method and per-request tables: [`docs/VLLM-COMPARISON.md`](docs/VLLM-COMPARISON.md).
Kernel-level numbers: [`docs/KERNELS.md`](docs/KERNELS.md).

## Quality

### W4A16 checkpoint

28-row suite against the CPU AWQ golden (`eval/data/awq-cpu-suite28.json`),
batch size 1:

| implementation | top-1 | max Δ | mean MAE | mean KL | gates |
|---|---:|---:|---:|---:|---|
| Decidra | **28/28** | **0.01027** | **0.000717** | **3.27e-5** | **pass** |
| vLLM 0.31.0 | 28/28 | 0.01377 | 0.00155 | 8.51e-5 | max and MAE fail |

Gates: top-1 `1.0`, mean KL ≤ `1e-3`, mean MAE ≤ `1e-3`, max Δ ≤ `0.011`.

Batched requests have deterministic BF16/M-tiling composition noise. Top-1 was
28/28 for the suite at c8 and c16; maximum deltas were `1.305e-2` (c8) and
`1.251e-2` (c16). The strict gates apply at batch size 1.

### W4A8 checkpoint

28-row suite, three implementations of the same W4A8 artifact + adapter against the
CPU artifact-math golden (`eval/data/w4a8-cpu-suite28.json`: compressed-tensors
decompress + LoRA + the configured per-token INT8):

| comparison | top-1 | max Δ | mean MAE | mean KL |
|---|---:|---:|---:|---:|
| Decidra vs CPU artifact | **28/28** | **0.0176** | **0.00328** | **2.33e-4** |
| vLLM 0.31.0 vs CPU artifact | 28/28 | 0.0267 | 0.00349 | 3.04e-4 |
| Decidra vs vLLM | 28/28 | 0.0304 | 0.00266 | 2.20e-4 |

Decidra is closer to the artifact math than vLLM 0.31.0 on max Δ and KL. On the card
example (billing) the engine gives 0.9606, the CPU artifact 0.9623, the local vLLM
0.9588. vLLM's W4A8 path behaves as weight-only here: the engine's weight-only path
(`DECIDRA_NATIVE_INT8=false`) matches it at 28/28, max 9.77e-3, MAE 1.23e-3. The
provider card's example value (0.9963) is not reproducible with the downloaded
artifact under any local implementation; it is close to the merged-BF16 teacher
(0.9969). Against that teacher the W4A8 quantization error is max 0.258, MAE 0.0598,
KL 0.0697 on this suite (28/28 top-1; engine and vLLM within 0.014 max of each other).

Gate for this artifact: `eval/suite.py --baseline eval/data/w4a8-cpu-suite28.json
--max-delta 0.02 --max-mae 0.004 --max-kl 0.001` against a running W4A8 server.
The W4A16 CPU golden does not apply to this artifact.

## Build from source

Prerequisites:

- Linux x86_64, NVIDIA driver with CUDA 13.x support.
- CUDA toolkit 13.x (`nvcc`) for the build.
- Rust 1.85 or newer (`rustup`).
- cuBLAS and cuBLASLt at run time (CUDA runtime package or `LD_LIBRARY_PATH`).
- The AWQ checkpoint (about 18 GB) with the files listed above.

Build both binaries:

```bash
cargo build --release -p decidra-native
cargo build --release --features native -p decidra
```

`decidra-native` is the command-line engine used for kernel and quality work.
`decidra` is the HTTP server. The build compiles the CUDA kernels with `nvcc`
and embeds the cubins and the FlashInfer library into both binaries.

If `nvcc` is not on `PATH`, set `DECIDRA_NVCC=/path/to/nvcc`, or let the build
use Docker with `DECIDRA_CUDA_IMAGE` (default
`nvidia/cuda:13.1.1-devel-ubuntu24.04`).

## Run the server

```bash
export DECIDRA_CHECKPOINT=/path/to/awq-w4a16-v2
export DECIDRA_NATIVE_MODEL="$DECIDRA_CHECKPOINT"
export DECIDRA_TOKENIZER="$DECIDRA_CHECKPOINT/tokenizer.json"
export DECIDRA_BIND_ADDR=0.0.0.0:8080
export LD_LIBRARY_PATH=/path/to/cublas:$LD_LIBRARY_PATH   # if cuBLAS is not in the default path

./target/release/decidra
```

Checks:

```bash
curl -fsS http://127.0.0.1:8080/health
./target/release/decidra --check-config
./target/release/decidra --native-check   # loads the GPU engine and runs one forward
```

Example decision request:

```bash
curl -fsS http://127.0.0.1:8080/v1/decide \
  -H 'content-type: application/json' \
  -d '{"kind":"choice","state":"The customer says the charge was duplicated.","question":"Route this ticket.","options":["billing","support"]}'
```

Response (abbreviated):

```json
{
  "kind": "choice",
  "effective_kind": "choice",
  "options": ["billing", "support"],
  "probabilities": [0.976, 0.024],
  "choice": "billing",
  "choice_index": 0,
  "confidence": 0.976
}
```

## LiteLLM

The OpenAI-compatible routes accept a JSON decision request as the user message
(or prompt) and return the decision JSON. Non-JSON user text returns HTTP 400:
these routes do not do free-form generation.

```yaml
model_list:
  - model_name: jev-awq
    litellm_params:
      model: openai/jev-awq
      api_base: http://127.0.0.1:8080/v1
      api_key: unused
```

`model` is accepted and ignored by the decision routes. `/v1/models` lists the
configured model id.

## Docker

The image contains the server, the embedded kernels, and cuBLAS/cuBLASLt. It does
not contain the model, a CUDA toolkit, Python, or the FlashInfer source tree.
The image is about **1.1 GB** on disk; cuBLASLt is the largest component. Both
supported checkpoints work; the W4A8 artifact needs its `adapter_vllm/` directory
mounted and the environment below.

Build:

```bash
docker build -t decidra:0.1.0 .
```

Run on one GPU, with the checkpoint mounted read-only:

```bash
docker run --gpus '"device=1"' --rm -p 8080:8080 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/awq-w4a16-v2:/models:ro \
  decidra:0.1.0
```

The container defaults to `/models` for the checkpoint, native model, and
tokenizer. Override with `DECIDRA_CHECKPOINT`, `DECIDRA_NATIVE_MODEL`, and
`DECIDRA_TOKENIZER` if the layout differs. The image runs as UID 10001 by
default; `--user "$(id -u):$(id -g)"` is the simplest fix when the checkpoint
files are not world-readable (for example `-rw-------`). The mount must be
readable by the runtime user.

The host needs the NVIDIA driver and the NVIDIA Container Toolkit. The image
starts as UID 10001, listens on `0.0.0.0:8080`, and has a `/health` healthcheck.

For the W4A8 checkpoint, mount the artifact (which contains `adapter_vllm/`) and
set the LoRA/quantization environment:

```bash
docker run --gpus '"device=1"' --rm -p 8080:8080 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/JEV-27B-VL-AWQ-INT4-W4A8:/models:ro \
  -e DECIDRA_LORA_ADAPTER=/models/adapter_vllm \
  -e DECIDRA_NATIVE_INT8_EXCLUDE=none \
  -e DECIDRA_NATIVE_INT8_GROUP_SCALE=false \
  -e DECIDRA_NATIVE_GPU_EMBED=false \
  decidra:0.1.0
```

### GHCR

The private image is published to GitHub Container Registry. The workflow
[`.github/workflows/docker.yml`](.github/workflows/docker.yml) builds and pushes
`0.1.0`, `latest`, and the commit SHA on every push to `master`.

```bash
docker login ghcr.io
docker pull ghcr.io/juanqui/decidra:0.1.0
docker run --gpus all --rm -p 8080:8080 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/awq-w4a16-v2:/models:ro \
  ghcr.io/juanqui/decidra:0.1.0
```

`docker login ghcr.io` needs a token with `read:packages` (and `write:packages`
to push). To push a build:

```bash
docker tag decidra:0.1.0 ghcr.io/juanqui/decidra:0.1.0
docker push ghcr.io/juanqui/decidra:0.1.0
```

`docker compose up -d --build` also works; see `docker-compose.yml` for the GPU
device reservation and the model volume.

## API

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/v1/models` | model ids |
| GET | `/v1/decide/info` | decision head metadata |
| POST | `/v1/decide` | one typed decision |
| POST | `/v1/decisions` | batch of decisions |
| POST | `/v1/systemone` | System 1 decision protocol |
| POST | `/v1/chat/completions` | OpenAI chat shape, decision JSON |
| POST | `/v1/completions` | OpenAI completions shape, decision JSON |
| GET | `/metrics` | Prometheus metrics |

Decision request:

```json
{
  "kind": "noul | score | choice",
  "state": "text to judge",
  "question": "instructions",
  "options": ["labels for choice/score; omit for noul"]
}
```

`options` is required for `choice` (2–256 labels). `noul` ignores supplied
options and uses `false`/`true`. `score` uses the six labels `0`–`5`.
`strategy` is `auto`, `single`, `tournament`, or `permute`; `model` is accepted
and ignored. The protocol and response fields are specified in
[`docs/SPEC.md`](docs/SPEC.md).

## Configuration

All runtime configuration is read from the environment and validated at startup.
Booleans accept `true/false`, `1/0`, `yes/no`, `on/off`.

### Server

| variable | default | meaning |
|---|---|---|
| `DECIDRA_BIND_ADDR` | `0.0.0.0:8080` | listen address |
| `DECIDRA_MODEL` | `jev-awq` | model id returned by `/v1/models` |
| `DECIDRA_SERVED_IDENTITY` | `autotrust/JEV-27B-VL@ba3f0d58+awq-w4a16` | identity for `/v1/systemone` |
| `DECIDRA_SIGNING_KEY` | unset | enables HMAC request signing when set |
| `DECIDRA_METRICS_ENABLED` | `true` | expose `/metrics` |
| `DECIDRA_MAX_BODY_BYTES` | `67108864` | maximum request body |
| `RUST_LOG` | `decidra=info` | log filter |

### Model

| variable | default | meaning |
|---|---|---|
| `DECIDRA_CHECKPOINT` | `/srv/models/jev-awq/awq-w4a16-v2` | checkpoint directory |
| `DECIDRA_NATIVE_MODEL` | checkpoint | AWQ artifact opened by the engine |
| `DECIDRA_TOKENIZER` | checkpoint tokenizer | tokenizer path |
| `DECIDRA_CALIBRATION` | checkpoint default | calibration file |
| `DECIDRA_CALIBRATION_FITTED` | checkpoint default | fitted calibration file |
| `DECIDRA_HF_SNAPSHOT` | host snapshot path | HF snapshot used for metadata |
| `DECIDRA_LORA_ADAPTER` | unset | PEFT adapter directory or file; enables the W4A8 LoRA |

### Native engine

| variable | default | meaning |
|---|---|---|
| `DECIDRA_NATIVE_CACHE` | `true` | keep packed decoder weights on the GPU |
| `DECIDRA_NATIVE_MARLIN` | `true` | Marlin W4A16/W4A8 kernels |
| `DECIDRA_NATIVE_FUSED_W4A16` | `true` | fused fallback policy when Marlin is off |
| `DECIDRA_NATIVE_BATCH` | `true` | request batching |
| `DECIDRA_NATIVE_MAX_BATCH` | `8` | maximum requests per batch |
| `DECIDRA_NATIVE_BATCH_WINDOW_US` | `2000` | batch window (lone requests use 500 µs) |
| `DECIDRA_NATIVE_MAX_BATCH_TOKENS` | `16384` | batch token cap |
| `DECIDRA_NATIVE_GDN_CHUNKED` | `true` | chunked GDN prefill |
| `DECIDRA_NATIVE_GDN_CHUNKED_MIN_SEQ` | `64` | minimum sequence for chunked GDN |
| `DECIDRA_NATIVE_INT8` | `true` | selective W4A8-INT8 activations |
| `DECIDRA_NATIVE_INT8_CLIP` | `1.0` | activation clip factor |
| `DECIDRA_NATIVE_INT8_GROUP_SCALE` | `true` | group-64 activation scales |
| `DECIDRA_NATIVE_INT8_GROUP_EXCLUDE` | `mlp.gate_proj` | modules kept on the per-token epilogue |
| `DECIDRA_NATIVE_INT8_EXCLUDE` | `self_attn.o_proj,self_attn.v_proj,self_attn.k_proj` | modules kept on W4A16 |
| `DECIDRA_NATIVE_INT8_ALL_LAYERS` | `last:8` | all-linear INT8 layer set |
| `DECIDRA_NATIVE_GPU_EMBED` | `true` | resident GPU embedding table |
| `DECIDRA_DECISION_CACHE_SIZE` | `256` | exact-decision cache entries; `0` disables |

### Build and diagnostics

| variable | default | meaning |
|---|---|---|
| `DECIDRA_NVCC` | `nvcc` on `PATH` | CUDA compiler |
| `DECIDRA_CUDA_IMAGE` | `nvidia/cuda:13.1.1-devel-ubuntu24.04` | toolchain image when `nvcc` is absent |
| `DECIDRA_SKIP_BUILD` | unset | reuse existing cubins |
| `DECIDRA_PROFILE` | unset | per-region GPU timing |
| `DECIDRA_CPU_PROFILE` | unset | host timing |
| `DECIDRA_GDN_PROF` | unset | GDN phase counters |
| `DECIDRA_MARLIN_DEBUG` | unset | Marlin launch diagnostics |
| `DECIDRA_GEMM_DEBUG` | unset | GEMM selection diagnostics |
| `DECIDRA_MARLIN_CHUNK` | `0` | split large M launches (tuning) |

## Architecture

```
src/                  HTTP server, decision head, cache, batching, API
native/src/           CUDA engine, model loader, Marlin/FlashInfer bindings
native/kernels/       engine kernels + vendored Marlin
eval/                 suite, benchmark, mock server, protocol client
docs/                 SPEC, RESULTS, KERNELS, DEPLOY, VLLM-COMPARISON
scripts/              run/build helpers
```

The server owns the request path: authentication/signing, JSON parsing, the
exact-decision cache, and the batch queue. The native backend owns the model:
tokenizer, prompt construction, tensor allocation, the decoder forward, and the
decision-head logits. One batch worker thread owns the GPU context; requests
are batched FIFO and each request returns only after its forward completes.

Hardware-specific code stays in `native/`. Porting to another NVIDIA GPU means
building the same kernels for the new architecture and validating the quantized
GEMM paths; the server and decision head do not change. Model-specific code is
limited to the checkpoint contract and the architecture in `native/src`.

## Tests

```bash
cargo test --locked
cargo clippy --all-targets -- -D warnings

# With a server on :8080
python3 eval/suite.py --a-url http://127.0.0.1:8080 \
  --baseline eval/data/awq-cpu-suite28.json \
  --max-kl 0.001 --max-mae 0.001 --max-delta 0.011
```

Current results: 47 unit tests, 10 integration tests, suite 28/28, KL 3.269e-5,
MAE 7.166e-4, max Δ 1.027e-2.

## Migration from `jev-engine`

The engine was renamed from `jev-engine` to **Decidra**. The model (JEV-27B-VL)
and the decision protocol keep their names.

| old | new |
|---|---|
| `jev-engine` binary | `decidra` |
| `jev-native` binary | `decidra-native` |
| `jev-engine` crate | `decidra` |
| `jev-native` crate | `decidra-native` |
| `JEV_*` env vars | `DECIDRA_*` |
| `RUST_LOG=jev_engine=info` | `RUST_LOG=decidra=info` |
| `jev_*` Prometheus metrics | `decidra_*` |
| `/opt/jev` image path | `/opt/decidra` |
| image user `jev` | `decidra` |

The API paths, the default model id `jev-awq`, and the checkpoint layout are
unchanged.

## License

MIT OR Apache-2.0. See [`LICENSE-MIT`](LICENSE-MIT) and
[`LICENSE-APACHE`](LICENSE-APACHE).

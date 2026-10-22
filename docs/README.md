# Decidra documentation

Start with the [project README](../README.md) for the quick start and API.

## Guides

| document | read it for |
|---|---|
| [ENGINE.md](ENGINE.md) | `prefix_bidir` semantics, state cache, kernels, correctness invariants |
| [BENCHMARKS.md](BENCHMARKS.md) | full latency/throughput matrices, profiles, correctness gates |
| [DEPLOY.md](DEPLOY.md) | build, Docker, operations, environment reference |
| [../CHANGELOG.md](../CHANGELOG.md) | release history |

## Development

| document | contents |
|---|---|
| [../native/README.md](../native/README.md) | native engine crate and execution contract |
| [../eval/README.md](../eval/README.md) | HF parity and serving benchmark tools |
| [../quant/d3turbo/README.md](../quant/d3turbo/README.md) | W4A8 g64 quantization pipeline |

## Design history

| document | contents |
|---|---|
| [specs/2026-10-21-d3turbo-state-once.md](specs/2026-10-21-d3turbo-state-once.md) | dated state-once design and review record |

## Conventions

- Performance numbers are end-to-end `POST /v1/systemone` measurements.
- SM89 numbers use the **W4A8 g64 v4** package; SM90 numbers use **BF16**.
- Accuracy claims require the 32-row HF gate and the state-once/negative-control
  tests; the startup self-test is a fail-closed safety check, not the quality
  authority.

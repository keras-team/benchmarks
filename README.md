# Keras Benchmarks

Public, multi-backend end-to-end and layer benchmarks for [Keras](https://keras.io/).

> [!NOTE]
> This repository is currently being bootstrapped. See the design RFC at [keras-team/keras#24216](https://github.com/keras-team/keras/issues/24216) for details and discussion.

## Goals

1. **Demonstrate multi-backend Keras:** Run identical workloads across backends (`JAX`, `TensorFlow`, `PyTorch`, `MLX`, `OpenVINO`, `Paddle`, `NumPy`).
2. **Measure and improve Keras performance:** Track throughput, step latency, time-to-first-token (TTFT), compile time, and peak device memory, and compare against non-Keras baselines.
3. **Improve end-to-end coverage:** Exercise real-world workflows (inference, fine-tuning, quantization, etc.) across accelerators (`TPU`, `GPU`, `CPU`, `Apple Silicon`, `Intel NPU`, `AMD ROCm`).

## Quickstart

Install the package in editable mode along with your target Keras backend (for example, `jax`, `tensorflow`, `torch`, or `mlx`):

```bash
pip install -e ".[dev]"
```

Run the MVP inference benchmark in a fresh per-config subprocess:

```bash
python -m keras_benchmarks.run \
  --workload inference \
  --backend jax \
  --model mlp \
  --compile-mode compiled
```

Or run across multiple backends sequentially (each in its own subprocess):

```bash
keras-benchmarks --backend jax,numpy --model mlp
```

Per-run JSON results are written to `results/<YYYY-MM>/` using the RFC filename format:

```text
results/<YYYY-MM>/{workload}-{backend}-{accelerator}-{topology}-{model}-{compile_mode}-{precision}-{run_id}.json
```

See [Submitting Benchmark Results](CONTRIBUTING.md#submitting-benchmark-results) for a step-by-step guide on running benchmarks and opening a PR.

## Development

```bash
ruff check
ruff format --check
pytest
```

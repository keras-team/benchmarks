# Keras Benchmarks

Public, multi-backend end-to-end and layer benchmarks for [Keras](https://keras.io/).

> [!NOTE]
> This repository is currently being bootstrapped. See the design RFC at [keras-team/keras#24216](https://github.com/keras-team/keras/issues/24216) for details and discussion.

## Goals

1. **Demonstrate multi-backend Keras:** Run identical workloads across backends (`JAX`, `TensorFlow`, `PyTorch`, `MLX`, `OpenVINO`, `Paddle`, `NumPy`).
2. **Measure and improve Keras performance:** Track throughput, step latency, time-to-first-token (TTFT), compile time, and peak device memory, and compare against non-Keras baselines.
3. **Improve end-to-end coverage:** Exercise real-world workflows (inference, fine-tuning, quantization, etc.) across accelerators (`TPU`, `GPU`, `CPU`, `Apple Silicon`, `Intel NPU`, `AMD ROCm`).

import argparse
import statistics
import time
import uuid
from datetime import datetime
from datetime import timezone

import keras
from keras_benchmarks.backend_utils import get_backend_plugin_version
from keras_benchmarks.backend_utils import get_backend_version
from keras_benchmarks.backend_utils import get_benchmarks_version
from keras_benchmarks.backend_utils import get_device_peak_memory_gb
from keras_benchmarks.backend_utils import get_hardware_metadata
from keras_benchmarks.backend_utils import get_host_os
from keras_benchmarks.backend_utils import get_host_ram_gb
from keras_benchmarks.backend_utils import get_package_version
from keras_benchmarks.backend_utils import make_inference_step
from keras_benchmarks.backend_utils import sync_backend
from keras_benchmarks.results import BenchmarkResult
from keras_benchmarks.results import write_result

SUPPORTED_MODELS = ("mlp", "resnet50")


def build_model(model_name: str) -> tuple[keras.Model, tuple[int, ...], str]:
    if model_name == "mlp":
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1024,)),
                keras.layers.Dense(1024, activation="relu"),
                keras.layers.Dense(1024, activation="relu"),
                keras.layers.Dense(1000),
            ],
            name="mlp",
        )
        return model, (1024,), "forward_pass"

    if model_name == "resnet50":
        model = keras.applications.ResNet50(
            weights=None,
            input_shape=(224, 224, 3),
            classes=1000,
        )
        return model, (224, 224, 3), "image_classification"

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Choose from: {', '.join(SUPPORTED_MODELS)}."
    )


def run_inference(
    model_name: str = "mlp",
    compile_mode: str = "compiled",
    precision: str = "float32",
    batch_size: int = 32,
    num_warmup_runs: int = 1,
    num_repeats: int = 5,
    source: str = "internal",
) -> BenchmarkResult:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_warmup_runs < 1:
        raise ValueError("num_warmup_runs must be >= 1.")
    if num_repeats < 1:
        raise ValueError("num_repeats must be >= 1.")

    backend = keras.backend.backend()
    keras.mixed_precision.set_global_policy(precision)
    input_dtype = keras.mixed_precision.global_policy().compute_dtype

    model, input_shape, task = build_model(model_name)
    x = keras.random.normal((batch_size, *input_shape), dtype=input_dtype)
    sync_backend(backend, x)

    step_fn = make_inference_step(model, backend, compile_mode)

    compile_start = time.perf_counter()
    first_out = step_fn(x)
    sync_backend(backend, first_out)
    compile_time_s = round(time.perf_counter() - compile_start, 6)

    for _ in range(num_warmup_runs):
        out = step_fn(x)
        sync_backend(backend, out)

    ms_per_step_runs: list[float] = []
    for _ in range(num_repeats):
        sync_backend(backend)
        step_start = time.perf_counter()
        out = step_fn(x)
        sync_backend(backend, out)
        elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        ms_per_step_runs.append(round(elapsed_ms, 4))

    ms_per_step = round(statistics.median(ms_per_step_runs), 4)
    hw = get_hardware_metadata(backend)

    return BenchmarkResult(
        run_id=uuid.uuid4().hex[:8],
        date=datetime.now(timezone.utc).isoformat(),
        workload="inference",
        keras_version=keras.__version__,
        keras_hub_version=get_package_version("keras-hub"),
        benchmarks_version=get_benchmarks_version(),
        backend=backend,
        backend_plugin_version=get_backend_plugin_version(backend),
        backend_version=get_backend_version(backend),
        compile_mode=compile_mode,
        accelerator=str(hw["accelerator"]),
        device_name=str(hw["device_name"]),
        num_chips=int(hw["num_chips"]),
        topology=str(hw["topology"]),
        host_ram_gb=get_host_ram_gb(),
        host_os=get_host_os(),
        model=model_name,
        task=task,
        precision=precision,
        batch_size=batch_size,
        num_warmup_runs=num_warmup_runs,
        num_repeats=num_repeats,
        ms_per_step=ms_per_step,
        ms_per_step_runs=ms_per_step_runs,
        compile_time_s=compile_time_s,
        device_peak_memory_gb=get_device_peak_memory_gb(backend),
        source=source,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Keras inference benchmark workload."
    )
    parser.add_argument(
        "--model",
        default="mlp",
        choices=SUPPORTED_MODELS,
        help="Model architecture to benchmark (default: mlp).",
    )
    parser.add_argument(
        "--compile-mode",
        default="compiled",
        choices=("eager", "compiled"),
        help="Execution mode (default: compiled).",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        help="Keras dtype policy name (default: float32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup passes (default: 1).",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=5,
        help="Number of timed repeats (default: 5).",
    )
    parser.add_argument(
        "--source",
        default="internal",
        choices=("internal", "partner", "community"),
        help="Submission source label (default: internal).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write per-run JSON files (default: results).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_inference(
        model_name=args.model,
        compile_mode=args.compile_mode,
        precision=args.precision,
        batch_size=args.batch_size,
        num_warmup_runs=args.num_warmup_runs,
        num_repeats=args.num_repeats,
        source=args.source,
    )
    output_path = write_result(result, results_dir=args.results_dir)
    print(f"Wrote benchmark result to {output_path}")
    print(result.to_json(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

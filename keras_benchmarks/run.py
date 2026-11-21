import argparse
import os
import subprocess
import sys

SUPPORTED_BACKENDS = (
    "jax",
    "tensorflow",
    "torch",
    "mlx",
    "openvino",
    "paddle",
    "numpy",
)
SUPPORTED_WORKLOADS = ("inference",)


def run_benchmark_subprocess(
    workload: str,
    backend: str,
    model: str,
    compile_mode: str,
    precision: str,
    batch_size: int,
    num_warmup_runs: int,
    num_repeats: int,
    source: str,
    results_dir: str,
) -> None:
    if workload not in SUPPORTED_WORKLOADS:
        raise ValueError(
            f"Unsupported workload '{workload}'. "
            f"Choose from: {', '.join(SUPPORTED_WORKLOADS)}."
        )
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. "
            f"Choose from: {', '.join(SUPPORTED_BACKENDS)}."
        )

    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend

    cmd = [
        sys.executable,
        "-m",
        f"keras_benchmarks.workloads.{workload}",
        "--model",
        model,
        "--compile-mode",
        compile_mode,
        "--precision",
        precision,
        "--batch-size",
        str(batch_size),
        "--num-warmup-runs",
        str(num_warmup_runs),
        "--num-repeats",
        str(num_repeats),
        "--source",
        source,
        "--results-dir",
        results_dir,
    ]
    subprocess.run(cmd, env=env, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Keras benchmarks in fresh per-config subprocesses."
    )
    parser.add_argument(
        "--workload",
        default="inference",
        choices=SUPPORTED_WORKLOADS,
        help="Benchmark workload module to run (default: inference).",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("KERAS_BACKEND", "jax"),
        help=(
            "Keras backend or comma-separated list of backends "
            "(default: $KERAS_BACKEND or jax)."
        ),
    )
    parser.add_argument(
        "--model",
        default="mlp",
        help="Model name to benchmark (default: mlp).",
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
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    if not backends:
        parser.error("At least one backend must be specified.")

    for backend in backends:
        compile_mode = args.compile_mode
        if backend == "numpy" and compile_mode == "compiled":
            compile_mode = "eager"
        run_benchmark_subprocess(
            workload=args.workload,
            backend=backend,
            model=args.model,
            compile_mode=compile_mode,
            precision=args.precision,
            batch_size=args.batch_size,
            num_warmup_runs=args.num_warmup_runs,
            num_repeats=args.num_repeats,
            source=args.source,
            results_dir=args.results_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

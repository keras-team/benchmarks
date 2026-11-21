import json

from keras_benchmarks.results import BenchmarkResult
from keras_benchmarks.results import format_result_filename
from keras_benchmarks.results import format_result_month
from keras_benchmarks.results import write_result


def _make_sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        run_id="abc12345",
        date="2026-11-21T23:00:00+00:00",
        workload="inference",
        keras_version="3.15.1",
        benchmarks_version="169c6ae",
        backend="jax",
        backend_version="0.12.1",
        compile_mode="compiled",
        accelerator="CPU",
        device_name="Intel Xeon",
        num_chips=1,
        topology="1x1",
        host_ram_gb=8.0,
        host_os="Ubuntu 24.04",
        model="mlp",
        task="forward_pass",
        precision="float32",
        batch_size=32,
        num_warmup_runs=1,
        num_repeats=5,
        ms_per_step=1.25,
        ms_per_step_runs=[1.3, 1.25, 1.2, 1.25, 1.28],
        compile_time_s=0.45,
        device_peak_memory_gb=0.25,
        source="internal",
    )


def test_format_result_filename() -> None:
    result = _make_sample_result()
    assert format_result_filename(result) == (
        "inference-jax-CPU-1x1-mlp-compiled-float32-abc12345.json"
    )


def test_format_result_month() -> None:
    assert format_result_month("2026-11-21T23:00:00Z") == "2026-11"


def test_to_dict_omits_none_optional_fields() -> None:
    result = _make_sample_result()
    payload = result.to_dict()
    assert "ttft_ms" not in payload
    assert "keras_hub_version" not in payload
    assert payload["run_id"] == "abc12345"


def test_write_result_creates_month_subdir(tmp_path) -> None:
    result = _make_sample_result()
    path = write_result(result, results_dir=tmp_path)
    assert path.parent.name == "2026-11"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["ms_per_step"] == 1.25
    assert loaded["ms_per_step_runs"] == [1.3, 1.25, 1.2, 1.25, 1.28]

import json
import re
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "unknown"


@dataclass
class BenchmarkResult:
    run_id: str
    date: str
    workload: str
    keras_version: str
    benchmarks_version: str
    backend: str
    backend_version: str
    compile_mode: str
    accelerator: str
    device_name: str
    num_chips: int
    topology: str
    host_ram_gb: float
    host_os: str
    model: str
    task: str
    precision: str
    batch_size: int
    num_warmup_runs: int
    num_repeats: int
    ms_per_step: float
    ms_per_step_runs: list[float]
    compile_time_s: float
    device_peak_memory_gb: float
    source: str
    keras_hub_version: str | None = None
    backend_plugin_version: str | None = None
    preset: str | None = None
    prompt_len: int | None = None
    decode_len: int | None = None
    tokens_per_sec: float | None = None
    ttft_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def format_result_filename(result: BenchmarkResult) -> str:
    parts = [
        _sanitize_token(result.workload),
        _sanitize_token(result.backend),
        _sanitize_token(result.accelerator),
        _sanitize_token(result.topology),
        _sanitize_token(result.model),
        _sanitize_token(result.compile_mode),
        _sanitize_token(result.precision),
        _sanitize_token(result.run_id),
    ]
    return "-".join(parts) + ".json"


def format_result_month(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc).strftime("%Y-%m")


def write_result(
    result: BenchmarkResult,
    results_dir: str | Path = "results",
) -> Path:
    month_dir = Path(results_dir) / format_result_month(result.date)
    month_dir.mkdir(parents=True, exist_ok=True)
    output_path = month_dir / format_result_filename(result)
    output_path.write_text(result.to_json(), encoding="utf-8")
    return output_path

import json

from keras_benchmarks.run import main


def test_run_end_to_end_jax_and_numpy(tmp_path) -> None:
    exit_code = main(
        [
            "--workload",
            "inference",
            "--backend",
            "jax,numpy",
            "--model",
            "mlp",
            "--compile-mode",
            "compiled",
            "--batch-size",
            "4",
            "--num-warmup-runs",
            "1",
            "--num-repeats",
            "2",
            "--results-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0

    result_files = sorted(tmp_path.rglob("*.json"))
    assert len(result_files) == 2

    records = [json.loads(p.read_text(encoding="utf-8")) for p in result_files]
    by_backend = {r["backend"]: r for r in records}
    assert set(by_backend) == {"jax", "numpy"}
    assert by_backend["jax"]["compile_mode"] == "compiled"
    assert by_backend["numpy"]["compile_mode"] == "eager"
    for record in records:
        assert record["workload"] == "inference"
        assert record["model"] == "mlp"
        assert record["batch_size"] == 4
        assert record["num_warmup_runs"] == 1
        assert record["num_repeats"] == 2
        assert len(record["ms_per_step_runs"]) == 2
        assert record["ms_per_step"] > 0.0
        assert record["compile_time_s"] >= 0.0
        assert record["device_peak_memory_gb"] > 0.0

import pytest

from keras_benchmarks.backend_utils import get_backend_version
from keras_benchmarks.backend_utils import get_device_peak_memory_gb
from keras_benchmarks.backend_utils import get_hardware_metadata
from keras_benchmarks.backend_utils import get_host_os
from keras_benchmarks.backend_utils import get_host_ram_gb
from keras_benchmarks.backend_utils import make_inference_step


def test_host_metadata_helpers() -> None:
    assert get_host_ram_gb() > 0.0
    assert len(get_host_os()) > 0
    assert get_backend_version("numpy") != "unknown"


def test_hardware_metadata_cpu_fields() -> None:
    hw = get_hardware_metadata("numpy")
    assert hw["accelerator"] == "CPU"
    assert hw["num_chips"] == 1
    assert hw["topology"] == "1x1"
    assert isinstance(hw["device_name"], str)


def test_device_peak_memory_positive() -> None:
    assert get_device_peak_memory_gb("numpy") > 0.0


def test_make_inference_step_rejects_invalid_compile_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported compile_mode"):
        make_inference_step(object(), "jax", "invalid")


def test_make_inference_step_rejects_compiled_numpy() -> None:
    with pytest.raises(ValueError, match="not supported"):
        make_inference_step(object(), "numpy", "compiled")

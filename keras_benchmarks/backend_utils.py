import os
import platform
import resource
import subprocess
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

_BACKEND_DISTRIBUTIONS = {
    "jax": "jax",
    "tensorflow": "tensorflow",
    "torch": "torch",
    "mlx": "mlx",
    "openvino": "openvino",
    "paddle": "paddlepaddle",
    "numpy": "numpy",
}

_BACKEND_PLUGINS = {
    "mlx": "keras-mlx",
    "openvino": "keras-openvino",
    "paddle": "keras-paddle",
}


def get_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def get_backend_version(backend: str) -> str:
    dist_name = _BACKEND_DISTRIBUTIONS.get(backend, backend)
    return get_package_version(dist_name) or "unknown"


def get_backend_plugin_version(backend: str) -> str | None:
    plugin_name = _BACKEND_PLUGINS.get(backend)
    if plugin_name is None:
        return None
    return get_package_version(plugin_name)


def get_benchmarks_version() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = output.strip()
        if sha:
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return get_package_version("keras-benchmarks") or "0.0.1"


def get_host_ram_gb() -> float:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return round((page_size * phys_pages) / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        pass
    return 0.0


def get_host_os() -> str:
    system = platform.system()
    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    if system == "Linux":
        os_release = Path("/etc/os-release")
        if os_release.exists():
            for line in os_release.read_text(encoding="utf-8").splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    return f"{system} {platform.release()}".strip()


def _get_cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "CPU"


def get_hardware_metadata(backend: str) -> dict[str, object]:
    if backend == "jax":
        try:
            import jax

            devices = jax.devices()
            if devices:
                plat = devices[0].platform
                kind = getattr(devices[0], "device_kind", plat)
                if plat == "tpu":
                    return {
                        "accelerator": "TPU",
                        "device_name": str(kind),
                        "num_chips": len(devices),
                        "topology": f"{len(devices)}x1",
                    }
                if plat == "gpu":
                    return {
                        "accelerator": "GPU",
                        "device_name": str(kind),
                        "num_chips": len(devices),
                        "topology": f"{len(devices)}x1",
                    }
        except Exception:
            pass
    elif backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return {
                    "accelerator": "GPU",
                    "device_name": torch.cuda.get_device_name(0),
                    "num_chips": count,
                    "topology": f"{count}x1",
                }
            if hasattr(torch, "mps") and torch.mps.is_available():
                return {
                    "accelerator": "Apple Silicon",
                    "device_name": platform.machine(),
                    "num_chips": 1,
                    "topology": "1x1",
                }
        except Exception:
            pass
    elif backend == "tensorflow":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return {
                    "accelerator": "GPU",
                    "device_name": gpus[0].name,
                    "num_chips": len(gpus),
                    "topology": f"{len(gpus)}x1",
                }
        except Exception:
            pass
    elif backend == "mlx":
        return {
            "accelerator": "Apple Silicon",
            "device_name": platform.machine(),
            "num_chips": 1,
            "topology": "1x1",
        }

    return {
        "accelerator": "CPU",
        "device_name": _get_cpu_name(),
        "num_chips": 1,
        "topology": "1x1",
    }


def sync_backend(backend: str, output: object = None) -> None:
    if backend == "jax":
        import jax

        if output is not None:
            jax.block_until_ready(output)
    elif backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()
    elif backend == "mlx":
        import mlx.core as mx

        if output is not None:
            mx.eval(output)
        else:
            mx.synchronize()
    elif backend == "tensorflow":
        import tensorflow as tf

        if hasattr(tf.experimental, "async_wait"):
            tf.experimental.async_wait()
        elif output is not None:
            tf.nest.map_structure(
                lambda x: x.numpy() if hasattr(x, "numpy") else x,
                output,
            )


def make_inference_step(
    model: object,
    backend: str,
    compile_mode: str,
) -> Callable[[object], object]:
    if compile_mode not in ("eager", "compiled"):
        raise ValueError(
            f"Unsupported compile_mode '{compile_mode}'. "
            "Expected 'eager' or 'compiled'."
        )

    if compile_mode == "eager":
        if backend == "torch":
            import torch

            def _torch_eager(x: object) -> object:
                with torch.no_grad():
                    return model(x, training=False)

            return _torch_eager

        return lambda x: model(x, training=False)

    if backend == "jax":
        import jax

        return jax.jit(lambda x: model(x, training=False))
    if backend == "tensorflow":
        import tensorflow as tf

        return tf.function(
            lambda x: model(x, training=False),
            jit_compile=True,
        )
    if backend == "torch":
        import torch

        compiled_model = torch.compile(model)

        def _torch_compiled(x: object) -> object:
            with torch.no_grad():
                return compiled_model(x, training=False)

        return _torch_compiled
    if backend == "mlx":
        import mlx.core as mx

        return mx.compile(lambda x: model(x, training=False))

    raise ValueError(
        f"compile_mode='compiled' is not supported for backend '{backend}'."
    )


def get_device_peak_memory_gb(backend: str) -> float:
    if backend == "jax":
        try:
            import jax

            stats = jax.devices()[0].memory_stats()
            if stats and "peak_bytes_in_use" in stats:
                return round(stats["peak_bytes_in_use"] / (1024**3), 4)
        except Exception:
            pass
    elif backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                return round(peak_bytes / (1024**3), 4)
        except Exception:
            pass
    elif backend == "mlx":
        try:
            import mlx.core as mx

            if hasattr(mx, "get_peak_memory"):
                return round(mx.get_peak_memory() / (1024**3), 4)
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
                return round(mx.metal.get_peak_memory() / (1024**3), 4)
        except Exception:
            pass

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(max_rss / (1024**3), 4)
    return round((max_rss * 1024) / (1024**3), 4)

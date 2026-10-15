import os
import sys

import pytest

# make the in-tree package importable without installation (src layout)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _gpu_available():
    """A CUDA device is present and the covering plugin .so is built."""
    from coldcore.gpu import default_plugin_path
    if not os.path.exists(default_plugin_path()):
        return False, f"plugin not built: {default_plugin_path()}"
    if not (os.path.exists("/dev/nvidia0") or
            os.path.exists("/dev/nvidiactl") or
            os.environ.get("COLDCORE_FORCE_GPU_TESTS")):
        return False, "no CUDA device node"
    return True, ""


@pytest.fixture(scope="session")
def gpu_backend_factory():
    ok, why = _gpu_available()
    if not ok:
        pytest.skip(why)
    from coldcore.gpu import GpuCoveringBackend
    return GpuCoveringBackend


def _total_ram_bytes():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return 0


# tests marked `bigmem` build ~10 GiB tables; a 16 GiB CI runner gets
# OOM-killed mid-allocation, so require headroom rather than crash.
BIGMEM_MIN_RAM = 32 << 30


def pytest_collection_modifyitems(config, items):
    ok, why = _gpu_available()
    if not ok:
        skip = pytest.mark.skip(reason=f"gpu unavailable: {why}")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
    ram = _total_ram_bytes()
    if ram and ram < BIGMEM_MIN_RAM:
        skip_mem = pytest.mark.skip(
            reason=f"needs >= {BIGMEM_MIN_RAM >> 30} GiB RAM "
                   f"(machine has {ram >> 30} GiB)")
        for item in items:
            if "bigmem" in item.keywords:
                item.add_marker(skip_mem)

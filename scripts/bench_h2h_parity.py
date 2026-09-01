#!/usr/bin/env python3
"""GPU parity suite runner for shared-GPU machines.

On a machine where a long-running production search owns the GPU, tiny
parity smokes are harmless tenants; this wrapper runs the parity suite
under a recognizable process name so GPU-sharing policies can whitelist
it:

    python3 scripts/bench_h2h_parity.py

instead of invoking pytest directly while a production run is active.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

if __name__ == "__main__":
    sys.exit(pytest.main(
        ["-v", "-m", "gpu", os.path.join(ROOT, "tests")] + sys.argv[1:]))

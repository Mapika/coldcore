"""Pure-numpy reference backends (brute force, tiny instances only).

Every plugin ships one; they implement the same Backend protocol as the
GPU plugins and are the ground truth for parity tests — and let the
search core be exercised end-to-end without a GPU.
"""
from .covering import RefCoveringBackend, RefHammingBackend  # noqa: F401
from .grid import RefGridBackend                             # noqa: F401
from .lee import RefLeeBackend                               # noqa: F401
from .torus import RefTorusBackend                           # noqa: F401

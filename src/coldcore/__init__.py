"""coldcore — GPU combinatorial-optimization framework for record hunting.

Three layers: problem plugin (CUDA .so producing exact whole-space
gain/loss fields), state core (dense fields, HBM/LPDDR placement), search
core (exact lazy greedy + ruin-and-recreate LNS), plus a symmetry layer
(group-invariant construction, orbit-restricted search, isomorph-free
exhaustive settlement) that drives the same backends.
"""

__version__ = "0.1.1"

from .space import ProductSpace                       # noqa: F401
from .protocol import Backend                         # noqa: F401
from .search import Searcher                          # noqa: F401
from .symmetry import Group, SymmetricSearcher        # noqa: F401

"""Problems on the explicit-incidence backend family (candidates and
universe are different finite sets; coverage is an incidence list).

The product-space problems (hamming, torus_linf, lee, grid_linf) live
in the plugin / ``coldcore.dp`` / ``coldcore.reference``; this package
holds the ones lowered through ``coldcore.incidence``.
"""
from .covering_design import CoveringDesignBackend  # noqa: F401

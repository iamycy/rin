"""rin-pitch: Relative Interval Network (RIN) pitch smoothing.

Two core functions, arrays in / arrays out:

- :func:`vqt_diff_calculator`: multi-hop relative pitch differences from audio.
- :func:`lp_smoother`: fuse absolute pitch estimates with relative differences
  through a network-flow linear program.

Both are plain functions obeying the contracts in :mod:`rin.interfaces`.
Bring your own implementations with the same signatures and wire them
however you like -- this package does no file loading and no caching.
"""

from .interfaces import DifferenceEstimator, Solver
from .lp import lp_smoother
from .relative import vqt_diff_calculator

__version__ = "0.1.0"

__all__ = [
    "DifferenceEstimator",
    "Solver",
    "lp_smoother",
    "vqt_diff_calculator",
    "__version__",
]

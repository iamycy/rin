"""rin-pitch: Relative Interval Network (RIN) pitch smoothing.

Three core functions, arrays in / arrays out:

- :func:`vqt_diff_calculator`: multi-hop relative pitch differences from audio.
- :func:`lp_smoother`: fuse absolute pitch estimates with relative differences
  through a network-flow linear program.
- :func:`estimate_voicing`: per-frame voicing from absolute and relative
  confidences.

Each is a plain function obeying the contracts in :mod:`rin.interfaces`.
Bring your own implementations with the same signatures and wire them
however you like -- this package does no file loading and no caching.
"""

from .interfaces import DifferenceEstimator, Solver, VoicingEstimator
from .lp import estimate_voicing, lp_smoother
from .relative import vqt_diff_calculator

__version__ = "0.1.0"

__all__ = [
    "DifferenceEstimator",
    "Solver",
    "VoicingEstimator",
    "estimate_voicing",
    "lp_smoother",
    "vqt_diff_calculator",
    "__version__",
]

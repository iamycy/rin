"""Relative Interval Network (RIN) pitch smoothing.

Three core functions, arrays in / arrays out:

- :func:`vqt_diff_calculator`: multi-hop relative pitch differences from audio.
- :func:`lp_smoother`: fuse absolute pitch estimates with relative differences
  through a network-flow linear program.
- :func:`estimate_voicing`: per-frame voicing from absolute and relative
  confidences.

:func:`smooth_pitch` chains the three with the paper's fixed settings; each
stage is an injectable callable, so it doubles as a swappable pipeline.
Each core is a plain function obeying the contracts in :mod:`rin.interfaces`.
Bring your own implementations with the same signatures and wire them
however you like; this package does no file loading and no caching.
"""

from importlib.metadata import PackageNotFoundError, version

from .interfaces import DifferenceEstimator, Solver, VoicingEstimator
from .lp import estimate_voicing, lp_smoother
from .pipeline import HOPS, smooth_pitch
from .relative import vqt_diff_calculator

try:
    __version__ = version("rin-pitch")
except PackageNotFoundError:  # pragma: no cover (source tree with no install)
    __version__ = "0.0.0+unknown"

__all__ = [
    "DifferenceEstimator",
    "HOPS",
    "Solver",
    "VoicingEstimator",
    "estimate_voicing",
    "lp_smoother",
    "smooth_pitch",
    "vqt_diff_calculator",
    "__version__",
]

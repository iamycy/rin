"""rin-pitch: Relative Interval Network (RIN) pitch smoothing.

Fuse per-frame absolute pitch estimates (from any F0 tracker) with multi-hop
relative pitch differences estimated from a magnitude VQT, through a
network-flow linear program. Both the difference estimator and the solver are
pluggable via :mod:`rin.interfaces`.
"""

from .interfaces import DifferenceEstimator, Solver
from .lp import lp_smoother
from .relative import vqt_diff_calculator
from .smooth import DEFAULT_HOPS, PERIOD, build_hop_cache, rin_smooth, smooth

__version__ = "0.1.0"

__all__ = [
    "smooth",
    "rin_smooth",
    "build_hop_cache",
    "lp_smoother",
    "vqt_diff_calculator",
    "DifferenceEstimator",
    "Solver",
    "DEFAULT_HOPS",
    "PERIOD",
    "__version__",
]

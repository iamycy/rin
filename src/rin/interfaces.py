"""Plugin interfaces for rin-pitch.

Python has no C++-style templates; the idiomatic equivalent for a plugin API
is a :class:`typing.Protocol` (PEP 544): a structural interface. Any class
that implements the documented methods satisfies the protocol -- no
inheritance required -- and static type checkers verify it. The protocols are
also :func:`typing.runtime_checkable`, so :func:`rin.smooth.smooth` can
reject objects that do not provide the required methods with a clear error.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DifferenceEstimator(Protocol):
    """Estimates inter-frame relative pitch differences from audio.

    Implement this to plug in your own pitch-difference calculation
    (e.g. a different frontend than the built-in VQT cross-correlation).
    """

    def estimate(
        self,
        x: np.ndarray,
        sr: int,
        hop_length: int,
        hops: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate relative pitch differences.

        Args:
            x: mono input audio.
            sr: sample rate in Hz.
            hop_length: frame hop in samples (defines the output frame grid).
            hops: relative frame offsets to estimate, e.g. ``(1, 2, 3, 5)``.

        Returns:
            edges: ``(E, 2)`` integer frame-index pairs ``(u, v)``.
            estimates: ``(E,)`` pitch differences ``f[v] - f[u]`` in cents.
            confidences: ``(E,)`` weights in ``[0, 1]``; zero-weight edges
                are dropped before solving.
        """
        ...


@runtime_checkable
class Solver(Protocol):
    """Fuses absolute pitch estimates with relative pitch differences.

    Implement this to plug in your own fusion algorithm (e.g. an
    alternative to the built-in network-flow LP).
    """

    def solve(
        self,
        abs_estimates: np.ndarray,
        abs_confidences: np.ndarray,
        rel_edges: np.ndarray,
        rel_estimates: np.ndarray,
        rel_confidences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse absolute and relative pitch information.

        Args:
            abs_estimates: ``(M,)`` absolute pitch per frame, in cents.
            abs_confidences: ``(M,)`` weights in ``[0, 1]``.
            rel_edges: ``(E, 2)`` integer frame-index pairs.
            rel_estimates: ``(E,)`` relative pitch differences in cents.
            rel_confidences: ``(E,)`` weights in ``[0, 1]``.

        Returns:
            smooth_pitch: ``(M,)`` fused pitch contour, in cents.
            voicing: ``(M,)`` voicing probabilities in ``[0, 1]``.
        """
        ...

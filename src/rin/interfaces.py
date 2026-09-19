"""Plugin interfaces for rin-pitch.

The three extension points are plain function signatures: implement a
function with the documented signature and call it in place of the built-in
cores (:func:`rin.vqt_diff_calculator`, :func:`rin.lp_smoother`,
:func:`rin.estimate_voicing`). No inheritance, no classes required -- any
callable works (function, lambda, ``functools.partial``, or a callable
object) as long as it obeys the signature.

The contracts are expressed as :data:`typing.Callable` type aliases so
static type checkers verify implementations.
"""

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

DifferenceEstimator: TypeAlias = Callable[
    [np.ndarray, int, int, Sequence[int]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]
"""Estimate inter-frame relative pitch differences from audio.

Signature: ``(x, sr, hop_length, hops) -> (edges, estimates, confidences)``.

Args:
    x: mono input audio.
    sr: sample rate in Hz.
    hop_length: frame hop in samples (defines the output frame grid).
    hops: relative frame offsets to estimate, e.g. ``(1, 2, 3, 5)``.

Returns:
    edges: ``(E, 2)`` integer frame-index pairs ``(u, v)``.
    estimates: ``(E,)`` pitch differences ``f[v] - f[u]`` in cents.
    confidences: ``(E,)`` weights in ``[0, 1]``; zero-weight edges are
        dropped before solving.
"""

Solver: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
"""Fuse absolute pitch estimates with relative pitch differences.

Signature: ``(abs_estimates, abs_confidences, rel_edges, rel_estimates,
rel_confidences) -> smooth_pitch``.

Args:
    abs_estimates: ``(M,)`` absolute pitch per frame, in cents.
    abs_confidences: ``(M,)`` weights in ``[0, 1]``.
    rel_edges: ``(E, 2)`` integer frame-index pairs.
    rel_estimates: ``(E,)`` relative pitch differences in cents.
    rel_confidences: ``(E,)`` weights in ``[0, 1]``.

Returns:
    smooth_pitch: ``(M,)`` fused pitch contour, in cents.
"""

VoicingEstimator: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
"""Estimate per-frame voicing from confidences.

Signature: ``(abs_confidences, rel_edges, rel_confidences) -> voicing``.

Args:
    abs_confidences: ``(M,)`` weights in ``[0, 1]``.
    rel_edges: ``(E, 2)`` integer frame-index pairs.
    rel_confidences: ``(E,)`` weights in ``[0, 1]``.

Returns:
    voicing: ``(M,)`` voicing probabilities in ``[0, 1]``.
"""

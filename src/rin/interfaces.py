"""Plugin interfaces for rin-pitch.

The three extension points are plain function signatures: implement a
function with the documented signature and call it in place of the built-in
cores (:func:`rin.vqt_diff_calculator`, :func:`rin.lp_smoother`,
:func:`rin.estimate_voicing`). No inheritance, no classes required: any
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

Parameters
----------
x : np.ndarray [shape=(n,)]
    Mono (1-D) input audio.
sr : int
    Sample rate in Hz.
hop_length : int
    Frame hop in samples (defines the output frame grid).
hops : sequence of int
    Relative frame offsets to estimate, e.g. ``(1, 2, 3, 5)``.

Returns
-------
edges : np.ndarray [shape=(E, 2)]
    Integer frame-index pairs ``(u, v)``.
estimates : np.ndarray [shape=(E,)]
    Pitch differences ``f[v] - f[u]`` in cents.
confidences : np.ndarray [shape=(E,)]
    Weights in ``[0, 1]``; zero-weight edges are dropped before solving.
"""

Solver: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
"""Fuse absolute pitch estimates with relative pitch differences.

Signature: ``(abs_estimates, abs_confidences, rel_edges, rel_estimates,
rel_confidences) -> smooth_pitch``.

The pitch domain is the caller's choice: ``abs_estimates`` and
``rel_estimates`` must share it, and the output is in the same domain.

Parameters
----------
abs_estimates : np.ndarray [shape=(M,)]
    Absolute pitch per frame.
abs_confidences : np.ndarray [shape=(M,)]
    Weights in ``[0, 1]``.
rel_edges : np.ndarray [shape=(E, 2)]
    Integer frame-index pairs.
rel_estimates : np.ndarray [shape=(E,)]
    Relative pitch differences.
rel_confidences : np.ndarray [shape=(E,)]
    Weights in ``[0, 1]``.

Returns
-------
smooth_pitch : np.ndarray [shape=(M,)]
    Fused pitch contour.
"""

VoicingEstimator: TypeAlias = Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]
"""Estimate per-frame voicing from confidences.

Signature: ``(abs_confidences, rel_edges, rel_confidences) -> voicing``.

Parameters
----------
abs_confidences : np.ndarray [shape=(M,)]
    Weights in ``[0, 1]``.
rel_edges : np.ndarray [shape=(E, 2)]
    Integer frame-index pairs.
rel_confidences : np.ndarray [shape=(E,)]
    Weights in ``[0, 1]``.

Returns
-------
voicing : np.ndarray [shape=(M,)]
    Voicing probabilities in ``[0, 1]``.
"""

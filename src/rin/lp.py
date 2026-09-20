"""Network-flow LP smoothing of absolute pitch with relative pitch constraints.

Given per-frame absolute pitch estimates (e.g. from any F0 tracker) and a set
of inter-frame relative pitch differences with confidences, this solves the
weighted L1 fusion problem

    min_f  sum_n w_n |f[n] - a[n]| + sum_{(u,v) in E} w_{uv} |f[v] - f[u] - d_{uv}|

as a linear program. The Lagrangian dual is solved: a min-cost network
circulation problem on a graph with one auxiliary ground node. The optimal
primal contour is read off as the Lagrange multipliers (node potentials) of
the flow-conservation constraints, so only one (smaller) LP is solved, with
the HiGHS simplex solver bundled with SciPy.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def _edge_keep_mask(rel_edges: np.ndarray, rel_confidences: np.ndarray, M: int) -> np.ndarray:
    """Boolean mask dropping out-of-range and zero-weight relative edges."""
    valid = np.all((rel_edges >= 0) & (rel_edges < M), axis=1)
    return valid & (rel_confidences > 0)


def _validate_edge_inputs(
    abs_confidences: np.ndarray,
    rel_edges: np.ndarray,
    rel_confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and normalize the inputs shared by both cores.

    Returns ``(abs_confidences, rel_edges, rel_confidences, M)``. The contract
    it enforces is documented on the two public functions that call it.
    """
    abs_confidences = np.asarray(abs_confidences, dtype=float)
    rel_confidences = np.asarray(rel_confidences, dtype=float)
    rel_edges = np.asarray(rel_edges)
    if rel_edges.size == 0:
        if rel_edges.ndim > 2 or (rel_edges.ndim == 2 and rel_edges.shape[1] not in (0, 2)):
            raise ValueError("expected rel_edges of shape (E, 2)")
        rel_edges = np.zeros((0, 2), dtype=int)
    if abs_confidences.ndim != 1 or rel_edges.ndim != 2 or rel_edges.shape[1] != 2:
        raise ValueError("expected abs_confidences 1-D of length M and rel_edges of shape (E, 2)")
    if not np.issubdtype(rel_edges.dtype, np.integer):
        raise ValueError("rel_edges must contain integer frame indices")
    if rel_confidences.shape != (len(rel_edges),):
        raise ValueError("rel_confidences must have length E")
    if not (np.isfinite(abs_confidences).all() and np.isfinite(rel_confidences).all()):
        raise ValueError("confidences must be finite")
    if (abs_confidences < 0).any() or (rel_confidences < 0).any():
        raise ValueError("confidences must be non-negative")
    return abs_confidences, rel_edges, rel_confidences, abs_confidences.size


def estimate_voicing(
    abs_confidences: np.ndarray,
    rel_edges: np.ndarray,
    rel_confidences: np.ndarray,
) -> np.ndarray:
    """Fuse absolute confidences with incident relative-edge confidences.

    Distance-weighted root-mean-square (RMS, p=2 with 1/m^2 distance decay)
    fusion: integrates incident relative edge confidences while giving
    higher priority to short-hop edges (d = 1/m^2), combined with the
    tracker's absolute confidence (in [0, 1]) via a fuzzy AND (geometric
    mean).

    Parameters
    ----------
    abs_confidences : np.ndarray [shape=(M,)]
        Confidence/weight of each absolute estimate.
    rel_edges : np.ndarray [shape=(E, 2)]
        Frame-index pairs ``(u, v)`` for each relative edge.
    rel_confidences : np.ndarray [shape=(E,)]
        Confidence/weight of each relative estimate.

    Returns
    -------
    np.ndarray [shape=(M,)]
        Voicing probabilities in [0, 1].

    Raises
    ------
    ValueError
        If shapes disagree, values are non-finite, confidences are negative,
        or ``rel_edges`` is not an ``(E, 2)`` integer array.

    Notes
    -----
    Edges referencing frames outside ``[0, M)`` are dropped, as are
    zero-weight edges (they carry no voicing evidence).
    """
    abs_confidences, rel_edges, rel_confidences, M = _validate_edge_inputs(
        abs_confidences, rel_edges, rel_confidences
    )

    keep = _edge_keep_mask(rel_edges, rel_confidences, M)
    rel_edges = rel_edges[keep]
    rel_confidences = rel_confidences[keep]

    # Accumulate every edge onto both of its endpoints: an edge is evidence
    # for the frames at both ends, whichever way round it was given.
    u, v = rel_edges[:, 0], rel_edges[:, 1]
    d = 1.0 / np.maximum(np.abs(v - u), 1).astype(float) ** 2
    weighted = d * rel_confidences**2
    num = np.bincount(u, weighted, minlength=M) + np.bincount(v, weighted, minlength=M)
    den = np.bincount(u, d, minlength=M) + np.bincount(v, d, minlength=M)

    rel_rms = np.sqrt(np.divide(num, np.maximum(den, 1e-10)))
    return np.sqrt(np.clip(abs_confidences, 0.0, 1.0) * np.clip(rel_rms, 0.0, 1.0))


def lp_smoother(
    abs_estimates: np.ndarray,
    abs_confidences: np.ndarray,
    rel_edges: np.ndarray,
    rel_estimates: np.ndarray,
    rel_confidences: np.ndarray,
) -> np.ndarray:
    """Smooth absolute pitch estimates with relative pitch constraints.

    The pitch domain is the caller's choice (cents, MIDI, log-frequency,
    ...): ``abs_estimates`` and ``rel_estimates`` must share it, and the
    output is in the same domain. This package never converts units.

    Parameters
    ----------
    abs_estimates : np.ndarray [shape=(M,)]
        Absolute pitch per frame.
    abs_confidences : np.ndarray [shape=(M,)]
        Confidence/weight of each absolute estimate.
    rel_edges : np.ndarray [shape=(E, 2)]
        Frame-index pairs ``(u, v)`` for each relative edge.
    rel_estimates : np.ndarray [shape=(E,)]
        Pitch difference ``f[v] - f[u]`` per edge.
    rel_confidences : np.ndarray [shape=(E,)]
        Confidence/weight of each relative estimate. Zero-weight edges are
        dropped (they cannot bind the optimum).

    Returns
    -------
    np.ndarray [shape=(M,)]
        Smoothed pitch contour.

    Raises
    ------
    ValueError
        If shapes disagree, values are non-finite, confidences are negative,
        or ``rel_edges`` is not an ``(E, 2)`` integer array.

    See Also
    --------
    estimate_voicing : Fuse confidences into per-frame voicing probabilities.

    Notes
    -----
    Edges referencing frames outside ``[0, M)`` are dropped, as are
    zero-weight edges (they cannot bind the optimum in either form).
    """
    abs_estimates = np.asarray(abs_estimates, dtype=float)
    rel_estimates = np.asarray(rel_estimates, dtype=float)
    abs_confidences, rel_edges, rel_confidences, M = _validate_edge_inputs(
        abs_confidences, rel_edges, rel_confidences
    )
    if abs_estimates.ndim != 1 or abs_estimates.shape != (M,):
        raise ValueError("expected abs_estimates 1-D of length M")
    if rel_estimates.shape != (len(rel_edges),):
        raise ValueError("rel_estimates must have length E")
    if not (np.isfinite(abs_estimates).all() and np.isfinite(rel_estimates).all()):
        raise ValueError("estimates must be finite")

    # Drop out-of-range and zero-weight edges (shared with estimate_voicing).
    keep = _edge_keep_mask(rel_edges, rel_confidences, M)
    rel_edges = rel_edges[keep]
    rel_estimates = rel_estimates[keep]
    rel_confidences = rel_confidences[keep]

    # Floor the absolute weights: with exact zeros the dual's ground arcs are
    # pinned to zero flow, which can make the circulation degenerate; 1e-6 is
    # negligible next to real confidence weights but keeps the LP well-posed.
    abs_conf_lp = np.maximum(abs_confidences, 1e-6)
    E = len(rel_estimates)

    # B is the M x (M+E) node-arc incidence matrix (flow conservation per frame):
    #   arcs 0..M-1    ground -> node i, carrying the absolute estimates
    #   arcs M..M+E-1  node u -> node v, carrying the relative differences
    # By strong duality the multipliers of B y = 0 (-res.eqlin.marginals) are
    # the optimal node potentials, i.e. the smoothed contour.
    num_arcs = M + E
    all_deltas = np.concatenate([abs_estimates, rel_estimates])
    all_weights = np.concatenate([abs_conf_lp, rel_confidences])

    rel_cols = np.arange(M, num_arcs)
    rows = np.concatenate([np.arange(M), rel_edges[:, 0], rel_edges[:, 1]])
    cols = np.concatenate([np.arange(M), rel_cols, rel_cols])
    data = np.concatenate([np.ones(M), -np.ones(E), np.ones(E)])
    B = csr_matrix((data, (rows, cols)), shape=(M, num_arcs))
    bounds = list(zip(-all_weights, all_weights))
    res = linprog(
        c=-all_deltas,
        A_eq=B,
        b_eq=np.zeros(M),
        bounds=bounds,
        method="highs",
    )
    if not res.success:  # pragma: no cover -- HiGHS fails only on degenerate input
        raise ValueError("LP smoothing failed: " + res.message)
    return -res.eqlin.marginals

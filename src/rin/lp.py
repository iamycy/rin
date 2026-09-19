"""Network-flow LP smoothing of absolute pitch with relative pitch constraints.

Given per-frame absolute pitch estimates (e.g. from any F0 tracker) and a set
of inter-frame relative pitch differences with confidences, this solves the
weighted L1 fusion problem

    min_f  sum_n w_n |f[n] - a[n]| + sum_{(u,v) in E} w_{uv} |f[v] - f[u] - d_{uv}|

as a linear program. Two equivalent formulations are available:

* ``dual_form=True`` (default): the Lagrangian dual, a min-cost network
  circulation problem on a graph with one auxiliary ground node. The optimal
  primal contour is read off as the Lagrange multipliers (node potentials) of
  the flow-conservation constraints, so only one (smaller) LP is solved.
* ``dual_form=False``: the primal with positive/negative slack variables.

Both are solved with the HiGHS simplex solver bundled with SciPy.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def lp_smoother(
    abs_estimates: np.ndarray,
    abs_confidences: np.ndarray,
    rel_edges: np.ndarray,
    rel_estimates: np.ndarray,
    rel_confidences: np.ndarray,
    dual_form: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth absolute pitch estimates with relative pitch constraints.

    Args:
        abs_estimates: (M,) absolute pitch per frame, in cents.
        abs_confidences: (M,) confidence/weight of each absolute estimate.
        rel_edges: (E, 2) frame-index pairs (u, v) for each relative edge.
        rel_estimates: (E,) pitch difference f[v] - f[u] per edge, in cents.
        rel_confidences: (E,) confidence/weight of each relative estimate.
            Zero-weight edges are dropped (they cannot bind the optimum).
        dual_form: solve the dual network-circulation LP (default) instead of
            the primal slack-variable LP. Both give the same optimum.

    Returns:
        smooth_pitch: (M,) smoothed pitch contour, in cents.
        new_voicing_probs: (M,) voicing probabilities fusing the absolute
            confidences with the distance-weighted (1/m^2) RMS of the incident
            relative-edge confidences, via a geometric mean.

    Raises:
        ValueError: if shapes disagree, values are non-finite, confidences
            are negative, or ``rel_edges`` is not an ``(E, 2)`` integer array.

    Notes:
        Edges referencing frames outside ``[0, M)`` are dropped, as are
        zero-weight edges (they cannot bind the optimum in either form).
    """
    abs_estimates = np.asarray(abs_estimates, dtype=float)
    abs_confidences = np.asarray(abs_confidences, dtype=float)
    rel_estimates = np.asarray(rel_estimates, dtype=float)
    rel_confidences = np.asarray(rel_confidences, dtype=float)
    rel_edges = np.asarray(rel_edges)
    if rel_edges.size == 0:
        rel_edges = np.zeros((0, 2), dtype=int)
    if (
        abs_estimates.ndim != 1
        or abs_confidences.shape != abs_estimates.shape
        or rel_edges.ndim != 2
        or rel_edges.shape[1] != 2
    ):
        raise ValueError(
            "expected abs_estimates/abs_confidences 1-D of length M and rel_edges of shape (E, 2)"
        )
    if not np.issubdtype(rel_edges.dtype, np.integer):
        raise ValueError("rel_edges must contain integer frame indices")
    if rel_estimates.shape != (len(rel_edges),) or rel_confidences.shape != (len(rel_edges),):
        raise ValueError("rel_estimates and rel_confidences must have length E")
    if not (
        np.isfinite(abs_estimates).all()
        and np.isfinite(abs_confidences).all()
        and np.isfinite(rel_estimates).all()
        and np.isfinite(rel_confidences).all()
    ):
        raise ValueError("estimates and confidences must be finite")
    if (abs_confidences < 0).any() or (rel_confidences < 0).any():
        raise ValueError("confidences must be non-negative")

    M = abs_estimates.size
    # Drop edges referencing frames outside [0, M): they carry no constraint.
    valid = np.all((rel_edges >= 0) & (rel_edges < M), axis=1)
    rel_edges = rel_edges[valid]
    rel_estimates = rel_estimates[valid]
    rel_confidences = rel_confidences[valid]

    # drop zero-weight relative edges: in the dual formulation, zero capacity
    # bounds [-0, 0] allow no flow; in the primal, zero-cost slack variables
    # would make the LP unbounded. Neither binds on f or affects the optimum.
    mask = rel_confidences > 0
    rel_edges = rel_edges[mask]
    rel_estimates = rel_estimates[mask]
    rel_confidences = rel_confidences[mask]

    abs_conf_lp = np.maximum(abs_confidences, 1e-6)
    E = len(rel_estimates)

    if dual_form:
        # Dual network flow formulation (equivalent to circulation network simplex):
        # Solves the Lagrangian dual on an augmented graph with an auxiliary ground
        # node (g). B is the M x (M+E) node-arc incidence matrix (flow conservation
        # at each frame node).
        # - Arcs 0..M-1 connect ground node -> node i (carrying observed absolute pitch).
        # - Arcs M..M+E-1 connect node u -> node v (carrying relative pitch differences).
        # By LP strong duality, the Lagrange multipliers of the node conservation
        # constraints B y = 0 (-res.eqlin.marginals) are the optimal node potentials (f).
        num_arcs = M + E
        all_deltas = np.concatenate([abs_estimates, rel_estimates])
        all_weights = np.concatenate([abs_conf_lp, rel_confidences])

        if E > 0:
            rel_cols = np.arange(M, num_arcs)
            rows = np.concatenate([np.arange(M), rel_edges[:, 0], rel_edges[:, 1]])
            cols = np.concatenate([np.arange(M), rel_cols, rel_cols])
            data = np.concatenate([np.ones(M), -np.ones(E), np.ones(E)])
        else:
            rows = np.arange(M)
            cols = np.arange(M)
            data = np.ones(M)

        B = csr_matrix((data, (rows, cols)), shape=(M, num_arcs))
        bounds = list(zip(-all_weights, all_weights))
        res = linprog(
            c=-all_deltas,
            A_eq=B,
            b_eq=np.zeros(M),
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise ValueError("LP smoothing failed: " + res.message)
        smooth_pitch = -res.eqlin.marginals
    else:
        # Primal formulation with positive/negative slack variables e_a+, e_a-, e_r+, e_r-:
        # Variables: [f, e_a+, e_a-, e_r+, e_r-] of total dimension 3*M + 2*E.
        tmp = [
            (np.ones(M), np.arange(M), np.arange(M)),
            (np.ones(M), np.arange(M), np.arange(M) + M),
            (-np.ones(M), np.arange(M), np.arange(M) + M * 2),
        ]
        row_offset = M
        col_offset = M * 3
        tmp += [
            (
                -np.ones(E),
                np.arange(E) + row_offset,
                rel_edges[:, 0],
            ),
            (
                np.ones(E),
                np.arange(E) + row_offset,
                rel_edges[:, 1],
            ),
            (
                np.ones(E),
                np.arange(E) + row_offset,
                np.arange(E) + col_offset,
            ),
            (
                -np.ones(E),
                np.arange(E) + row_offset,
                np.arange(E) + col_offset + E,
            ),
        ]

        vals, rows, cols = [np.concatenate(x) for x in zip(*tmp)]
        weights = np.concatenate(
            [
                np.zeros_like(abs_estimates),
                abs_conf_lp,
                abs_conf_lp,
                rel_confidences,
                rel_confidences,
            ]
        )
        b_eq = np.concatenate([abs_estimates, rel_estimates])
        A_eq = csr_matrix((vals, (rows, cols)), shape=(b_eq.size, col_offset + E * 2))
        n_slack = 2 * (M + E)
        bounds = [(None, None)] * M + [(0, None)] * n_slack
        res = linprog(
            c=weights,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise ValueError("LP smoothing failed: " + res.message)
        smooth_pitch = res.x[:M]

    # Distance-weighted root-mean-square (RMS, p=2 with 1/m^2 distance decay)
    # voicing fusion: integrates incident relative edge confidences while giving
    # higher priority to short-hop edges (d = 1/m^2), combined with the tracker's
    # absolute confidence (s in [0, 1]) via a fuzzy AND (geometric mean).
    hops = np.maximum(np.abs(rel_edges[:, 1] - rel_edges[:, 0]), 1)
    d = 1.0 / (hops.astype(float) ** 2)
    num_sparse = csr_matrix(
        (d * (rel_confidences**2), (rel_edges[:, 0], rel_edges[:, 1])), shape=(M, M)
    )
    num_sym = num_sparse + num_sparse.T
    num = num_sym.sum(axis=1).A1

    den_sparse = csr_matrix((d, (rel_edges[:, 0], rel_edges[:, 1])), shape=(M, M))
    den_sym = den_sparse + den_sparse.T
    den = den_sym.sum(axis=1).A1

    rel_rms = np.sqrt(np.divide(num, np.maximum(den, 1e-10)))
    new_voicing_probs = np.sqrt(np.clip(abs_confidences, 0.0, 1.0) * np.clip(rel_rms, 0.0, 1.0))

    return smooth_pitch, new_voicing_probs

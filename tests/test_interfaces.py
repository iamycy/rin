"""Contract test: the built-in cores match the documented signatures.

The package exposes exactly three core functions, each obeying one of the
``rin.interfaces`` signatures. Swapping one out is just calling a different
function -- no wrapper, no registration -- which makes the positional
parameter names part of the public contract. Injection through
``smooth_pitch`` is covered in ``test_pipeline.py``.
"""

import inspect

from rin import estimate_voicing, lp_smoother, vqt_diff_calculator


def test_builtin_cores_match_documented_signatures():
    est_params = list(inspect.signature(vqt_diff_calculator).parameters)
    assert est_params[:4] == ["x", "sr", "hop_length", "hops"]
    solver_params = list(inspect.signature(lp_smoother).parameters)
    assert solver_params[:5] == [
        "abs_estimates",
        "abs_confidences",
        "rel_edges",
        "rel_estimates",
        "rel_confidences",
    ]
    voicing_params = list(inspect.signature(estimate_voicing).parameters)
    assert voicing_params == ["abs_confidences", "rel_edges", "rel_confidences"]

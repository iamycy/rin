"""Small numerical helpers shared by the rin-pitch modules."""

import numpy as np


def parabolic_interpolation(a, b, c):
    """Parabolic interpolation over a three-point stencil.

    Fits a parabola through the stencil values ``(a, b, c)`` and returns the
    interpolated peak value and its offset from the center bin.

    Parameters
    ----------
    a, b, c : np.ndarray
        Stencil values at bins ``(-1, 0, +1)``.

    Returns
    -------
    value : np.ndarray
        Interpolated peak value.
    offset : np.ndarray
        Offset of the peak from the center bin, in bins.

    Notes
    -----
    Kept local on purpose: librosa's equivalent (``_parabolic_interpolation``
    in ``librosa.core.pitch``) is private, takes the full array instead of a
    stencil triple, and has different edge semantics, so it is not a drop-in
    replacement.
    """
    # Parabola through three stencil values; return the interpolated value
    # and the offset from the center bin.
    denom = a - 2 * b + c
    offset = np.divide(
        0.5 * (a - c),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=np.abs(denom) > 1e-10,
    )
    return b - 0.25 * (a - c) * offset, offset

"""Small numerical helpers shared by the rin-pitch modules."""

import numpy as np


def hz2cent(f):
    return 1200 * np.log2(f)


def cent2hz(c):
    return 2 ** (c / 1200)


def parabolic_interpolation(a, b, c):
    # return the interpolated value and the offset from the center
    denom = a - 2 * b + c
    offset = np.divide(
        0.5 * (a - c),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=np.abs(denom) > 1e-10,
    )
    return b - 0.25 * (a - c) * offset, offset

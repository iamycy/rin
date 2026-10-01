# RIN (Relative Interval Networks)

[![arXiv](https://img.shields.io/badge/arXiv-2609.39852-b31b1b.svg)](https://arxiv.org/abs/2609.39852)
[![PyPI](https://img.shields.io/pypi/v/rin-pitch)](https://pypi.org/project/rin-pitch/)
[![Python versions](https://img.shields.io/pypi/pyversions/rin-pitch)](https://pypi.org/project/rin-pitch/)
[![CI](https://github.com/iamycy/rin/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/iamycy/rin/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/iamycy/rin/branch/main/graph/badge.svg)](https://app.codecov.io/gh/iamycy/rin)
[![License: MIT](https://img.shields.io/pypi/l/rin-pitch)](https://github.com/iamycy/rin/blob/main/LICENSE)

This is the reference implementation accompanying the paper [Pitch Smoothing Using Relative Interval Networks](https://arxiv.org/abs/2609.39852), submitted to ICASSP 2027.
We propose a method for smoothing pitch estimates by combining absolute pitch measurements with multi-hop relative pitch differences, using a network-flow linear program for optimal fusion.
Specifically, we solve the following objective function:

```math
\min_{\mathbf{f}} \mathbf{w}^\top \lvert \mathbf{B}^\top \mathbf{f} - \mathbf{\Delta} \rvert,
```
where $\mathbf{f}$ is the vector of smoothed pitch estimates, $\mathbf{\Delta}$ is the vector of observed absolute pitch and **relative pitch differences**, $\mathbf{B}$ is the incidence matrix encoding the absolute and relative edges, and $\mathbf{w}$ is a vector of weights.

## Install

With [pixi](https://pixi.sh) (recommended):

```sh
pixi install
```

Or with pip:

```sh
pip install rin-pitch
```

## Quickstart

```python
import numpy as np
from rin import smooth_pitch

# x: mono waveform, sr: sample rate
# f0: (M,) absolute pitch in cents on a 20 ms grid, NaN = unvoiced
# strength: (M,) voicing confidence in [0, 1]
hop_length = int(0.02 * sr)

# the paper's pipeline in one call: VQT relative diffs -> dual LP fusion
# -> voicing, with the paper's fixed settings (hops 1,2,3,5; Pearson xcorr;
# arcsin x peak2mean weighting)
f0_smooth, voicing = smooth_pitch(x, f0, strength, sr, hop_length)
```

The package never converts pitch units: absolute and relative estimates must share one pitch domain (the built-in estimator outputs cents), and converting to or from your own domain (Hz, MIDI, ...) is your responsibility.

If your tracker reports Hz, convert before calling; `smooth_pitch` expects cents and will not convert for you:

```python
voiced = np.isfinite(f0_hz)
f0_cents = np.where(voiced, 1200 * np.log2(np.where(voiced, f0_hz, 1.0)), np.nan)
f0_smooth, voicing = smooth_pitch(x, f0_cents, strength, sr, hop_length)
```

Arrays in, arrays out: the package does no file loading and no caching.

## API

One high-level function plus three cores:

- `rin.smooth_pitch(x, f0, strength, sr, hop_length, hops=(1, 2, 3, 5), ...)` runs the paper's pipeline in one call: VQT relative diffs, dual LP fusion, and voicing.
  `f0` in cents with NaN = unvoiced; returns `(f0_smooth, voicing)` in cents.
  `f0` must sit on the same `hop_length` grid as the audio, and `smooth_pitch` raises if it does not.
  Each stage is swappable via `difference_estimator`, `solver`, and `voicing_estimator` keyword arguments (any callable obeying the `rin.interfaces` contracts); configure a stage with `functools.partial`, e.g. `difference_estimator=partial(vqt_diff_calculator, max_diff_cents=500.0)`.

Three core functions (for custom wiring):

- `rin.vqt_diff_calculator(x, sr, hop_length, hops=(1,), ...)` returns multi-hop relative pitch differences (cents) with confidences.
  Uses the paper's fixed estimation path: Pearson (mean-subtracted) normalized cross-correlation of VQT magnitude slices with the arcsin x peak2mean confidence weighting.
  `max_diff_cents` (600), `bins_per_octave` (36), `n_bins` (252), and other VQT options are plain kwargs, so pass your own.
  `max_diff_cents` must stay inside the VQT's range: it has to buy at least one bin of search and fewer than `n_bins` of it, or the correlation window runs off the spectrogram.
- `rin.lp_smoother(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences)` does network-flow LP fusion (dual min-cost circulation, HiGHS); absolute and relative estimates must share one pitch domain (caller's choice, e.g. cents).
- `rin.estimate_voicing(abs_confidences, rel_edges, rel_confidences)` returns per-frame voicing probabilities from the two confidence streams.

## Plugins

Each core is a plain function obeying a contract in `rin.interfaces` (`DifferenceEstimator`, `Solver`, and `VoicingEstimator` are `Callable` type aliases).
Implement your own function with the same signature and pass it to `smooth_pitch`, or call it directly in your own wiring.
No classes or inheritance needed; any callable (function, lambda, `functools.partial`, callable object) works:

```python
from functools import partial

from rin import smooth_pitch, vqt_diff_calculator


def my_estimator(x, sr, hop_length, hops):
    # -> (edges (E,2) int, estimates (E,) cents, confidences (E,) in [0,1])
    ...


def my_solver(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    # -> smooth_pitch (M,) cents
    ...


def my_voicing(abs_confidences, rel_edges, rel_confidences):
    # -> voicing (M,) in [0,1]
    ...


# One call, custom stages (tune a stage via functools.partial):
f0_smooth, voicing = smooth_pitch(
    x,
    f0,
    strength,
    sr,
    hop_length,
    difference_estimator=partial(vqt_diff_calculator, max_diff_cents=500.0),
    solver=my_solver,
    voicing_estimator=my_voicing,
)

# ...or wire them by hand:
edges, estimates, confidences = my_estimator(x, sr, hop_length, hops=(1, 2, 3, 5))
smooth_cents = my_solver(f0_cents, abs_conf, edges, estimates, confidences)
voicing = my_voicing(abs_conf, edges, confidences)
```

## Development

```sh
pixi run test        # pytest with branch coverage
pixi run lint        # ruff check
pixi run docstrings  # numpydoc validation of the public API
pixi run format      # ruff format
pixi run build       # sdist + wheel
pixi run smoke       # install the built wheel in a clean venv and check its version
```

CI runs `lint` and `docstrings` once, and `test` on Linux, macOS and Windows across Python 3.10-3.13 (`pixi run -e py310 test` reproduces one cell locally).

The version comes from the git tag (no file in the repo declares one), so a release is `git tag v1.1.0 && git push --tags`, which builds and publishes to PyPI.

## License

MIT

# rin-pitch

Relative Interval Network (RIN) pitch smoothing: fuse per-frame absolute pitch
estimates (from any F0 tracker) with multi-hop relative pitch differences
estimated from a magnitude VQT, through a network-flow linear program.

This is the reference implementation accompanying the paper
"Better than Viterbi: ..." (ICASSP 2027).

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
# f0: (M,) absolute pitch in Hz on a 20 ms grid, NaN = unvoiced
# strength: (M,) voicing confidence in [0, 1]
hop_length = int(0.02 * sr)

# the paper's pipeline in one call: VQT relative diffs -> dual LP fusion
# -> voicing, with the paper's fixed settings (hops 1,2,3,5; Pearson xcorr;
# arcsin x peak2mean weighting)
f0_smooth, voicing = smooth_pitch(x, f0, strength, sr, hop_length)
```

For custom behavior, wire the three cores yourself:

```python
from rin import vqt_diff_calculator, lp_smoother, estimate_voicing

# 1. relative pitch differences between frames (cents), with confidences
edges, estimates, confidences = vqt_diff_calculator(x, sr, hop_length, hops=(1, 2, 3, 5))

# 2. fuse with the absolute estimates through the network-flow LP (cents in/out).
#    Unvoiced (NaN) frames get zero absolute confidence, so the LP bridges
#    them through the relative edges alone.
voiced = np.isfinite(f0)
f0_cents = 1200 * np.log2(np.where(voiced, f0, 1.0))
abs_conf = np.where(voiced, strength, 0.0)
smooth_cents = lp_smoother(f0_cents, abs_conf, edges, estimates, confidences)

# 3. per-frame voicing from the absolute and relative confidences
voicing = estimate_voicing(abs_conf, edges, confidences)
f0_smooth = 2 ** (smooth_cents / 1200)
```

Arrays in, arrays out -- the package does no file loading and no caching.
Each caller wires their own pipeline (and their own VQT caching) around the
three cores.

## API

One high-level function plus three cores:

- `rin.smooth_pitch(x, f0_hz, strength, sr, hop_length, hops=(1, 2, 3, 5), ...)`
  -- the paper's pipeline in one call: VQT relative diffs, dual LP fusion,
  and voicing. `f0_hz` in Hz with NaN = unvoiced; returns `(f0_smooth_hz,
  voicing)`. Extra kwargs (`max_diff_cents`, `bins_per_octave`, `n_bins`,
  ...) go to the estimator.

Three core functions (for custom wiring):

- `rin.vqt_diff_calculator(x, sr, hop_length, hops=(1,), ...)` -- multi-hop
  relative pitch differences (cents) with confidences. Uses the paper's
  fixed estimation path: Pearson (mean-subtracted) normalized
  cross-correlation of VQT magnitude slices with the arcsin x peak2mean
  confidence weighting. `max_diff_cents`, `bins_per_octave`, and other VQT
  options are plain kwargs -- pass your own.
- `rin.lp_smoother(abs_estimates, abs_confidences, rel_edges, rel_estimates,
  rel_confidences)` -- network-flow LP fusion (dual min-cost circulation,
  HiGHS); inputs and output in cents.
- `rin.estimate_voicing(abs_confidences, rel_edges, rel_confidences)` --
  per-frame voicing probabilities from the two confidence streams.

## Plugins

Each core is a plain function obeying a contract in `rin.interfaces`
(`DifferenceEstimator`, `Solver`, and `VoicingEstimator` are `Callable`
type aliases). Implement your own function with the same signature and call
it instead -- no classes or inheritance needed; any callable (function,
lambda, `functools.partial`, callable object) works:

```python
def my_estimator(x, sr, hop_length, hops):
    # -> (edges (E,2) int, estimates (E,) cents, confidences (E,) in [0,1])
    ...

def my_solver(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    # -> smooth_pitch (M,) cents
    ...

def my_voicing(abs_confidences, rel_edges, rel_confidences):
    # -> voicing (M,) in [0,1]
    ...

edges, estimates, confidences = my_estimator(x, sr, hop_length, hops=(1, 2, 3, 5))
smooth_cents = my_solver(f0_cents, abs_conf, edges, estimates, confidences)
voicing = my_voicing(abs_conf, edges, confidences)
```

## Development

```sh
pixi run test    # pytest
pixi run lint    # ruff check
pixi run format  # ruff format
pixi run build   # sdist + wheel
```

CI runs `lint` + `test` on Linux and macOS via pixi; pushing a `v*` tag
builds and publishes to PyPI.

## License

MIT

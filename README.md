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
from rin import lp_smoother, vqt_diff_calculator

# x: mono waveform, sr: sample rate
# f0: (M,) absolute pitch in Hz on a 20 ms grid, NaN = unvoiced
# strength: (M,) voicing confidence in [0, 1]
hop_length = int(0.02 * sr)

# 1. relative pitch differences between frames (cents), with confidences
edges, estimates, confidences = vqt_diff_calculator(x, sr, hop_length, hops=(1, 2, 3, 5))

# 2. fuse with the absolute estimates through the network-flow LP (cents in/out).
#    Unvoiced (NaN) frames get zero absolute confidence, so the LP bridges
#    them through the relative edges alone.
voiced = np.isfinite(f0)
f0_cents = 1200 * np.log2(np.where(voiced, f0, 1.0))
abs_conf = np.where(voiced, strength, 0.0)
smooth_cents, voicing = lp_smoother(f0_cents, abs_conf, edges, estimates, confidences)
f0_smooth = 2 ** (smooth_cents / 1200)
```

Arrays in, arrays out -- the package does no file loading and no caching.
Each caller wires their own pipeline (and their own VQT caching) around the
two cores.

## API

Two core functions:

- `rin.vqt_diff_calculator(x, sr, hop_length, hops=(1,), ...)` -- multi-hop
  relative pitch differences (cents) with confidences. Extra VQT options
  (`max_diff_cents`, `bins_per_octave`, `weighting`, `corr_mode`, ...) are
  plain kwargs -- pass your own.
- `rin.lp_smoother(abs_estimates, abs_confidences, rel_edges, rel_estimates,
  rel_confidences, dual_form=True)` -- network-flow LP fusion; inputs and
  outputs in cents.

## Plugins

Both cores are plain functions obeying the contracts in `rin.interfaces`
(`DifferenceEstimator` and `Solver` are `Callable` type aliases). Implement
your own function with the same signature and call it instead -- no classes
or inheritance needed; any callable (function, lambda, `functools.partial`,
callable object) works:

```python
def my_estimator(x, sr, hop_length, hops):
    # -> (edges (E,2) int, estimates (E,) cents, confidences (E,) in [0,1])
    ...

def my_solver(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    # -> (smooth_pitch (M,) cents, voicing (M,) in [0,1])
    ...

edges, estimates, confidences = my_estimator(x, sr, hop_length, hops=(1, 2, 3, 5))
smooth_cents, voicing = my_solver(f0_cents, strength, edges, estimates, confidences)
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

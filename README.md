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
import rin

# f0: (M,) absolute pitch in Hz on a 20 ms grid, NaN = unvoiced
# strength: (M,) voicing confidence in [0, 1]
# x: mono waveform, sr: sample rate
f0_smooth, voicing = rin.smooth(f0, strength, x, sr, return_voicing=True)
```

Arrays in, arrays out -- the package does no file loading.

## API

- `rin.smooth(f0, strength, x, sr, hops=(1, 2, 3, 5), ...)` -- high-level entry point.
- `rin.rin_smooth(f0, strength, hop_cache, hops, ...)` -- fuse with a precomputed hop cache.
- `rin.build_hop_cache(x, sr, hops, ...)` -- VQT once, per-hop `(edges, estimates, confidences)`.
- `rin.lp_smoother(...)` -- the network-flow LP core (dual circulation form by default).
- `rin.vqt_diff_calculator(...)` -- multi-hop VQT relative pitch differences.

## Plugins

Both stages are plain function signatures documented in `rin.interfaces` --
implement a function with the right signature and pass it in. No classes or
inheritance needed; any callable (function, lambda, `functools.partial`,
callable object) works:

```python
def my_estimator(x, sr, hop_length, hops):
    # -> (edges (E,2) int, estimates (E,) cents, confidences (E,) in [0,1])
    ...

def my_solver(abs_estimates, abs_confidences, rel_edges, rel_estimates, rel_confidences):
    # -> (smooth_pitch (M,) cents, voicing (M,) in [0,1])
    ...

f0_smooth = rin.smooth(f0, strength, x, sr, estimator=my_estimator, solver=my_solver)
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

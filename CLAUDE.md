# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`rin-pitch` is the reference implementation for "Pitch Smoothing Using Relative Interval Networks" (ICASSP 2027).
It fuses per-frame absolute F0 estimates (from any external tracker) with multi-hop relative pitch differences estimated from a magnitude VQT, via a network-flow linear program.

## Commands

Development runs through [pixi](https://pixi.sh); the `default` environment carries both the `test` and `dev` features:

```sh
pixi install
pixi run test          # pytest -v with branch coverage, writes coverage.xml
pixi run lint          # ruff check src tests
pixi run format        # ruff format src tests
pixi run build         # python -m build (sdist + wheel)
pixi run smoke         # install the built wheel in a clean venv, check its version
pixi run docstrings    # numpydoc validation of the public API
```

`py310`, `py311`, `py312` and `py313` pin one interpreter each for the CI matrix, e.g. `pixi run -e py310 test`.
They carry the `test` feature but not `dev`: `numpydoc >=1.11` requires Python >=3.11, so bundling the tooling in would make a 3.10 environment unsolvable.
Keep the runner in `test` and the tooling in `dev`.

Run a single test or file:

```sh
pixi run pytest tests/test_lp.py -v
pixi run pytest tests/test_lp.py::test_dual_matches_primal -v
pixi run pytest -k voicing -v
```

CI (`.github/workflows/ci.yml`) runs `lint` + `docstrings` once, then `test` across a 3-OS x 4-Python matrix, plus a `build` job ending in `smoke`.
Every leg writes `coverage.xml` (the `test` task carries `--cov --cov-branch --cov-report=xml`), but only the `ubuntu-latest` + `py313` leg uploads to Codecov, since no code path here is platform or version specific.
That leg is picked out by a matrix `include` key rather than an `if` comparing matrix values: if `py313` is ever dropped from the list, `include` re-adds it as a leg that fails on `pixi run -e py313`, instead of the condition quietly never matching and coverage uploads stopping unnoticed.
Coverage scope lives in `[tool.coverage.run]` in `pyproject.toml`, which uses `omit` rather than `source`: `source` rewrites the XML to bare basenames under a `src/rin` prefix and Codecov then fails to match them against the repo.
Pushing a `v*` tag triggers `release.yml`, which lints, checks docstrings, tests, builds, smoke-tests the wheel, and publishes to PyPI via trusted publishing.
Both workflows pass `locked: true` to `setup-pixi`, so a `pixi.toml` edit that was not re-locked fails CI instead of silently resolving to something else.

Ruff config lives in `pyproject.toml`: line length 100, rules `D, E, F, W, I, UP`.

## Conventions

**Docstrings are numpydoc.** Enforced two ways: ruff's `D` rules with `convention = "numpy"` (tests are exempt via `per-file-ignores`, since test names describe themselves), and `pixi run docstrings`, configured by `[tool.numpydoc_validation]` in `pyproject.toml`.
Private helpers are excluded there and keep a one-line summary by design; do not give them full Parameters/Returns blocks.
Note numpydoc names a return value only when a function returns *several*; single-return functions document the bare type (`RT02`).

**Never use a dash as punctuation.** No `--`, no `—`, no `–`, anywhere in the repo: Markdown, docstrings, comments, commit messages and PR descriptions alike.
Use a colon, a semicolon, a comma, parentheses, or a new sentence, whichever fits the relation the dash was standing in for.
Only non-punctuation uses survive: CLI flags (`--tags`), numpydoc section underlines (`----------`), Markdown table rules (`| --- |`), arithmetic minus signs, and hyphenated words.

**Markdown files are never hard-wrapped.** One complete sentence per source line, however long, including `README.md` and this file.
Never reflow them to a column limit: the `line-length = 100` in `pyproject.toml` is Ruff's and governs Python only.
Sentence-per-line keeps diffs tight, since rewording one sentence touches one line.
Code fences, tables and list markers are unaffected; it is prose that stays unbroken.

## Architecture

Three core functions plus one pipeline that chains them.
Arrays in, arrays out: **no file loading, no caching, no classes, no registry**.
This is deliberate: callers wire their own pipeline (and their own VQT caching) around the cores.

| Module | Role |
| --- | --- |
| [src/rin/relative.py](src/rin/relative.py) | `vqt_diff_calculator`: multi-hop relative pitch differences from audio |
| [src/rin/lp.py](src/rin/lp.py) | `lp_smoother` (LP fusion) and `estimate_voicing` |
| [src/rin/pipeline.py](src/rin/pipeline.py) | `smooth_pitch`: chains the three stages with the paper's defaults |
| [src/rin/interfaces.py](src/rin/interfaces.py) | `Callable` type aliases documenting the three stage contracts |

### Plugin model

`DifferenceEstimator`, `Solver`, and `VoicingEstimator` in `interfaces.py` are `typing.Callable` aliases, not base classes.
Any callable with the right positional signature works (function, lambda, `functools.partial`, callable object).
`smooth_pitch` takes no stage-specific kwargs by design; configure a stage with `functools.partial`, e.g. `difference_estimator=partial(vqt_diff_calculator, max_diff_cents=500.0)`.
`tests/test_interfaces.py` asserts the built-in cores' parameter names match the documented contracts, so renaming a core's positional parameters is a breaking change and will fail there.

### Invariants that cut across modules

- **Pitch domain is never converted.** `abs_estimates` and `rel_estimates` must already share one domain; the output is in that same domain.
  The built-in estimator emits cents, so `smooth_pitch` documents cents in / cents out, but `lp_smoother` itself is domain-agnostic.
  Hz↔cents conversion is the caller's responsibility and must not be added inside the package.
- **`f0` defines the frame grid, unchecked.** `smooth_pitch` treats `len(f0)` as authoritative and never compares it against a frame count derived from `x`, because trackers disagree on the exact count (librosa counts the frame centred on the clip's last sample, libf0's SWIPE and PENN do not).
  Estimator edges that overshoot `len(f0)` are dropped downstream by the shared `_edge_keep_mask`.
  Only negative endpoints are rejected up front, since no framing convention justifies them and they would otherwise vanish silently.
  There is deliberately no equality check against `edges.max() + 1`, which is only a lower bound on the frame count, and an estimator may legitimately leave trailing frames unconnected.
- **Unvoiced frames are expressed as zero absolute confidence,** not as a separate mask.
  `smooth_pitch` maps non-finite `f0` to `abs_est = 0.0, abs_conf = 0.0`, and the LP positions those frames from the relative edges.
  (The `1e-6` weight floor in `lp_smoother` leaves a residual pull toward `0.0`, negligible beside real edge weights.)
- **Relative evidence is undirected.** `estimate_voicing` accumulates each edge onto *both* endpoints (`np.bincount` over `u` and over `v`), so edge direction, duplicate edges and self-loops all fall out correctly with no special-casing.
  It was built from two `M × M` sparse adjacency matrices before; `tests/test_voicing.py` keeps that form as `_sparse_voicing` and asserts equivalence on random adversarial graphs, the way `test_lp.py` keeps a primal LP for the dual solver.
  `bincount` is ~10× faster, so don't reintroduce the sparse version.
- **Edge filtering is shared.** `_edge_keep_mask` in `lp.py` drops out-of-range and zero-confidence edges, and both `lp_smoother` and `estimate_voicing` apply it, so the two stages always see the same edge set.

### LP formulation (the subtle part)

`lp_smoother` solves the weighted L1 problem

```
min_f  Σ_n w_n |f[n] − a[n]| + Σ_(u,v)∈E w_uv |f[v] − f[u] − d_uv|
```

by solving its **Lagrangian dual** (a min-cost circulation on a graph with one auxiliary ground node) instead of the primal.
Consequences when editing:

- `B` is the `M × (M+E)` node-arc incidence matrix: arcs `0..M-1` are ground→node (absolute terms), arcs `M..M+E-1` are `u→v` (relative terms).
- The objective is `c = all_deltas` and the answer is `res.eqlin.marginals` (node potentials).
  The two signs are coupled, not independent: the feasible set (`B y = 0`, bounds `±w`) is invariant under `y → -y`, so negating `c` negates the marginals exactly and flipping both together is a no-op, which is why the older `c = -all_deltas` / `-marginals` pairing gave identical contours.
  Flipping only one silently negates the contour.
- Arc bounds are `(-w, +w)`; absolute weights are floored at `1e-6` because exact zeros pin the ground arcs and make the circulation degenerate.
- `tests/test_lp.py` contains `_primal_smoother`, an explicit primal LP with slack variables, used as the reference oracle for the dual.
  Any change to the dual formulation must keep that equivalence test passing.

### VQT relative estimation

`relative.py` implements one fixed estimation path, the paper's, and it is intentionally not configurable: Pearson (mean-subtracted) normalized cross-correlation of VQT magnitude slices, with confidence = `arcsin` dot weight × `peak2mean` flatness weight (`_dot_weight` × `_flat_weight`).
VQT shape parameters (`bins_per_octave=36`, `n_bins=252`, `max_diff_cents=600`) and extra `librosa.vqt` kwargs are plain arguments.
The paper's hop set is `HOPS = (1, 5)`, exported from `rin`.

Performance structure: `_compute_vqt` and `_vqt_xcorr_setup` are factored out because the padded VQT, sliding-window view, and window norms depend only on `(V, max_diff_bins)` and not on the hop, so they are computed once per clip and passed into every `_hop_diff` call.
Keep new per-hop work out of the setup and vice versa.
Both `_vqt_xcorr_setup` and `_hop_diff` force `V` Fortran-ordered: the correlation's two reduction axes are contiguous only under F order, and C order costs 4.7x.
`librosa.vqt` already returns F-ordered output, so the guards are no-ops in practice and `test_correlation_reduction_axes_stay_contiguous` is what makes their removal visible.

Peak picking rectifies correlations before `argmax`; frames whose peak lands on the search boundary get zero confidence via `hit_boundary`, which also covers the all-negative-column collapse to index 0.
Dropping the sub-bin offset at boundary peaks (`p = np.where(hit_boundary, 0.0, p)`) is what keeps every returned estimate inside the ±`max_diff_cents` window.
The stencil around a boundary peak is not a neighbourhood of it (`idx - 1 == -1` indexes from the far end), which is harmless only because both consumers of that stencil (`p` and `diff_probs`) are zeroed at `hit_boundary`.
`max_diff_cents` is validated from both sides: at least one VQT bin of search, and strictly fewer than `n_bins` (at `n_bins` the overlap length `K_tau` hits zero and the correlations go NaN).

## Tests

All tests use synthetic signals (sines, chirps, seeded RNG); there are no audio fixtures, and tests should stay self-contained that way.
Coverage is split by module: `test_lp.py` (dual vs. primal, validation), `test_relative.py` (estimator accuracy on known chirps), `test_voicing.py`, `test_pipeline.py` (end-to-end plus the manual-wiring pattern), `test_interfaces.py` (signature conformance only).

## Release mechanics

**The git tag is the only source of truth for the version; no file in the repo declares one.** `hatch-vcs` derives it from `git describe` at build time, and `rin.__version__` reads the installed metadata through `importlib.metadata`.
Releasing is `git tag v1.1.0 && git push --tags`, which triggers `release.yml`.
Between tags the version is a dev string such as `0.1.dev18+g6afb39f`, so a working checkout will not show a clean number.

Both workflows check out with `fetch-depth: 0`.
This is load-bearing: `hatch-vcs` needs the full history and tags, and the default shallow clone would silently produce a `0.0.0` version and publish it to PyPI.
`pixi run smoke` is the backstop for exactly that: it installs the built wheel into a throwaway venv and fails unless `rin.__version__` matches the version in the wheel's own filename, which catches a shallow checkout and any other break in the tag → `hatch-vcs` → `importlib.metadata` chain before anything reaches PyPI.

`pixi build` is deliberately **not** configured (no `[package]` table, no `preview = ["pixi-build"]`).
Pixi requires a static `[project] version` and rejects `dynamic = ["version"]` with `There was no version defined for the recipe`, so its build backend is incompatible with tag-derived versioning.
Nothing depended on it: CI and release both run `pixi run build` (`python -m build`), and conda-forge builds from the PyPI sdist, whose `PKG-INFO` carries the resolved version.

`conda-recipe/meta.yaml` is prepared but **not submitted** to conda-forge.
Its header lists the submission steps, each of which needs explicit approval: the recipe's `sha256` is a placeholder until the first PyPI sdist exists, and per the owner's decision recorded there, PyPI publication is blocked before 2026-09-23.

# Getting started

A twenty-minute path from `pip install` to a solved instance, entirely
on the CPU. The GPU is what makes coldcore *fast*; it is not what makes
it *work*, and nothing on this page needs one.

## 1. Install

```bash
git clone https://github.com/Mapika/coldcore
cd coldcore
pip install -e .            # or: pip install -e ".[test]" for the suite
```

Requirements: Python 3.10+ and numpy. That is the whole runtime
dependency list, on purpose — the GPU bindings are ctypes, so there is
no pybind11/nanobind/torch in the tree.

> The editable install needs a pip that speaks PEP 660. Ubuntu 22.04
> ships pip 22.0.2, which refuses it with *"build backend is missing the
> 'build_editable' hook"*; `python -m pip install --upgrade pip` first
> and it works. A plain `pip install .` needs nothing special.

The distribution is pure Python (`py3-none-any`), so it installs the
same way on x86-64 and on aarch64 (Grace/GH200) with no compiler
involved. What you get from pip:

| you get | you do not get |
|---|---|
| the search core (`coldcore.search`) | the CUDA plugins |
| the CLI (`coldcore …`) | `libcoldcore.so` |
| the numpy reference backends | the big-memory placement policy |
| the backend protocol to write your own | |

Check the install from *outside* the checkout, so you know you are
testing the installed package and not the source directory:

```bash
cd /tmp
coldcore info
python -m coldcore --help
```

`coldcore info` prints the version and probes for a plugin `.so`;
reporting `plugin: not loadable` is the expected answer on a pip-only
install.

## 2. Solve something

`K_2(4,1)` — the smallest interesting covering code: binary words of
length 4, every word within Hamming distance 1 of a codeword. The
answer is 4.

```bash
coldcore solve --backend ref -p q=2 -p n=4 -p R=1 --floor 4
```

```
[coldcore] feasible at M=4; descending (floor=4)
[coldcore] SOLVED M=4
[coldcore] descent ended at M=4 stats={...}
```

`solve` runs greedy build → peel → (coset start, Hamming only) → notch
descent. Useful flags:

| flag | meaning |
|---|---|
| `--backend ref \| cpu \| gpu` | brute-force numpy oracle, the fast numpy axis-DP backend, or a CUDA plugin |
| `-p key=value` | problem parameters, repeatable (see below) |
| `--floor N` | stop descending at size N |
| `--hours H` | wall-clock budget for the whole run |
| `--out FILE` | write every improvement atomically (crash-safe) |
| `--seed-file FILE` | start from an existing solution instead of greedy |
| `--notch-budget S` | notch time unit: a notch ends after `2 S` seconds without a decrease of the uncovered count, or after `4 S` (default: a tenth of `--hours`, clamped to 10..300 s; 300 without `--hours`) |
| `--start greedy` | skip the coset start that runs by default on homogeneous Hamming cells with `--hours` ([cosets.md](cosets.md)) |

Problem parameters are `-p` pairs:

| parameter | meaning |
|---|---|
| `problem=` | `hamming` (default), `torus_linf`, `grid_linf`, `lee` |
| `q=`, `n=` | homogeneous alphabet: the cube `q^n` |
| `axes=9x9` | mixed-radix axes, instead of `q`/`n` |
| `R=` | coverage radius |
| `mu=` | µ-fold covering (every point covered ≥ µ times; default 1) |

So a 9×9 king-torus dominating set is:

```bash
coldcore solve --backend ref -p problem=torus_linf -p axes=9x9 -p R=1
```

Reference backends are brute force and capped at 2^16 points — they are
ground truth for parity tests and for learning the API, not a
production path.

## 3. Read the examples

Three self-contained scripts, each of which asserts its own expected
result, so they double as tests (`tests/test_examples.py` runs them):

```bash
python examples/01_covering_code.py    # K_3(4,1) = 9, K_3(4,2) = 3
python examples/02_domination.py       # 9x9 king torus = 9 kings
python examples/03_custom_problem.py   # a brand-new problem, ~90 lines
```

Read them in that order. The third is the one that matters if you came
here to extend coldcore: it adds *toroidal queen domination* — a problem
whose coverage neighborhood is not a metric ball at all — as a single
class implementing `coldcore.protocol.Backend`, and the stock search
core solves it to proven optimality with no changes anywhere else.

## 4. Use it from Python

The three-line version:

```python
import numpy as np
from coldcore.reference import RefCoveringBackend
from coldcore.search import Searcher

backend = RefCoveringBackend(3, 4, 1)          # K_3(4,1), optimum 9
solutions = []
s = Searcher(backend, seed=1,
             on_solved=lambda M, idx: solutions.append((M, idx.copy())))
s.load(np.zeros(0, dtype=np.int64))            # start empty
s.greedy_fill(target_m=10 ** 9)                # exact lazy greedy
s.descend(floor=9, notch_budget_s=10)          # notch ladder + LNS repair

M, idx = min(solutions, key=lambda t: t[0])    # -> 9
print(backend.index_word(idx))                 # (9, 4) digit rows
```

Two habits worth picking up early:

- **Read results from `on_solved`, not from `searcher.code`.** The
  callback fires once per *verified feasible* size. `searcher.code` is
  working state and can be mid-ruin when a budget expires.
- **Verify outside the framework.** House rule from `CONTRIBUTING.md`:
  the search core reports solutions, an independent check decides what
  is real. All three examples re-derive coverage from the digit rows
  without touching the backend that produced them.

`docs/api.md` is the tour of the rest of the surface.

## 5. GPU backends

The CUDA plugins are built by cmake, not by pip:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release    # needs CUDA toolkit >= 12
cmake --build build -j
```

This produces `build/lib/libcoldcore.so` and
`build/plugins/libcoldcore_covering.so`. Then `--backend gpu` works:

```bash
coldcore solve --backend gpu -p q=2 -p n=4 -p R=1
```

`coldcore.gpu.default_plugin_path()` looks for the plugin at
`<repo>/build/plugins/libcoldcore_covering.so`, relative to the repo
root (three levels up from `src/coldcore/gpu.py`). That resolves
correctly for a source checkout and for
`pip install -e .`; **for a non-editable install, set
`COLDCORE_PLUGIN_PATH` to the absolute path of the `.so`** (or pass
`--plugin /path/to/libcoldcore_covering.so`).

Other environment variables:

| variable | meaning |
|---|---|
| `COLDCORE_PLUGIN_PATH` | absolute path of the plugin `.so` to load |
| `COLDCORE_HBM_BUDGET` | bytes of device memory the placement policy may use (default 80e9) |
| `COLDCORE_CUDA_ARCHS` | cmake: CUDA architectures to compile for (default 90) |
| `COLDCORE_FORCE_GPU_TESTS` | run `@pytest.mark.gpu` tests without a `/dev/nvidia*` node |

Hardware notes and the memory-placement study are in
[benchmarks.md](benchmarks.md) and [performance.md](performance.md);
the C ABI the plugins speak is in
[PLUGIN.md](PLUGIN.md) and the walkthrough for writing one is
[plugin-authoring.md](plugin-authoring.md).

## 6. Run the tests

```bash
python -m pytest tests -k "not gpu"    # CPU suite, seconds
python -m pytest tests                 # adds GPU parity if a device is present
```

GPU tests skip themselves when there is no CUDA device or no built
plugin, so the second command is safe to run anywhere.

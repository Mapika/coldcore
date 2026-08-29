# Contributing

Thanks for your interest. coldcore is young; the highest-value
contributions right now are **new problem plugins** and **search-core
improvements benchmarked honestly**.

## Ground rules

- **Exactness is the product.** The framework's identity is exact
  whole-space gain/loss evaluation. Approximate or sampled fields belong
  behind an explicitly named capability, never silently.
- **Every plugin ships a reference.** A pure-numpy brute-force backend
  implementing `src/coldcore/protocol.py` on tiny instances, plus parity
  tests. No reference, no merge.
- **No records claimed in-code.** The search core reports solutions;
  verification gates (independent re-check of feasibility) decide what is
  a record. Keep that separation.
- **Benchmarks are paired.** A/B kernel or search changes are reported as
  paired runs on the same seeds and cells, wall-clock and quality, not
  best-of.

## Development setup

```bash
pip install -e ".[test,lint]"               # pure Python: search core, CLI, references
python -m pytest tests/ -m "not gpu"        # CPU-only suite
ruff check .                                # lint (config in pyproject.toml)

cmake -B build -DCMAKE_BUILD_TYPE=Release   # GPU plugins; needs CUDA toolkit >= 12
cmake --build build -j
python -m pytest tests/                     # includes GPU parity if a device is present
```

pip installs the pure-Python half only — the CUDA plugins are a
separate cmake step, and everything above the plugin seam (search core,
CLI, reference backends, examples) works without them. New to the
codebase: `docs/getting-started.md`, then `examples/03_custom_problem.py`,
then `docs/api.md`.

Python: 3.10+, numpy only (the ctypes binding style is deliberate — no
pybind11/nanobind dependency). Keep the package importable without a GPU
and without the compiled libraries.

## Writing a plugin

See `docs/plugin-authoring.md`. Summary: implement the C ABI
(`src/include/coldcore/plugin.h`), add a `src/plugins/<name>/CMakeLists.txt`, a
numpy reference backend under `src/coldcore/reference/`, and fixture tests
under `tests/`.

## Style

- C/CUDA: keep the upstream engine's style — flat C ABI, `rc != 0` error
  returns, no exceptions across the ABI, caller-allocated output buffers.
- Python: PEP 8-ish, no heavy frameworks, standard library + numpy.
- Commit messages: imperative summary line; reference the cell/bench in
  the body when a change is performance-motivated.

## Tests

`pytest`. Mark GPU-requiring tests with `@pytest.mark.gpu`. CI runs the
CPU suite on Python 3.10-3.12, a `ruff` lint job, and a compile-only
CUDA build; GPU tests run on maintainers' hardware before release tags.

Lint is deliberately permissive — pyflakes, pycodestyle errors and
obvious modernisations, no autoformatter, no style bikeshedding. The
rule set lives in `[tool.ruff]` in `pyproject.toml`; if a rule fights
the codebase's conventions, turn the rule off there rather than
reformatting working code.

New examples go in `examples/` as single runnable `.py` files that end
by asserting their expected result, and get added to the list in
`tests/test_examples.py` (a test fails if you forget).

"""Smoke-run the worked examples in examples/ on the reference backend.

Each example ends by asserting its own expected result (a known optimum,
independently verified inside the script), so running it to a zero exit
status is the whole test. No GPU involved.

They run as subprocesses rather than being imported so that a broken
example fails as a *user* would see it — `python examples/xx.py` — and
so that one example's global state cannot leak into the next.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")

SCRIPTS = ["01_covering_code.py", "02_domination.py", "03_custom_problem.py",
           "04_symmetry.py"]


def test_every_example_is_covered():
    """Keep this test honest when someone adds an example."""
    on_disk = sorted(f for f in os.listdir(EXAMPLES) if f.endswith(".py"))
    assert on_disk == sorted(SCRIPTS), (
        f"examples/ has {on_disk}; update SCRIPTS in tests/test_examples.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script):
    env = dict(os.environ, PYTHONPATH=REPO)      # importable without install
    r = subprocess.run([sys.executable, os.path.join(EXAMPLES, script)],
                       capture_output=True, text=True, timeout=600,
                       cwd=REPO, env=env)
    assert r.returncode == 0, (
        f"{script} failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout[-4000:]}\n--- stderr ---\n{r.stderr[-4000:]}")
    assert "OK" in r.stdout, f"{script} did not report success:\n{r.stdout[-2000:]}"

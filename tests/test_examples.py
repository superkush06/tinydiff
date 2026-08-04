"""The examples are part of the public surface: if they break, the README lies.

Two of these are dataset tests rather than library tests, which is deliberate.
`spiral()` is what the README's headline figure is drawn from, and a generator
that quietly degenerates into a cloud of noise would make the decision-boundary
picture a lie without failing a single gradient test.

`validate.py` and `transformer_block.py` are checked on their output rather
than only on their exit code: both are documents as much as they are scripts,
and both would still exit 0 while printing something the docs do not say.
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from examples.spiral import spiral

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(script, *args, env_extra=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(ROOT / "examples" / script),
                           *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)


def test_spiral_arms_stay_arms():
    """A point's nearest neighbour should almost always share its class.

    This is the property that makes the picture a two-spiral picture: the
    arms are separated by more than the jitter applied along them.
    """
    X, y = spiral(seed=0)
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    agrees = y[dist.argmin(axis=1)] == y
    assert agrees.mean() > 0.95


def test_spiral_defeats_every_straight_line():
    """The best least-squares linear classifier is no better than a coin."""
    X, y = spiral(seed=0)
    design = np.c_[X, np.ones(len(X))]
    w, *_ = np.linalg.lstsq(design, np.where(y == 1, 1.0, -1.0), rcond=None)
    acc = float(((design @ w > 0).astype(np.int64) == y).mean())
    assert acc < 0.6


def test_spiral_seeds_give_independent_draws():
    X0, y0 = spiral(seed=0)
    X1, y1 = spiral(seed=1)
    assert not np.allclose(np.sort(X0, axis=0), np.sort(X1, axis=0))
    assert y0.sum() == y1.sum() == len(y0) // 2


@pytest.mark.parametrize("script,args", [
    ("xor.py", ()),
    ("fit_sine.py", ("--epochs", "20")),
    ("spiral.py", ("--epochs", "20")),
])
def test_example_runs(script, args):
    proc = _run(script, *args)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "epoch" in proc.stdout


def test_validate_script_exits_clean():
    """docs/validation.md is pasted from this; a non-zero exit means it lies."""
    proc = _run("validate.py")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "all sections agree with their references" in proc.stdout
    assert "FAIL" not in proc.stdout


def test_transformer_block_matches_the_hand_written_backward():
    """The integration example must actually agree, not merely run."""
    proc = _run("transformer_block.py")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "the two derivations are the same function" in proc.stdout
    assert "<-- wrong" in proc.stdout  # the injected bugs are still detected


def test_figures_regenerate(tmp_path):
    pytest.importorskip("matplotlib")
    proc = _run("make_figures.py", "--quick", "--out", str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in ("engine_report.png", "validation.png", "spiral_training.png"):
        assert (tmp_path / name).stat().st_size > 10_000

"""An ML pipeline where the data gets a veto.

Ordinary software CI asks one question: does the code work? That is not
enough for a model, because the code can be perfect and the data can have
moved underneath it. So this pipeline has a stage that ordinary CI does
not: a drift gate sitting between training and packaging. If today's
applicants no longer resemble the ones the model learned from, the run
stops and nothing is packaged.

Every stage does real work and is timed. There is deliberately no stage
that pretends to deploy: packaging writes a genuine model artifact and
Dockerfile you can inspect, and actually shipping it is one documented
gcloud command that lives outside this script, because a demo that prints
"deployed" without deploying teaches the wrong thing.

Run it clean, then run it with --force-drift and watch where it stops.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import drift
import retraining

ARTIFACTS = Path(__file__).parent / "artifacts"

# The model must beat this on a held-out cohort to be worth packaging.
ACCURACY_FLOOR = 0.75


@dataclass
class Result:
    passed: bool
    detail: str


def stage(name: str, fn: Callable[[], Result]) -> tuple[bool, float]:
    print(f"  {name:<34}", end="", flush=True)
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        result = Result(False, f"{exc.__class__.__name__}: {exc}")
    elapsed = time.perf_counter() - started
    print(f"{'PASS' if result.passed else 'FAIL':<6}{elapsed:>6.2f}s   {result.detail}")
    return result.passed, elapsed


# --- stages ----------------------------------------------------------------


def s_repository() -> Result:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )
    if proc.returncode != 0:
        return Result(True, "not a git checkout, no commit to pin")
    return Result(True, f"commit {proc.stdout.strip()}")


def s_dependencies() -> Result:
    import sklearn

    return Result(True, f"numpy {np.__version__}, scikit-learn {sklearn.__version__}")


def s_tests() -> Result:
    """Assertions over the functions this pipeline is about to rely on."""
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)

    identical = drift.population_stability_index(sample, sample)
    assert identical < 1e-9, f"PSI of a sample against itself was {identical}"

    shifted = drift.population_stability_index(sample, sample + 3)
    assert shifted > drift.PSI_DRIFTED, "a three-sigma shift should register as drift"

    ks = drift.ks_statistic(sample, sample)
    assert ks == 0.0, f"KS of a sample against itself was {ks}"

    cohort = retraining.make_cohort(0, np.random.default_rng(1), rows=200)
    assert cohort.X.shape == (200, 4), f"unexpected cohort shape {cohort.X.shape}"
    assert set(np.unique(cohort.y)) <= {0, 1}, "labels must be binary"

    return Result(True, "5 assertions passed")


def s_train() -> Result:
    rng = np.random.default_rng(7)
    cohort = retraining.make_cohort(0, rng)
    model = retraining.train(cohort)

    ARTIFACTS.mkdir(exist_ok=True)
    coefficients = np.concatenate([model.coef_.ravel(), model.intercept_])
    np.save(ARTIFACTS / "model.npy", coefficients)

    return Result(True, f"coefficients {model.coef_.ravel().round(2).tolist()}")


def s_evaluate() -> Result:
    """Score the trained model on a cohort it has not seen."""
    rng = np.random.default_rng(99)
    holdout = retraining.make_cohort(0, rng)
    model = retraining.train(retraining.make_cohort(0, np.random.default_rng(7)))

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(
        holdout.y, model.predict(retraining._standardise(holdout.X))
    )
    return Result(
        accuracy >= ACCURACY_FLOOR,
        f"held-out accuracy {accuracy:.4f} (floor {ACCURACY_FLOOR})",
    )


def s_drift_gate(force_drift: bool) -> Result:
    """The stage ordinary CI does not have.

    Clean runs compare two cohorts drawn the same way, so any PSI is pure
    sampling noise. With --force-drift the second cohort is drawn eight
    weeks later, by which point the population has genuinely moved.
    """
    rng = np.random.default_rng(3)
    reference = retraining.make_cohort(0, rng)
    current = retraining.make_cohort(8 if force_drift else 0, rng)

    worst_column, worst_psi = "", 0.0
    for index, name in enumerate(retraining.FEATURES):
        psi = drift.population_stability_index(reference.X[:, index], current.X[:, index])
        if psi > worst_psi:
            worst_column, worst_psi = name, psi

    detail = f"worst {worst_column} PSI {worst_psi:.4f} (threshold {drift.PSI_DRIFTED})"
    return Result(worst_psi < drift.PSI_DRIFTED, detail)


def s_package() -> Result:
    """Write the two things that have to ship together and match."""
    ARTIFACTS.mkdir(exist_ok=True)
    model_path = ARTIFACTS / "model.npy"
    if not model_path.exists():
        return Result(False, "no model artifact to package")

    (ARTIFACTS / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "RUN pip install --no-cache-dir numpy scikit-learn\n"
        "COPY model.npy /app/model.npy\n"
        'CMD ["python", "-c", "import numpy; print(numpy.load(\'model.npy\'))"]\n',
        encoding="utf-8",
    )
    manifest = {
        "model": "loan-repayment",
        "artifact": "model.npy",
        "artifact_bytes": model_path.stat().st_size,
        "features": list(retraining.FEATURES),
    }
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return Result(True, f"model.npy + Dockerfile + manifest.json in {ARTIFACTS.name}/")


def s_smoke() -> Result:
    """Load the packaged artifact from disk and make one prediction with it."""
    coefficients = np.load(ARTIFACTS / "model.npy")
    weights, intercept = coefficients[:-1], coefficients[-1]

    applicant = np.array([41.0, np.log(52_000.0), 30.0, 2.0])
    probability = 1 / (1 + np.exp(-(applicant @ weights + intercept)))
    return Result(
        0.0 <= probability <= 1.0, f"repayment probability {probability:.4f}"
    )


def run(force_drift: bool) -> int:
    print("\nML pipeline, real stages and real timings")
    print("=" * 78)

    stages: list[tuple[str, Callable[[], Result]]] = [
        ("Repository state", s_repository),
        ("Dependencies", s_dependencies),
        ("Unit tests", s_tests),
        ("Train model", s_train),
        ("Evaluate on held-out cohort", s_evaluate),
        ("Drift gate", lambda: s_drift_gate(force_drift)),
        ("Package artifact", s_package),
        ("Smoke test packaged model", s_smoke),
    ]

    total = 0.0
    for name, fn in stages:
        passed, elapsed = stage(name, fn)
        total += elapsed
        if not passed:
            print("=" * 78)
            print(f"STOPPED at '{name}' after {total:.2f}s. Nothing was packaged.")
            return 1

    print("=" * 78)
    print(f"All stages passed in {total:.2f}s.")
    print("To ship it: gcloud run deploy loan-model --source ./artifacts")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--force-drift",
        action="store_true",
        help="draw the current cohort eight weeks later, so the gate trips",
    )
    args = parser.parse_args(argv)
    return run(args.force_drift)


if __name__ == "__main__":
    sys.exit(main())

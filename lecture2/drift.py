"""Detect when today's data stopped looking like the data you trained on.

The model hasn't changed. The world has. Last year's loan applicants were
in their forties; this year a campaign brought in twenty-somethings. Same
form, same questions, different people answering.

Two ways to measure it, both implemented here from scratch so you can see
what they actually do:

PSI (Population Stability Index) chops the reference range into bins,
counts how many rows land in each bin then and now, and sums the weighted
log difference. It is the industry default for numeric columns because the
thresholds are widely agreed: under 0.1 stable, 0.1 to 0.25 worth watching,
over 0.25 drifted.

KS (Kolmogorov-Smirnov) skips the bins entirely and measures the widest
gap between the two cumulative distributions. Because it never bins, it
notices movement at the extremes that PSI's bins average away.

Exits 1 when any column crosses the threshold, which is the signal a CI
drift gate reads.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PSI_STABLE = 0.10
PSI_DRIFTED = 0.25

# Bins with zero rows make the log blow up. This is the standard fudge.
EPSILON = 1e-6


def load_numeric_columns(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV and keep the columns that are entirely numeric."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")

    columns: dict[str, np.ndarray] = {}
    for name in rows[0]:
        try:
            columns[name] = np.array([float(r[name]) for r in rows])
        except (ValueError, TypeError):
            continue  # not numeric, skip it
    return columns


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two samples of the same column.

    Bin edges come from the reference, not from the combined data. The
    reference is what the model learned; the question is how far today has
    moved away from it, so today does not get a vote on where the bins go.
    """
    if reference.min() == reference.max():
        return 0.0

    edges = np.linspace(reference.min(), reference.max(), bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf  # today may fall outside yesterday's range

    ref_share = np.histogram(reference, bins=edges)[0] / len(reference) + EPSILON
    cur_share = np.histogram(current, bins=edges)[0] / len(current) + EPSILON

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Widest vertical gap between the two cumulative distributions, 0 to 1.

    Implemented directly rather than pulled from scipy: it is six lines,
    and seeing them makes the metric less mysterious than a function call.
    """
    grid = np.sort(np.concatenate([reference, current]))
    ref_cdf = np.searchsorted(np.sort(reference), grid, side="right") / len(reference)
    cur_cdf = np.searchsorted(np.sort(current), grid, side="right") / len(current)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def verdict(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_DRIFTED:
        return "watch"
    return "DRIFTED"


def compare(reference_path: Path, current_path: Path) -> int:
    """Print a per-column report. Returns the process exit code."""
    reference = load_numeric_columns(reference_path)
    current = load_numeric_columns(current_path)

    shared = [c for c in reference if c in current]
    if not shared:
        print("No numeric columns in common between the two files.")
        return 2

    print(f"\n{reference_path.name} -> {current_path.name}\n")
    print(f"{'column':<24}{'PSI':>10}{'KS':>10}   verdict")
    print("-" * 60)

    drifted = []
    for column in shared:
        psi = population_stability_index(reference[column], current[column])
        ks = ks_statistic(reference[column], current[column])
        print(f"{column:<24}{psi:>10.4f}{ks:>10.4f}   {verdict(psi)}")
        if psi >= PSI_DRIFTED:
            drifted.append(column)

    print("-" * 60)
    print(f"PSI thresholds: under {PSI_STABLE} stable, over {PSI_DRIFTED} drifted")

    if drifted:
        print(f"\nDrift detected in {drifted}. Exiting 1 so a deploy gate can stop.")
        return 1
    print("\nNo column crossed the threshold.")
    return 0


def write_sample(directory: Path, rows: int = 2000, seed: int = 0) -> tuple[Path, Path]:
    """Two days of loan applications, where one column genuinely moves.

    The numbers are drawn from distributions rather than typed in, and the
    drift is created by shifting one distribution, not by editing values.
    Age is the column that moves: a marketing campaign brought in a much
    younger cohort. Everything else is drawn from the same distribution
    both days, so a good detector should leave them alone.
    """
    rng = np.random.default_rng(seed)

    def applications(age_mean: float, age_sd: float) -> dict[str, np.ndarray]:
        return {
            "age": rng.normal(age_mean, age_sd, rows).clip(18, 95),
            "income": rng.lognormal(10.8, 0.45, rows),
            "months_at_address": rng.gamma(2.0, 18.0, rows),
            "existing_credit_lines": rng.poisson(2.4, rows).astype(float),
        }

    yesterday = applications(age_mean=44, age_sd=11)
    today = applications(age_mean=26, age_sd=6)

    paths = []
    for name, data in (("yesterday.csv", yesterday), ("today.csv", today)):
        path = directory / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data.keys())
            writer.writerows(zip(*(np.round(v, 4) for v in data.values())))
        paths.append(path)

    print(f"Wrote {paths[0].name} and {paths[1].name}, {rows} rows each.")
    print("Age was drawn from a younger distribution today. The rest were not.")
    return paths[0], paths[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="write two days of example applications")
    sample.add_argument("--rows", type=int, default=2000)
    sample.add_argument("--seed", type=int, default=0)

    check = sub.add_parser("check", help="compare two CSVs, exit 1 on drift")
    check.add_argument("reference", type=Path)
    check.add_argument("current", type=Path)

    args = parser.parse_args(argv)

    if args.command == "sample":
        write_sample(Path(__file__).parent, rows=args.rows, seed=args.seed)
        return 0
    return compare(args.reference, args.current)


if __name__ == "__main__":
    sys.exit(main())

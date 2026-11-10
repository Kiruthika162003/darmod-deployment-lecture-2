"""Watch a real model degrade, and fire a retraining signal when it does.

Nothing here counts a number downwards. A logistic regression is trained
on week zero's loan applicants and then scored, week after week, against
cohorts that genuinely move away from it. The accuracy you see printed is
measured by that model on data it has not seen.

Two things drift at once, because that is what happens in practice:

  The population shifts. Applicants get younger every week, so the model
  is increasingly asked about people unlike its training set.

  The relationship shifts. Holding several credit lines used to signal an
  over-extended borrower. Cards became free and everybody holds a few, so
  the signal decays week by week and eventually means nothing.

When measured accuracy falls through the floor, a message is published to
Pub/Sub. The message is the product: this script never retrains anything.
A separate worker subscribes to the topic and starts the training run,
which is why the two can be deployed, scaled and fail independently.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

LOCAL_OUTBOX = Path(__file__).parent / "retrain_signals.jsonl"

FEATURES = ("age", "income", "months_at_address", "existing_credit_lines")


@dataclass
class Cohort:
    """One week of applicants and whether they went on to repay."""

    week: int
    X: np.ndarray
    y: np.ndarray
    features: tuple[str, ...] = field(default=FEATURES, repr=False)


def make_cohort(week: int, rng: np.random.Generator, rows: int = 4000) -> Cohort:
    """Draw one week of applicants, and decide who repaid.

    Week zero is the world the model will be trained on. Later weeks move
    away from it in the two ways described in the module docstring. The
    label is generated from the features through a logistic link, so the
    relationship is real and learnable rather than assigned by hand.
    """
    age = rng.normal(44 - 1.6 * week, 11, rows).clip(18, 95)
    income = rng.lognormal(10.8, 0.45, rows)
    months_at_address = rng.gamma(2.0, 18.0, rows)
    credit_lines = rng.poisson(2.4, rows).astype(float)

    # Standardised so the coefficients below are on a comparable scale.
    z_age = (age - 44) / 11
    z_income = (np.log(income) - 10.8) / 0.45
    z_months = (months_at_address - 36) / 36
    z_lines = (credit_lines - 2.4) / 1.55

    # The weight on credit lines decays towards zero and then past it: what
    # was once a warning sign gradually stops carrying information.
    lines_weight = -2.7 + 0.30 * week

    log_odds = (
        0.3
        + 1.6 * z_age
        + 2.1 * z_income
        + 1.0 * z_months
        + lines_weight * z_lines
    )
    repaid = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))

    X = np.column_stack([age, income, months_at_address, credit_lines])
    return Cohort(week=week, X=X, y=repaid)


def train(cohort: Cohort) -> LogisticRegression:
    """Fit the model we will then watch go stale."""
    model = LogisticRegression(max_iter=1000)
    # Standardising inside the pipeline would be tidier, but keeping it
    # explicit here means the coefficients stay readable if you print them.
    model.fit(_standardise(cohort.X), cohort.y)
    return model


def _standardise(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, 1] = np.log(X[:, 1])  # income is lognormal
    return X


def _publish(payload: dict, topic: str | None) -> str:
    """Publish to Pub/Sub if it is reachable, otherwise append locally.

    Falling back to a file is not a lesser path: it keeps the demo honest
    on a laptop with no cloud account, and the payload is byte-identical
    either way.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project and topic:
        try:
            from google.cloud import pubsub_v1

            publisher = pubsub_v1.PublisherClient()
            path = publisher.topic_path(project, topic)
            message_id = publisher.publish(
                path, json.dumps(payload).encode("utf-8")
            ).result(timeout=30)
            return f"pubsub {path} id={message_id}"
        except Exception as exc:  # topic missing, no creds, no network
            print(f"[Pub/Sub unavailable ({exc.__class__.__name__}), writing locally]")

    with LOCAL_OUTBOX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return f"file {LOCAL_OUTBOX.name}"


def monitor(weeks: int, floor: float, topic: str | None, seed: int) -> int:
    rng = np.random.default_rng(seed)

    baseline = make_cohort(0, rng)
    model = train(baseline)
    trained_on = accuracy_score(baseline.y, model.predict(_standardise(baseline.X)))

    print(f"\nTrained on week 0. Accuracy on its own cohort: {trained_on:.4f}")
    print(f"Retraining floor: {floor}\n")
    print(f"{'week':>6}{'accuracy':>11}   verdict")
    print("-" * 40)

    for week in range(1, weeks + 1):
        cohort = make_cohort(week, rng)
        accuracy = accuracy_score(cohort.y, model.predict(_standardise(cohort.X)))

        if accuracy >= floor + 0.05:
            verdict = "healthy"
        elif accuracy >= floor:
            verdict = "watch"
        else:
            verdict = "RETRAIN"
        print(f"{week:>6}{accuracy:>11.4f}   {verdict}")

        if accuracy < floor:
            payload = {
                "model_id": "loan-repayment",
                "trained_on_week": 0,
                "evaluated_week": week,
                "accuracy": round(float(accuracy), 4),
                "accuracy_when_trained": round(float(trained_on), 4),
                "floor": floor,
                "detected_at": dt.datetime.now(dt.UTC).isoformat(),
                "action": "start a training run on the most recent cohorts",
            }
            print(f"\npublished to {_publish(payload, topic)}")
            print(json.dumps(payload, indent=2))
            return 0

    print(f"\nAccuracy never fell below {floor} in {weeks} weeks. No signal sent.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weeks", type=int, default=30)
    parser.add_argument("--floor", type=float, default=0.80)
    parser.add_argument(
        "--topic",
        default=os.environ.get("RETRAIN_TOPIC"),
        help="Pub/Sub topic name. Falls back to a local file if unset or unreachable.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    return monitor(args.weeks, args.floor, args.topic, args.seed)


if __name__ == "__main__":
    sys.exit(main())

"""A feature store small enough to read in one sitting.

A feature store keeps the values a model needs, and answers two questions
that look similar but aren't. Serving asks "what does this user look like
right now?" and needs an answer in milliseconds. Training asks "what did
this user look like on March 14th?" and needs history.

The failure this prevents is training/serving skew: two pipelines that both
claim to compute `age`, written months apart by different people, quietly
disagreeing. The model is fine. It is being fed a number it never trained
on. Nobody notices until the accuracy report.

So one write fans out to both stores, and a consistency check reads both
back and fails loudly when they disagree.

Online store is Firestore when GOOGLE_CLOUD_PROJECT is set, otherwise a
local SQLite file so this runs on a laptop with no cloud account. Offline
store is an append-only JSONL log standing in for BigQuery.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

ONLINE_DB = Path(__file__).parent / "online_features.sqlite3"
OFFLINE_LOG = Path(__file__).parent / "offline_features.jsonl"

# Namespaced so running this against a project you already use for other
# things can't overwrite documents that matter.
COLLECTION = os.environ.get("FEATURE_COLLECTION", "feature_store_demo")

# Floats never compare exactly, everything else should.
FLOAT_TOLERANCE = 1e-9


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _firestore_collection():
    """Return a Firestore collection, or None if we should use SQLite.

    Absence of credentials is a normal case, not an error: the whole point
    is that this runs anywhere. We only surface the reason once, so a demo
    doesn't spew the same warning on every command.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return None
    try:
        from google.cloud import firestore
    except ImportError:
        print("[google-cloud-firestore not installed, using local SQLite]")
        return None
    try:
        return firestore.Client(project=project).collection(COLLECTION)
    except Exception as exc:  # credentials, network, disabled API
        print(f"[Firestore unavailable ({exc.__class__.__name__}), using local SQLite]")
        return None


# --- online store ----------------------------------------------------------


def _sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(ONLINE_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS features ("
        "  entity TEXT, feature TEXT, value_json TEXT, updated_at TEXT,"
        "  PRIMARY KEY (entity, feature))"
    )
    return conn


def online_write(entity: str, feature: str, value: Any) -> str:
    """Write one feature to the online store. Returns the backend used."""
    collection = _firestore_collection()
    if collection is not None:
        collection.document(entity).set(
            {feature: value, "_updated_at": _now()}, merge=True
        )
        return "firestore"
    with _sqlite() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO features VALUES (?, ?, ?, ?)",
            (entity, feature, json.dumps(value), _now()),
        )
    return "sqlite"


def online_read(entity: str) -> dict[str, Any]:
    """Every feature the serving path can see for this entity."""
    collection = _firestore_collection()
    if collection is not None:
        doc = collection.document(entity).get()
        raw = doc.to_dict() or {}
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    with _sqlite() as conn:
        rows = conn.execute(
            "SELECT feature, value_json FROM features WHERE entity = ?", (entity,)
        ).fetchall()
    return {feature: json.loads(blob) for feature, blob in rows}


# --- offline store ---------------------------------------------------------


def offline_append(entity: str, feature: str, value: Any) -> None:
    """Append one row. The offline store never overwrites; that's the point."""
    record = {"entity": entity, "feature": feature, "value": value, "ts": _now()}
    with OFFLINE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def offline_latest(entity: str) -> dict[str, Any]:
    """The most recent value of each feature, which is what training would use."""
    if not OFFLINE_LOG.exists():
        return {}
    latest: dict[str, Any] = {}
    with OFFLINE_LOG.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["entity"] == entity:
                latest[row["feature"]] = row["value"]
    return latest


def offline_history(entity: str, feature: str) -> list[tuple[str, Any]]:
    """Every value this feature has ever held, oldest first."""
    if not OFFLINE_LOG.exists():
        return []
    history = []
    with OFFLINE_LOG.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["entity"] == entity and row["feature"] == feature:
                history.append((row["ts"], row["value"]))
    return history


# --- the consistency check -------------------------------------------------


def _differs(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a is not b
        return abs(float(a) - float(b)) > FLOAT_TOLERANCE
    return a != b


def skew_report(entity: str) -> dict[str, list[str]]:
    """Compare both stores.

    Features present in only one store matter as much as features that
    disagree: a feature serving has but training never saw is a model
    reading something it wasn't trained on.
    """
    online = online_read(entity)
    offline = offline_latest(entity)
    return {
        "disagree": sorted(
            k for k in set(online) & set(offline) if _differs(online[k], offline[k])
        ),
        "serving_only": sorted(set(online) - set(offline)),
        "training_only": sorted(set(offline) - set(online)),
    }


# --- commands --------------------------------------------------------------


def _parse_value(raw: str) -> Any:
    """Numbers, booleans and JSON stay typed; anything else is a string.

    Feature stores hold more than floats. A country code is a string, a
    verified flag is a boolean, and the last three products viewed is a
    list. Forcing everything through float() is the kind of simplification
    that falls over the first time someone asks a question about it.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def cmd_write(args: argparse.Namespace) -> int:
    value = _parse_value(args.value)
    backend = online_write(args.entity, args.feature, value)
    offline_append(args.entity, args.feature, value)
    print(f"online  ({backend}): {args.entity}.{args.feature} = {value!r}")
    print(f"offline (jsonl)   : appended to {OFFLINE_LOG.name}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    print(f"serving sees: {online_read(args.entity)}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    history = offline_history(args.entity, args.feature)
    if not history:
        print(f"no history for {args.entity}.{args.feature}")
        return 0
    for ts, value in history:
        print(f"  {ts}  {value!r}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    report = skew_report(args.entity)
    online, offline = online_read(args.entity), offline_latest(args.entity)

    print(f"\nonline  (serving) : {online}")
    print(f"offline (training): {offline}\n")

    for feature in report["disagree"]:
        print(f"  SKEW  {feature}: serving={online[feature]!r} training={offline[feature]!r}")
    for feature in report["serving_only"]:
        print(f"  ONLY IN SERVING  {feature}: training has never seen this")
    for feature in report["training_only"]:
        print(f"  ONLY IN TRAINING {feature}: serving cannot read this")

    if any(report.values()):
        print("\nStores disagree. A deploy gate reading this exit code would stop here.")
        return 2
    print("Both stores agree on every feature.")
    return 0


def cmd_diverge(args: argparse.Namespace) -> int:
    """Write to the online store only, leaving training's history untouched.

    This is what a changed serving pipeline looks like from the outside:
    someone edits how `age` is computed, the next run writes a different
    number, and training's records still hold the old one.
    """
    value = _parse_value(args.value)
    backend = online_write(args.entity, args.feature, value)
    print(f"online  ({backend}): {args.entity}.{args.feature} = {value!r}")
    print("offline            : untouched, as if the serving pipeline alone changed")
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Forget everything about one entity, in both stores.

    Worth having because the online store outlives the process. Without
    this, a second run inherits whatever the first one left behind and the
    check reports a difference nobody introduced today.
    """
    collection = _firestore_collection()
    if collection is not None:
        collection.document(args.entity).delete()
    else:
        with _sqlite() as conn:
            conn.execute("DELETE FROM features WHERE entity = ?", (args.entity,))

    if OFFLINE_LOG.exists():
        kept = [
            line
            for line in OFFLINE_LOG.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["entity"] != args.entity
        ]
        OFFLINE_LOG.write_text(
            "".join(line + "\n" for line in kept), encoding="utf-8"
        )

    print(f"Cleared {args.entity} from both stores.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    write = sub.add_parser("write", help="write a feature to both stores")
    write.add_argument("entity")
    write.add_argument("feature")
    write.add_argument("value")
    write.set_defaults(func=cmd_write)

    read = sub.add_parser("read", help="what the serving path can see")
    read.add_argument("entity")
    read.set_defaults(func=cmd_read)

    history = sub.add_parser("history", help="every value a feature has held")
    history.add_argument("entity")
    history.add_argument("feature")
    history.set_defaults(func=cmd_history)

    check = sub.add_parser("check", help="compare both stores, exit 2 if they disagree")
    check.add_argument("entity")
    check.set_defaults(func=cmd_check)

    diverge = sub.add_parser("diverge", help="write online only, to demonstrate skew")
    diverge.add_argument("entity")
    diverge.add_argument("feature")
    diverge.add_argument("value")
    diverge.set_defaults(func=cmd_diverge)

    reset = sub.add_parser("reset", help="clear one entity from both stores")
    reset.add_argument("entity")
    reset.set_defaults(func=cmd_reset)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

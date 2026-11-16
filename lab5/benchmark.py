import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from functions import get_function
from optimizer import DEFAULTS
from run import positive_int, run_once, summarize


ROOT = Path(__file__).resolve().parent
MODES = ("hybrid", "sa", "tabu")


def fingerprint():
    return hashlib.sha256((ROOT / "optimizer.py").read_bytes()).hexdigest()


def metadata(runs, nfe):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                platform=platform.platform(), machine=platform.machine(), numpy=np.__version__,
                optimizer_sha256=fingerprint(), defaults=DEFAULTS, runs=runs, max_nfe=nfe,
                protocol="One warmup per configuration, sequential interleaved measured runs, external counter and timer")


def measure(tf, seed, nfe, params):
    row, result = run_once(tf, seed, nfe, **params)
    row["first_success_nfe"] = next((i + 1 for i, f in enumerate(result["history"]) if tf.error(f) <= tf.eps), None)
    row["stats"] = result["stats"]
    return row


def compare(tf, runs=30, nfe=10000, seed=1, configurations=None):
    configurations = configurations or {mode: dict(mode=mode) for mode in MODES}
    rows = {label: [] for label in configurations}
    order = list(configurations)
    rng = np.random.default_rng(20261116)
    for params in configurations.values():
        run_once(tf, seed, nfe, **params)
    for trial in range(seed, seed + runs):
        for label in rng.permutation(order):
            rows[label].append(measure(tf, trial, nfe, configurations[label]))
    return {label: dict(params=configurations[label], summary=summarize(values), rows=values)
            for label, values in rows.items()}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Воспроизводимое сравнение гибрида и компонентов")
    parser.add_argument("--func", default="levy13")
    parser.add_argument("--runs", type=positive_int, default=30)
    parser.add_argument("--nfe", type=positive_int, default=10000)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark.json")
    args = parser.parse_args()
    comparison = compare(get_function(args.func), args.runs, args.nfe)
    data = dict(metadata=metadata(args.runs, args.nfe), comparison={args.func: comparison})
    if args.study:
        configurations = {str(value): dict(local_step=value) for value in (0.000001, 0.0001, 0.0025, 0.025, 0.25)}
        data["step_sensitivity"] = compare(get_function(args.func), args.runs, args.nfe,
                                           configurations=configurations)
    for mode, entry in comparison.items():
        summary = entry["summary"]
        print(args.func, mode, f"SR={summary['success_rate']:.2%}",
              f"median={summary['median']:.6g}", f"time={summary['median_time_s']:.6f}")
    save(args.output, data)
    print("Результаты:", args.output)


if __name__ == "__main__":
    main()

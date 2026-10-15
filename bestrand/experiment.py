"""Offline evaluation, portable checkpoints, and equally weighted summaries."""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr

from .data import DATA_ROOT, ROOT, file_digest, load_benchmark, manifest, read_json
from .methods import estimate, random_select, select

FIELDS = ["benchmark", "method", "fraction", "budget", "seed", "mae_pp", "spearman",
          "pool_size", "probe_evaluations", "selection_sha256"]
METHODS = ("BestRand", "Random")


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError("cannot write an empty result table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def selection_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def metrics(truth, predicted):
    mae = float(np.abs(predicted - truth).mean() * 100.0)
    correlation = (float(spearmanr(truth, predicted).statistic)
                   if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(predicted) > 0 else float("nan"))
    return mae, correlation


def evaluate_cell(benchmark, fraction, budget, seeds, config):
    probes, targets = benchmark.split_scores()
    truth = targets.mean(axis=1)
    n = targets.shape[1]
    for seed in seeds:
        for method in METHODS:
            if method == "BestRand":
                indices = select(probes, budget, seed, pool_multiplier=config["pool_multiplier"],
                                 candidate_count=config["candidate_count"])
                predicted = estimate(targets[:, indices])
                pool_size = min(n, config["pool_multiplier"] * budget)
            else:
                indices = random_select(n, budget, seed)
                selected = targets[:, indices]
                # These reductions preserve the reference Random implementation,
                # including exact ties in binary score rankings.
                predicted = (np.count_nonzero(selected, axis=1) / budget
                             if benchmark.scores.dtype == np.uint8 else selected.mean(axis=1))
                pool_size = 0
            mae, correlation = metrics(truth, predicted)
            yield {"benchmark": benchmark.name, "method": method, "fraction": fraction,
                   "budget": budget, "seed": seed, "mae_pp": mae, "spearman": correlation,
                   "pool_size": pool_size, "probe_evaluations": probes.shape[0] * pool_size,
                   "selection_sha256": selection_digest(indices)}


def read_checkpoint(path, name, fraction, budget, config):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"invalid checkpoint columns: {path}")
        rows = list(reader)
    keys = set()
    for row in rows:
        for key in ("budget", "seed", "pool_size", "probe_evaluations"):
            row[key] = int(row[key])
        for key in ("fraction", "mae_pp", "spearman"):
            row[key] = float(row[key])
        key = (row["seed"], row["method"])
        if (row["benchmark"] != name or row["budget"] != budget or row["fraction"] != fraction
                or row["method"] not in METHODS or key in keys
                or not config["seed_start"] <= row["seed"] < config["seed_start"] + config["seed_count"]
                or not math.isfinite(row["mae_pp"]) or row["mae_pp"] < 0
                or not (math.isnan(row["spearman"]) or -1 <= row["spearman"] <= 1)
                or len(row["selection_sha256"]) != 64):
            raise ValueError(f"invalid checkpoint row: {path}")
        keys.add(key)
    seeds = sorted({row["seed"] for row in rows})
    if seeds != list(range(config["seed_start"], config["seed_start"] + len(seeds))):
        raise ValueError(f"checkpoint seeds must be a contiguous prefix: {path}")
    if len(rows) != 2 * len(seeds):
        raise ValueError(f"checkpoint has an incomplete method pair: {path}")
    return rows


def _mean(values):
    return float(np.mean(values))


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def summarize(rows, config):
    groups = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["fraction"], row["method"]), []).append(row)
    expected_seeds = set(range(config["seed_start"], config["seed_start"] + config["seed_count"]))
    detail = []
    for name in config["benchmarks"]:
        for fraction in config["fractions"]:
            random_rows = groups.get((name, fraction, "Random"), [])
            random_errors = [row["mae_pp"] for row in random_rows]
            if not random_errors:
                raise ValueError("missing Random cell")
            mean_threshold = _mean(random_errors)
            q70_threshold = float(np.quantile(random_errors, 0.7, method="linear"))
            for method in METHODS:
                cell = groups.get((name, fraction, method), [])
                if len(cell) != len(expected_seeds) or {r["seed"] for r in cell} != expected_seeds:
                    raise ValueError(f"incomplete result cell: {name}/{fraction}/{method}")
                errors = np.asarray([r["mae_pp"] for r in cell])
                correlations = [r["spearman"] for r in cell]
                detail.append({"benchmark": name, "method": method, "fraction": fraction,
                               "budget": cell[0]["budget"], "seed_count": len(cell),
                               "mae_mean_pp": _mean(errors), "mae_seed_std_pp": _std(errors),
                               "spearman_mean": _mean(correlations), "spearman_seed_std": _std(correlations),
                               "random_mean_threshold_pp": mean_threshold,
                               "random_q70_threshold_pp": q70_threshold,
                               "risk_above_random_mean": _mean(errors > mean_threshold),
                               "risk_above_random_q70": _mean(errors > q70_threshold)})
    aggregate = []
    for fraction in [*config["fractions"], "all"]:
        for method in METHODS:
            cells = [r for r in detail if r["method"] == method and (fraction == "all" or r["fraction"] == fraction)]
            means = [r["mae_mean_pp"] for r in cells]
            aggregate.append({"method": method, "fraction": fraction, "benchmark_count": len(config["benchmarks"]),
                              "cell_count": len(cells), "mae_mean_pp": _mean(means),
                              "mae_benchmark_std_pp": _std(means),
                              "spearman_mean": _mean([r["spearman_mean"] for r in cells]),
                              "risk_above_random_mean": _mean([r["risk_above_random_mean"] for r in cells]),
                              "risk_above_random_q70": _mean([r["risk_above_random_q70"] for r in cells])})
    return detail, aggregate


def render_summary(aggregate, config):
    def number(value, digits=3):
        return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"
    lines = ["# BestRand Results", "",
             f"Seeds {config['seed_start']}–"
             f"{config['seed_start'] + config['seed_count'] - 1}; five construction models and twenty test models.", "",
             "MAE is in percentage points (lower is better). Benchmarks have equal weight.", "",
             "| Method | Retained | MAE | Benchmark std. | Spearman | Above Random mean | Above Random q70 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        fraction = "Mean" if row["fraction"] == "all" else f"{100 * row['fraction']:g}%"
        lines.append(f"| {row['method']} | {fraction} | {number(row['mae_mean_pp'])} | "
                     f"{number(row['mae_benchmark_std_pp'])} | {number(row['spearman_mean'])} | "
                     f"{number(100 * row['risk_above_random_mean'], 1)}% | "
                     f"{number(100 * row['risk_above_random_q70'], 1)}% |")
    lines += ["", "High-error probabilities use Random's mean and linearly interpolated 70th-percentile MAE "
              "as thresholds within each benchmark and budget; comparisons use strict >.", "",
              "Std. is across benchmark means at each ratio, and across benchmark–budget means in the Mean row. "
              "Undefined Spearman entries and their affected averages are N/A.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run BestRand and Random on bundled BestRand-Data.")
    parser.add_argument("--preset", choices=["quick", "paper"], default="quick")
    parser.add_argument("--benchmarks", nargs="+", help="Benchmark IDs (default: all bundled benchmarks)")
    parser.add_argument("--seeds", type=int, help="Override the number of seeds")
    parser.add_argument("--seed-start", type=int, help="Override the first seed")
    parser.add_argument("--output", type=Path, help="Output directory (default: results/<preset>)")
    parser.add_argument("--resume", action="store_true", help="Continue an identical interrupted run")
    args = parser.parse_args(argv)
    config = read_json(ROOT / "configs" / f"{args.preset}.json")
    catalog = manifest()
    config["benchmarks"] = args.benchmarks or list(catalog["benchmarks"])
    if len(set(config["benchmarks"])) != len(config["benchmarks"]) or not set(config["benchmarks"]) <= set(catalog["benchmarks"]):
        parser.error("benchmark IDs must be distinct and present in BestRand-Data/manifest.json")
    if args.seeds is not None:
        config["seed_count"] = args.seeds
    if args.seed_start is not None:
        config["seed_start"] = args.seed_start
    if config["seed_count"] < 1 or config["seed_start"] < 0:
        parser.error("--seeds must be positive and --seed-start must be non-negative")
    config.update({"preset": args.preset, "methods": list(METHODS), "construction_model_count": 5,
                   "test_model_count": 20, "python": platform.python_version(),
                   "numpy": np.__version__, "scipy": scipy.__version__,
                   "data_manifest_sha256": file_digest(DATA_ROOT / "manifest.json"),
                   "code_sha256": {str(p.relative_to(ROOT)): file_digest(p)
                                   for p in sorted((ROOT / "bestrand").glob("*.py")) + [ROOT / "run.py"]}})
    output = (args.output or ROOT / "results" / args.preset).resolve()
    config_path = output / "run.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            parser.error(f"output directory is not empty: {output}; use --resume for an identical run")
        if not config_path.exists() or read_json(config_path) != config:
            parser.error("cannot resume: configuration, data, code, or environment differs")
    atomic_text(config_path, json.dumps(config, indent=2) + "\n")
    all_rows = []
    for name in config["benchmarks"]:
        benchmark = load_benchmark(name)
        for fraction, budget in zip(benchmark.split["fractions"], benchmark.split["budgets"]):
            checkpoint = output / "checkpoints" / f"{name}-k{budget}.csv"
            rows = read_checkpoint(checkpoint, name, fraction, budget, config) if args.resume else []
            start = config["seed_start"] + len(rows) // 2
            stop = config["seed_start"] + config["seed_count"]
            for row in evaluate_cell(benchmark, fraction, budget, range(start, stop), config):
                rows.append(row)
                if row["method"] == "Random" and (row["seed"] + 1 - config["seed_start"]) % 100 == 0:
                    write_csv(checkpoint, rows, FIELDS)
            write_csv(checkpoint, rows, FIELDS)
            all_rows.extend(rows)
            print(f"{name:14s} {fraction:5.0%} k={budget:<5d} {len(rows) // 2} seeds complete", flush=True)
        del benchmark
    detail, aggregate = summarize(all_rows, config)
    write_csv(output / "per_seed.csv", all_rows, FIELDS)
    write_csv(output / "per_benchmark.csv", detail)
    write_csv(output / "aggregate.csv", aggregate)
    atomic_text(output / "summary.md", render_summary(aggregate, config))
    print(f"Results: {output / 'summary.md'}", flush=True)
    return 0

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bestrand.data import ROOT, load_benchmark, load_items, manifest, read_json
from bestrand.experiment import (FIELDS, evaluate_cell, main, metrics, read_checkpoint,
                                summarize, write_csv)


class ExperimentTest(unittest.TestCase):
    def test_all_data_and_fixed_splits(self):
        catalog = manifest()["benchmarks"]
        self.assertEqual(len(catalog), 12)
        self.assertEqual(sum(s["item_count"] for s in catalog.values()), 49619)
        for name, spec in catalog.items():
            with self.subTest(benchmark=name):
                benchmark = load_benchmark(name)
                items = load_items(name)
                self.assertEqual(len(items), benchmark.scores.shape[1])
                self.assertEqual(benchmark.scores.dtype, np.float64 if name == "truthfulqa" else np.uint8)
                probes, targets = benchmark.split_scores()
                self.assertEqual(probes.shape, (5, spec["item_count"]))
                self.assertEqual(targets.shape, (20, spec["item_count"]))
                self.assertTrue(all(row["item_id"].startswith(name + ":") for row in items))
                self.assertEqual(len({r["source_item_id"] for r in items}), len(items))

    def test_quick_matches_reference_for_every_cell_and_seed(self):
        with (Path(__file__).parent / "fixtures/quick_expected.csv").open(newline="", encoding="utf-8") as f:
            expected = {(r["benchmark"], r["method"], int(r["budget"]), int(r["seed"])): r
                        for r in csv.DictReader(f)}
        self.assertEqual(len(expected), 1440)
        config = read_json(ROOT / "configs/quick.json")
        for name in manifest()["benchmarks"]:
            benchmark = load_benchmark(name)
            for fraction, budget in zip(benchmark.split["fractions"], benchmark.split["budgets"]):
                for row in evaluate_cell(benchmark, fraction, budget, range(10), config):
                    key = (name, row["method"], budget, row["seed"])
                    reference = expected.pop(key)
                    self.assertEqual(row["selection_sha256"], reference["selection_sha256"], key)
                    for metric in ("mae_pp", "spearman"):
                        target = float(reference[metric])
                        if np.isnan(target):
                            self.assertTrue(np.isnan(row[metric]), key)
                        else:
                            self.assertLessEqual(abs(row[metric] - target), 1e-12, (key, metric))
        self.assertFalse(expected)

    def test_paper_grid(self):
        quick = read_json(ROOT / "configs/quick.json")
        paper = read_json(ROOT / "configs/paper.json")
        self.assertEqual(paper["seed_count"], 10000)
        self.assertEqual(paper["seed_start"], 0)
        self.assertEqual({k: v for k, v in quick.items() if k != "seed_count"},
                         {k: v for k, v in paper.items() if k != "seed_count"})

    def test_no_target_score_leakage(self):
        benchmark = load_benchmark("gpqa")
        changed = copy.deepcopy(benchmark)
        lookup = {mid: i for i, mid in enumerate(changed.model_ids)}
        changed.scores[[lookup[mid] for mid in changed.split["test_model_ids"]]] = 0
        config = read_json(ROOT / "configs/quick.json")
        a = list(evaluate_cell(benchmark, 0.2, 109, [0], config))
        b = list(evaluate_cell(changed, 0.2, 109, [0], config))
        self.assertEqual([r["selection_sha256"] for r in a], [r["selection_sha256"] for r in b])
        self.assertTrue(np.isnan(b[0]["spearman"]))

    def test_macro_weighting_and_undefined_spearman(self):
        config = {"benchmarks": ["a", "b"], "fractions": [0.2], "seed_start": 0, "seed_count": 2}
        rows = []
        for name, offset in [("a", 0), ("b", 10)]:
            for method in ["BestRand", "Random"]:
                for seed in range(2):
                    rows.append({"benchmark": name, "method": method, "fraction": 0.2,
                                 "budget": 2, "seed": seed, "mae_pp": offset + seed,
                                 "spearman": float("nan") if name == "b" else 1.0})
        detail, aggregate = summarize(rows, config)
        self.assertEqual(aggregate[0]["mae_mean_pp"], 5.5)
        self.assertTrue(np.isnan(aggregate[0]["spearman_mean"]))
        self.assertEqual(detail[0]["risk_above_random_mean"], 0.5)
        self.assertEqual(detail[0]["risk_above_random_q70"], 0.5)
        with self.assertRaises(ValueError):
            summarize(rows[:-1], config)
        self.assertTrue(np.isnan(metrics(np.ones(3), np.arange(3))[1]))

    def test_resume_and_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["--benchmarks", "gpqa", "--seeds", "2", "--output", directory]
            self.assertEqual(main(args), 0)
            first = (Path(directory) / "per_seed.csv").read_bytes()
            self.assertEqual(main(args + ["--resume"]), 0)
            self.assertEqual(first, (Path(directory) / "per_seed.csv").read_bytes())
            # A valid partial checkpoint resumes only missing paired seeds.
            checkpoint = Path(directory) / "checkpoints/gpqa-k11.csv"
            with checkpoint.open(newline="", encoding="utf-8") as handle:
                prefix = list(csv.DictReader(handle))[:2]
            write_csv(checkpoint, prefix, FIELDS)
            main(args + ["--resume"])
            self.assertEqual(first, (Path(directory) / "per_seed.csv").read_bytes())
            with self.assertRaises(SystemExit):
                main(args)
            with self.assertRaises(SystemExit):
                main(["--benchmarks", "gpqa", "--seeds", "3", "--output", directory, "--resume"])

    def test_checkpoint_rejects_duplicate_and_partial_pairs(self):
        benchmark = load_benchmark("gpqa")
        config = read_json(ROOT / "configs/quick.json")
        rows = list(evaluate_cell(benchmark, 0.02, 11, [0], config))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.csv"
            for bad in [rows[:1], rows + rows[:1]]:
                write_csv(path, bad, FIELDS)
                with self.assertRaises(ValueError):
                    read_checkpoint(path, "gpqa", 0.02, 11, config)


if __name__ == "__main__":
    unittest.main()

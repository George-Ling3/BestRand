"""Read the bundled BestRand-Data; no pickle or external paths."""

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "BestRand-Data"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    payload = read_json(DATA_ROOT / "manifest.json")
    if payload.get("schema_version") != "bestrand-data-v1":
        raise ValueError("unsupported data schema")
    return payload


@dataclass
class Benchmark:
    name: str
    scores: np.ndarray
    model_ids: tuple
    split: dict

    def split_scores(self):
        lookup = {mid: i for i, mid in enumerate(self.model_ids)}
        probes = self.scores[[lookup[mid] for mid in self.split["construction_model_ids"]]]
        targets = self.scores[[lookup[mid] for mid in self.split["test_model_ids"]]]
        return probes.astype(np.float64), targets.astype(np.float64)


def load_benchmark(name):
    catalog = manifest()["benchmarks"]
    if name not in catalog:
        raise ValueError(f"unknown benchmark: {name}")
    spec = catalog[name]
    directory = DATA_ROOT / name
    for filename, expected in spec["files"].items():
        if file_digest(directory / filename) != expected["sha256"]:
            raise ValueError(f"data checksum mismatch: {name}/{filename}")
    with np.load(directory / "scores.npz", allow_pickle=False) as archive:
        scores = archive["scores"]
    models = read_json(directory / "models.json")
    model_ids = tuple(row["model_id"] for row in models)
    split = read_json(directory / "split.json")
    if scores.shape != (spec["model_count"], spec["item_count"]) or len(model_ids) != scores.shape[0]:
        raise ValueError(f"matrix shape mismatch: {name}")
    if len(set(model_ids)) != len(model_ids) or not np.isfinite(scores).all():
        raise ValueError(f"invalid model identities or scores: {name}")
    if spec["score_type"] == "binary":
        if scores.dtype != np.uint8 or not np.isin(scores, (0, 1)).all():
            raise ValueError(f"invalid binary scores: {name}")
    elif spec["score_type"] == "continuous":
        if scores.dtype != np.float64:
            raise ValueError(f"continuous scores must be float64: {name}")
    else:
        raise ValueError(f"unknown score type: {name}")
    train, test = split["construction_model_ids"], split["test_model_ids"]
    if len(train) != 5 or len(test) != 20 or len(set(train + test)) != 25 or not set(train + test) <= set(model_ids):
        raise ValueError(f"invalid fixed model split: {name}")
    fractions, budgets = split["fractions"], split["budgets"]
    expected = [min(scores.shape[1], max(1, int(scores.shape[1] * f + 0.5))) for f in fractions]
    if fractions != [0.02, 0.05, 0.1, 0.15, 0.2, 0.5] or budgets != expected:
        raise ValueError(f"invalid fixed budgets: {name}")
    return Benchmark(name, scores, model_ids, split)


def load_items(name):
    """Return item records in score-column order, including complete prompts."""
    catalog = manifest()["benchmarks"]
    if name not in catalog:
        raise ValueError(f"unknown benchmark: {name}")
    path = DATA_ROOT / name / "items.jsonl.gz"
    if file_digest(path) != catalog[name]["files"]["items.jsonl.gz"]["sha256"]:
        raise ValueError(f"item checksum mismatch: {name}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle]
    if len(items) != catalog[name]["item_count"] or len({row["item_id"] for row in items}) != len(items):
        raise ValueError(f"item count or identity mismatch: {name}")
    if any(not isinstance(row["prompt"], str) for row in items):
        raise ValueError(f"invalid prompt: {name}")
    return items

# BestRand-Data

BestRand-Data contains full evaluation prompts and per-item model scores for
text benchmarks, derived from [RouterEval](https://huggingface.co/datasets/linggm/RouterEval).
All available models in the corresponding source score collections are included.
The complete data is bundled with this repository and occupies about **86 MiB**.

## Benchmarks

| Benchmark ID | Items | Models | Scores |
|---|---:|---:|---|
| `arc-challenge` | 1,172 | 5,000 | Binary |
| `bbh` | 5,761 | 3,811 | Binary |
| `gpqa` | 546 | 3,811 | Binary |
| `gsm8k` | 1,319 | 5,000 | Binary |
| `truthfulqa` | 817 | 5,000 | Continuous |
| `hellaswag` | 10,042 | 5,000 | Binary |
| `ifeval` | 541 | 3,811 | Binary |
| `math` | 1,324 | 3,811 | Binary |
| `mmlu-pro` | 12,032 | 1,823 | Binary |
| `mmlu` | 14,042 | 5,000 | Binary |
| `musr` | 756 | 3,811 | Binary |
| `winogrande` | 1,267 | 5,000 | Binary |

The suite contains **49,619 items**. `math` consists of the **level-5 MATH test
problems** across seven subjects; `gpqa` uses the extended set. Prompts include
the original context, options, and few-shot examples where present.

## Data Format

Each benchmark has four files:

| File | Contents |
|---|---|
| `scores.npz` | Compressed array `scores`, shaped `[model, item]` |
| `models.json` | Ordered records with `model_id` and original `source_model_id` |
| `items.jsonl.gz` | Ordered records with `item_id`, `source_item_id`, `source_task`, `source_index`, `subset`, and `prompt` |
| `split.json` | Construction/test model IDs, retention fractions, and integer budgets |

**Row `i` corresponds to model record `i`; column `j` corresponds to item record `j`.**
Keep this order when loading the scores and metadata. Item IDs identify questions
and should not be interpreted as column indices.

Binary scores are stored as compressed `uint8`; TruthfulQA scores are stored as
compressed `float64`. Both are lossless. All matrices are complete and finite.

```python
from bestrand.data import load_benchmark, load_items

benchmark = load_benchmark("gpqa")
scores = benchmark.scores                # all models, [3811, 546]
model_ids = benchmark.model_ids          # matching row order
items = load_items("gpqa")               # matching column order
probe_scores, test_scores = benchmark.split_scores()
```

`manifest.json` records benchmark shapes, score types, and file checksums.

## Experiment Settings

Each benchmark includes a fixed split of **five construction models and twenty
held-out test models**, using split seed 2026. Construction models cover the score
range; test models are sampled from the remainder. Additional model rows are
available for further experiments.

Retention fractions are `[0.02, 0.05, 0.10, 0.15, 0.20, 0.50]`. Each corresponding
budget is `floor(fraction * item_count + 0.5)` and is recorded in `split.json`.

## Attribution and License

RouterEval provides the model identities, evaluation prompts, and score records.
Underlying benchmark content retains its applicable licenses. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for sources, citations, and
[license texts](licenses/).

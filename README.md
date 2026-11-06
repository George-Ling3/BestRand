# Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand

[![Paper](https://img.shields.io/badge/Paper-arXiv--preprint-b31b1b)](https://arxiv.org/pdf/2611.00883)
[![Data](https://img.shields.io/badge/Data-BestRand--Data-orange)](BestRand-Data/README.md)
![Python](https://img.shields.io/badge/Python-3.10-blue)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

This repository provides the implementation of **BestRand** and the accompanying
**BestRand-Data** for our paper *Random Is Almost Enough? Stable and Low-Cost
Benchmark Compression via BestRand*.

**TLDR:** Random benchmark subsets often estimate full-benchmark scores accurately but can occasionally produce large errors. 
BestRand uses a few probe models to pick a representative subset, improving stability at low construction cost.

## ✨ Highlights

- **Simple method:** five probe models, a random pool, and 128 candidate subsets.
- **Ready-to-use data:** full prompts and model–item score matrices for benchmark compression.
- **Lightweight experiments:** CPU only; NumPy and SciPy; no embeddings or model downloads.

## ⚙️ Environment Setup

Use Python 3.10 and install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

## 📦 BestRand-Data

The dataset is included in `BestRand-Data/` (about **86 MiB**). It contains full
evaluation prompts, model identities, per-item scores, and fixed model splits.
No separate data download is needed.

The benchmarks include ARC-Challenge, BBH, GPQA, GSM8K, TruthfulQA, HellaSwag,
IFEval, MATH (level 5), MMLU-Pro, MMLU, MuSR, and WinoGrande.

```text
BestRand-Data/
├── manifest.json
├── gpqa/
│   ├── scores.npz       # [model, item] score matrix
│   ├── models.json      # model identities, in matrix-row order
│   ├── items.jsonl.gz   # item identities and full prompts, in column order
│   └── split.json       # construction/test models and evaluation budgets
└── ...                 # one directory per benchmark
```

See [BestRand-Data/README.md](BestRand-Data/README.md) for benchmark sizes and the data format.

## 🚀 Quick Start

Run the following commands from the repository root:

```bash
# All bundled benchmarks, six budgets, and 10 seeds per method.
python run.py --preset quick

# Paper settings: 10,000 seeds per method and benchmark–budget pair.
python run.py --preset paper
```

The runner evaluates BestRand and Random using the same five construction models
and twenty held-out test models. Retention ratios are **2%, 5%, 10%, 15%, 20%, and 50%**.

Outputs are written to `results/quick/` or `results/paper/`:

| File | Contents |
|---|---|
| `summary.md` | Overall MAE, Spearman, and high-error probabilities |
| `per_benchmark.csv` | Results for each benchmark and budget |
| `per_seed.csv` | Results for each random seed |
| `aggregate.csv` | Machine-readable overall results |
| `run.json` | Parameters and environment needed to resume the run |

For a smaller experiment or an interrupted run:

```bash
python run.py --benchmarks gpqa truthfulqa --seeds 3 --output results/example
python run.py --preset paper --resume
```

Use the same arguments with `--resume` when continuing a run. Choose a new output
directory for a different configuration. Tests: `python -m unittest discover -s tests -v`.

## 🛠️ Using BestRand on Your Own Data

Provide a score matrix with shape `[probe model, item]`. BestRand returns the
indices of the selected items. Evaluate your target models on those items and
average their scores to estimate full-benchmark performance.

```python
from bestrand import select, estimate
from bestrand.data import load_benchmark, load_items

benchmark = load_benchmark("gpqa")
probe_scores, target_scores = benchmark.split_scores()

indices = select(probe_scores, k=109, seed=0)
estimated_scores = estimate(target_scores[:, indices])

items = load_items("gpqa")
selected_prompts = [items[i]["prompt"] for i in indices]
```

`select` accepts binary or continuous probe scores. Set `pool_multiplier` and
`candidate_count` to change the defaults of 4 and 128. `estimate` expects a
`[target model, selected item]` matrix whose columns follow the returned index order.

## 🧩 How BestRand Works

1. Sample a pool of `min(4k, N)` items and obtain the probe scores on that pool.
2. Draw 128 random subsets of size `k` from the pool.
3. Select the subset whose mean probe-score vector is closest to the pool mean in squared L2 distance.
4. Evaluate target models on the selected subset and average their scores.

All candidates reuse the same pool scores. Target-model scores are used only
after subset selection. The bundled experiments use precomputed score matrices.

## 🙏 Acknowledgements and License

We thank the original benchmark creators for their contributions to model evaluation.

The implementation is released under the [MIT License](LICENSE). Data licenses
and source attributions are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


## 📝 Citation

If you find BestRand useful in your research, please consider citing our paper:

```bibtex
@misc{ling2026randomenoughstablelowcost,
  title={Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand},
  author={George Ling and Shanshan Zhong and Lijin Yang and Hao Yang and Nanxiang Li and Zhongzhan Huang},
  year={2026},
  eprint={2611.00883},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2611.00883}
}
```

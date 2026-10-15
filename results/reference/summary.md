# BestRand Results

Seeds 0–9999; five construction models and twenty test models.

MAE is in percentage points (lower is better). Benchmarks have equal weight.

| Method | Retained | MAE | Benchmark std. | Spearman | Above Random mean | Above Random q70 |
|---|---:|---:|---:|---:|---:|---:|
| BestRand | 2% | 5.448 | 3.057 | 0.823 | 26.0% | 15.5% |
| Random | 2% | 6.177 | 3.260 | 0.820 | 41.4% | 30.0% |
| BestRand | 5% | 3.362 | 1.898 | 0.884 | 25.3% | 15.1% |
| Random | 5% | 3.833 | 2.034 | 0.881 | 41.2% | 30.0% |
| BestRand | 10% | 2.294 | 1.303 | 0.921 | 24.1% | 13.9% |
| Random | 10% | 2.631 | 1.390 | 0.921 | 41.3% | 30.0% |
| BestRand | 15% | 1.806 | 1.032 | 0.941 | 23.0% | 13.2% |
| Random | 15% | 2.087 | 1.103 | 0.939 | 41.4% | 30.0% |
| BestRand | 20% | 1.497 | 0.861 | 0.953 | 21.3% | 11.9% |
| Random | 20% | 1.757 | 0.926 | 0.952 | 41.1% | 30.0% |
| BestRand | 50% | 0.737 | 0.428 | 0.982 | 19.4% | 10.7% |
| Random | 50% | 0.878 | 0.463 | 0.981 | 41.3% | 30.0% |
| BestRand | Mean | 2.524 | 2.226 | 0.917 | 23.2% | 13.4% |
| Random | Mean | 2.894 | 2.437 | 0.916 | 41.3% | 30.0% |

High-error probabilities use Random's mean and linearly interpolated 70th-percentile MAE as thresholds within each benchmark and budget; comparisons use strict >.

Std. is across benchmark means at each ratio, and across benchmark–budget means in the Mean row. Undefined Spearman entries and their affected averages are N/A.

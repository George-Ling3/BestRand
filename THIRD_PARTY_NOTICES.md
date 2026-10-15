# Data attribution and licenses

The MIT license at the repository root covers the BestRand implementation and
documentation. It does not replace the licenses of third-party benchmark content.

**BestRand-Data is derived from [RouterEval](https://huggingface.co/datasets/linggm/RouterEval).**
RouterEval supplies the prompts, model identities, and model–item scores.
Its dataset card declares MIT; the original 2025 MilkThink Lab notice is
preserved in [BestRand-Data/licenses/RouterEval-MIT.txt](BestRand-Data/licenses/RouterEval-MIT.txt).
See the [official repository](https://github.com/MilkThink-Lab/RouterEval) and
[RouterEval paper](https://aclanthology.org/2025.findings-emnlp.208/).

Prompt text and numerical scores are unchanged. The dataset label and storage
format have been updated. Underlying benchmark content retains its applicable terms:

| Benchmark | Original source and license information |
|---|---|
| ARC-Challenge | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc): CC BY-SA 4.0 |
| BBH | [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard): MIT; preserved notice in `BestRand-Data/licenses/BBH-MIT.txt` |
| GPQA | [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa): CC BY 4.0 |
| GSM8K | [Grade School Math](https://github.com/openai/grade-school-math): MIT; preserved OpenAI notice |
| TruthfulQA | [TruthfulQA dataset card](https://huggingface.co/datasets/truthfulqa/truthful_qa): Apache 2.0 |
| HellaSwag | [HellaSwag dataset card](https://huggingface.co/datasets/Rowan/hellaswag): MIT stated in its licensing section |
| IFEval | [Google IFEval](https://huggingface.co/datasets/google/IFEval): Apache 2.0 |
| MATH | [MATH](https://github.com/hendrycks/math): MIT; preserved Dan Hendrycks notice |
| MMLU-Pro | [Dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): MIT; [repository](https://github.com/TIGER-AI-Lab/MMLU-Pro) declares Apache 2.0, also preserved |
| MMLU | [MMLU](https://github.com/hendrycks/test): MIT; preserved Dan Hendrycks notice |
| MuSR | [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR): CC BY 4.0 |
| WinoGrande | [WinoGrande](https://github.com/allenai/winogrande): Apache 2.0; preserved license |

CC BY and CC BY-SA content remains under those licenses, including attribution
and share-alike requirements where applicable. License texts and available
upstream copyright notices are bundled under `BestRand-Data/licenses/`. Source links
identify the original creators; retain these attributions when redistributing.

## RouterEval citation

```bibtex
@inproceedings{huang-etal-2025-routereval,
  title = {RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs},
  author = {Huang, Zhongzhan and Ling, Guoming and Lin, Yupei and Chen, Yandong and Zhong, Shanshan and Wu, Hefeng and Lin, Liang},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year = {2025},
  url = {https://aclanthology.org/2025.findings-emnlp.208/}
}
```


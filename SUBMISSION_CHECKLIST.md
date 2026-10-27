# Assignment requirements and evidence

| Requirement | Implementation or evidence |
|---|---|
| Public English–Hindi parallel-text dataset | IITB source link in README; pinned revision and split counts in `runs/laptop/results/dataset_manifest.json` |
| Describe, preprocess, and split data | `project/data.py`; README dataset section; official held-out splits; deduplication and seeded reservoir sampling |
| Train a basic encoder–decoder Transformer | `project/basic.py`; `project/experiment.py`; training history in basic result metadata |
| Fine-tune mT5 | LoRA fine-tuning in `project/experiment.py`; mT5 result metadata and loss curves |
| Apply LLaMA and Gemma prompting/fine-tuning | Zero-shot instruction prompting using each model's chat template |
| Same test set and comparable generation | Stable test fingerprint; greedy beam-one inference; batch one; common output-token cap |
| BLEU and chrF | SacreBLEU values and signatures in four result JSON files; independently recomputed by `scripts/verify_results.py` |
| Adaptation time, inference time, parameter count | `runs/laptop/results/comparison.csv`; wall-clock and memory-scope limitations documented |
| Compare architecture and compute requirements | `runs/laptop/results/ANALYSIS.md`; `REPORT.md`; plots |
| Inspect generated translations and responsible AI | `runs/laptop/results/QUALITATIVE_REVIEW.md`; aligned `samples.html`; predictions JSON; review CSV |
| Temperature analysis | Three temperatures × three seeds plus greedy; fixed test subset; means and standard deviations; saved outputs |
| README with execution, results and conclusion | `README.md` and its linked analysis/report |
| Python or notebook implementation | Python modules in `project/`, scripts, configurations, tests and pinned dependencies |
| GitHub repository and source link | https://github.com/electro-glitch/fgai-assignment3-english-hindi-mt |
| Word cover with identity, title, dataset, repository | Personalized file in the local `submission/` directory; identity details excluded from Git |
| PDF cover | Exported from Word and visually inspected; local `submission/` directory |
| LMS upload | The user will manually upload the final PDF, as requested |

Completion of experiments is established by `runs/laptop/results/verification.json` with status `passed`, not by the presence of this checklist. Qualitative observations are AI-assisted and are not represented as an independent professional translator's review.

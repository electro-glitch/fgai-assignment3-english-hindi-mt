# English to Hindi Machine Translation

ICT4403 Foundations of Generative AI, Assignment 3. This project compares a basic encoder–decoder Transformer, mT5, LLaMA and Gemma using the same English–Hindi test sentences. It measures translation quality and computational cost on a laptop with 16 GB RAM and an RTX 3050 with 4 GB VRAM.

## Current status

The project implements all four required methods. See **Latest measured results** below and the linked report for experiment completion and scores. LLaMA and Gemma require approved Hugging Face access and local login. Scores, generated samples and a winning model are reported only after actual execution. Tests use synthetic fixtures to validate software behavior; they are not assignment results.

## Dataset

[IIT Bombay English–Hindi parallel corpus on Hugging Face](https://huggingface.co/datasets/cfilt/iitb-english-hindi), maintained by CFILT, IIT Bombay. It contains roughly 1.66 million training pairs and official validation and test splits. Each row has a `translation` dictionary with `en` and `hi` text. Data spans multiple sources, including software text; a small subset need not represent all Hindi usage. Cite: Anoop Kunchukuttan, Pratik Mehta and Pushpak Bhattacharyya, *The IIT Bombay English-Hindi Parallel Corpus*, LREC 2018.

The default uses 20,000 training, 300 validation and 300 test pairs. Preparation streams each complete split and performs seeded reservoir sampling rather than selecting the first rows. It normalizes Unicode NFC and whitespace; drops empty, over-400-character, or script-mismatched pairs; deduplicates both source and target; and excludes all eligible held-out source/target sentences from training. Test takes precedence over validation when duplicates overlap. The tokenizer is trained on training text only. The dataset revision, counts, configuration hash and test fingerprint are saved in `runs/laptop/data/manifest.json`. Do not change prepared files between model runs.

Streaming avoids a full in-memory dataset but still reads the corpus over the network. Deduplication keeps hashes in RAM. Downloads and model caches can require several GB of disk space. Model and dataset licenses remain those of their original publishers; accepting gated-model terms is the user's responsibility.

## Models and memory choices

| Model | Method | Resource choices |
|---|---|---|
| Basic Transformer | From-scratch PyTorch encoder–decoder; sinusoidal positions; causal decoder mask | 192 hidden units, 3 encoder + 3 decoder layers, 4 heads, 8k byte-level BPE vocabulary, batch 16 |
| [mT5-small](https://huggingface.co/google/mt5-small) | Actual supervised LoRA fine-tuning of query/value projections | Rank 8, batch 1, accumulate 16, checkpoint activations, BF16 on supported CUDA / FP32 on CPU |
| [LLaMA 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Zero-shot translation prompting | BF16/FP16 inference, batch 1, one model per process |
| [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it) | Zero-shot translation prompting | BF16/FP16 inference, batch 1, one model per process |

The assignment allows prompting for decoder-only models. This avoids optimizer memory for the two 1B models. mT5 is multilingual pretrained but is not already an instruction translation model; short LoRA runs can still perform poorly. Gemma's small variant has limited Hindi capability; report its measured result honestly. Approximate names such as “1B” are not parameter measurements; the pipeline counts loaded parameters, including adapter parameters where applicable.

The default caps encoder input/training targets at 96 tokens and generated output at 96 tokens. Decoder prompts allow 512 input tokens to accommodate instructions. Source strings and references remain identical, but tokenization and truncation differ; those differences are recorded and must be discussed. The output cap is comparable in tokens, not characters. FP16 is intentionally avoided for mT5 training because it can overflow. No bitsandbytes or Linux-specific setup is required.

## Windows setup

Use Python 3.10 or 3.11 and an up-to-date NVIDIA driver. Run PowerShell in this folder:

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m project doctor
```

`setup.ps1` creates a virtual environment and installs PyTorch 2.6 CUDA 12.4 plus dependencies. `setup.ps1 -Cpu` installs CPU PyTorch instead. Do not install into the system Python. If PowerShell blocks local scripts, run the commands listed in the scripts manually or use your institution's permitted execution policy.

Request access on both model pages, accept the model terms, then authenticate **locally**:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

Do not paste a token into source files, chat, notebooks or GitHub. The doctor command checks access without printing tokens. Model files and dataset downloads are cached in `.cache/huggingface` inside this project. A missing/gated model is an error; the project never substitutes another architecture silently.

## Execution

Run the complete pipeline, sequentially:

```powershell
.\run.ps1
```

Or run individual stages:

```powershell
.\.venv\Scripts\python.exe -m project prepare
.\.venv\Scripts\python.exe -m project basic
.\.venv\Scripts\python.exe -m project mt5
.\.venv\Scripts\python.exe -m project llama
.\.venv\Scripts\python.exe -m project gemma
.\.venv\Scripts\python.exe -m project report
.\.venv\Scripts\python.exe -m pytest -q
```

Every stage accepts `--config configs/laptop.json --run runs/laptop`. Inference/training also accept `--device cpu` or `--device cuda`. Completed checkpoints are reused. mT5 saves optimizer and random-generator state at epoch boundaries and resumes from the last completed epoch; an interrupted partial epoch is repeated. The scratch Transformer restarts incomplete training. If configuration changes, use a **new run directory** to prevent mixed comparisons.

For this RTX 3050, mT5 can use `$env:FGAI_MT5_MICROBATCH='4'` before launch. Gradient accumulation automatically becomes four, preserving effective batch size sixteen. The actual microbatch and accumulation are recorded in model metadata. If memory is insufficient, omit this environment variable to use the conservative batch-one configuration. Batch grouping changes token weighting slightly because loss is averaged per microbatch.

For a short real-data integration run, use `configs/smoke.json` with `--run runs/smoke`. It checks execution, not translation quality. Full training can take hours; actual duration depends on the laptop's power and thermal settings. Keep it plugged in. If GPU memory runs out, close other GPU applications, reduce batch size/sequence lengths in a new configuration, or use CPU mode for the affected model. Do not compare CPU and GPU timings as if they were equivalent.

## Evaluation and generated outputs

- `results/comparison.csv` and `results/REPORT.md`: measured BLEU, chrF, timing, parameter counts, memory and comparison.
- `results/comparison.png`: quality and latency plots.
- `predictions/{model}.json`: all generated translations, sources, references and stable IDs.
- `results/samples.html`: readable aligned translations with Hindi support.
- `results/{model}_temperature.json`, `temperature_summary.json`, `temperature.png`: temperature trials and plots.
- `results/ethics_review.csv`: aligned outputs and a human-review worksheet. Existing human notes are preserved on regeneration.
- `checkpoints/*/history.json`: training/validation losses by epoch; model revisions and adapters alongside them.

SacreBLEU uses `intl` tokenization and case-sensitive corpus BLEU, with its signature recorded. chrF uses character order 6 and beta 2. Scores run from 0 to 100. All four models use the same test fingerprint. Primary inference is greedy, beam size one, batch size one. Timing includes tokenization and decoding, excludes loading and warmup, and synchronizes CUDA. Training timing includes validation/checkpoint writing but excludes initial loading. It is elapsed wall time and can include laptop sleep or idle intervals, so it is not pure GPU compute time. Prompting has N/A training time, not misleading zero-cost training. Peak CUDA allocation covers the current invocation only; a reused checkpoint has inference-only allocation. It is not total system GPU usage.

The predeclared primary ranking metric is chrF; BLEU breaks ties. The report identifies a winner only when all four results exist. A result is specific to this data subset and setup. Differences in model size, pretraining, tokenization, precision and adaptation confound architecture-only conclusions.

## Sampling-temperature analysis

Compare greedy decoding with temperatures 0.3, 0.7 and 1.0 on the same first 50 shuffled test examples. Each positive temperature uses seeds 42, 43 and 44; report mean and population standard deviation of BLEU/chrF. Other sampling filters are disabled. Examine whether higher temperature produces diverse wording, omissions or invented content in the saved outputs; do not assume the trend in advance. Test temperatures are a descriptive experiment, not a mechanism for selecting a winning checkpoint.

## Responsible AI and review

Read generated outputs, not just scores. For each model, inspect examples involving gender roles, caste/religion, cultural references, honorifics, numbers, negation, uncertainty and sensitive content. Use `results/ethics_review.csv` to record actual evidence, harm and suggested corrections. Number-change and empty-output flags only identify candidates for review: Hindi digits or valid paraphrases can trigger them. No model-specific ethical finding is asserted without examining its translations.

Potential harms include stereotypes introduced through gender agreement, invented information, dropped uncertainty, loss of cultural meaning, and stronger or weaker wording for sensitive content. Mitigations include broader data coverage, subgroup evaluation, terminology checks, human review for consequential content, clear limitations, privacy protection and correction channels. BLEU/chrF cannot establish fairness or factual safety.

## Submission checklist

1. Complete all four real-data runs. Open the generated report, plots and translations. Add evidence-based human ethical observations and a discussion of temperature results.
2. Update this README's status with the measured results and conclusion, linking the actual run report. Do not use smoke-run scores as final results.
3. Create a GitHub repository and upload Python source, configuration, README, generated results/plots and permissible sample outputs. Exclude virtual environments, tokens, model weights and caches. Dataset source is linked above.
4. Create `YourName_RegNum_FGAI_Assignment3.docx` with your name, registration number, assignment title, system title (**English to Hindi Translation with Transformer mT5 LLaMA and Gemma**), dataset name/link, and GitHub URL. `submission_cover_template.txt` supplies the fields; replace every placeholder.
5. Save the Word cover as PDF, check the links, and submit the PDF to LMS yourself.

No GitHub repository or LMS submission is created automatically. The cover cannot be finalized without your name, registration number and repository URL.

<!-- MEASURED_RESULTS -->

## Latest measured results

Status: INCOMPLETE; missing mt5, llama, gemma

Dataset revision: `321516f50bdcc1214fa75164c545478976ed84bd`. Test fingerprint: `277e8288e56ba73a6015ebb92ce3cea16b728f997622438cfa853db2ffff73b4`.

| Model | BLEU | chrF | Training seconds | Inference seconds | Parameters |
|---|---:|---:|---:|---:|---:|
| basic | 0.49 | 12.91 | 23263.0 | 128.53 | 6,195,776 |

No four-model winner can be identified until all required runs are complete.


[Full report and analysis](runs/laptop/results/REPORT.md)

# Comparison and conclusion

On the shared 300-sentence IITB English–Hindi test subset, **gemma** has the highest chrF (37.31) with BLEU 11.76. The chrF ranking is gemma, llama, basic, mt5. **basic** has the lowest measured latency (0.428 seconds per sentence).

These are corpus-level overlap scores, not accuracy percentages or guarantees of semantic correctness. The best model in this experiment may still produce unsafe or unusable translations; consult the qualitative review.

## Architecture and compute

The basic Transformer and mT5 are encoder–decoder systems: the encoder reads the source, and the decoder generates target tokens using causal self-attention and cross-attention. LLaMA and Gemma are decoder-only systems: source instructions and the source sentence occupy the same causal context as generated target tokens. They use zero-shot instruction prompting here. The shared source sentences and greedy decoding control part of the comparison, but architecture is confounded by pretraining, model size, tokenization, precision and adaptation method. The scratch implementation recomputes the decoder prefix each generation step, while the Hugging Face models use cached generation when supported; latency therefore also reflects implementation choices. Pretraining cost is excluded. Loss curves should be interpreted within each model because vocabularies and tokenization differ.

| Model | Parameters | Trainable during adaptation | Adaptation time (seconds) | Inference (seconds/sentence) | Output cap hits |
|---|---:|---:|---:|---:|---:|
| basic | 6,195,776 | 6,195,776 | 23263.0 | 0.428 | 90/300 |
| mt5 | 300,520,832 | 344,064 | 2013.0 | 1.359 | 159/300 |
| llama | 1,235,814,400 | 0 | N/A: prompting | 1.382 | 80/300 |
| gemma | 999,885,952 | 0 | N/A: prompting | 2.535 | 1/300 |

Gemma has fewer loaded parameters than LLaMA here, yet a higher chrF. Parameter count alone did not predict translation quality in this experiment.

The mT5 adapter scored below the scratch Transformer despite its larger pretrained base. Its validation loss decreased, but its free-running translations still showed repetition and omissions. This demonstrates a limitation of this short, restricted LoRA adaptation, not an inherent weakness of mT5 or encoder–decoder models.

Output-cap hits can indicate either a genuinely long translation or degenerate repetition. The qualitative examples show repeated fragments in several capped outputs, so a cap hit should not automatically be interpreted as ordinary length truncation.

Training times are recorded wall-clock durations, including validation and saving; the original Transformer run spanned a long desktop session. Possible sleep/idle time was not instrumented, so its wall time cannot establish a pure compute-speed advantage or disadvantage. Inference timing uses GPU synchronization and excludes model loading and warmup. All reported primary timings must be checked for matching devices. CUDA peaks cover the invocation in which they were measured; the reused Transformer checkpoint has inference-only memory measurement.

## Temperature observations

The primary benchmark uses greedy decoding. The auxiliary experiment uses the same 50 test examples for all models, with three independent seeds at each positive temperature. Numbers below are means across those seeds; greedy has one run. These test-set observations describe sensitivity and are not used to select trained checkpoints.

**basic:** T=0: chrF 13.66; T=0.3: chrF 14.14; T=0.7: chrF 14.20; T=1: chrF 13.40. Highest observed subset chrF occurs at T=0.7. The change from greedy to T=1 is -0.26 chrF points. This is descriptive and not evidence of statistical significance.

**mt5:** T=0: chrF 10.73; T=0.3: chrF 13.68; T=0.7: chrF 14.64; T=1: chrF 14.03. Highest observed subset chrF occurs at T=0.7. The change from greedy to T=1 is +3.31 chrF points. This is descriptive and not evidence of statistical significance.

**llama:** T=0: chrF 28.48; T=0.3: chrF 27.57; T=0.7: chrF 25.26; T=1: chrF 15.34. Highest observed subset chrF occurs at T=0. The change from greedy to T=1 is -13.14 chrF points. This is descriptive and not evidence of statistical significance.

**gemma:** T=0: chrF 38.28; T=0.3: chrF 36.96; T=0.7: chrF 36.44; T=1: chrF 34.46. Highest observed subset chrF occurs at T=0. The change from greedy to T=1 is -3.83 chrF points. This is descriptive and not evidence of statistical significance.

Higher temperature flattens token probabilities. It can change wording and increase variation, but it does not reliably improve translation fidelity. Check the saved translations for omitted facts, repetition and invented information rather than interpreting diversity as quality.

## Limitations and responsible use

Only 20,000 training pairs and 300 test pairs were used. The training and held-out domains differ, and the corpus contains imperfect references. A single reference penalizes valid paraphrases and may reward overlap with a flawed translation. The shared 96-token output limit can cut off translations, and tokens correspond to different amounts of text across tokenizers. No significance test, broad fairness audit or professional translator assessment was conducted.

See [qualitative review](QUALITATIVE_REVIEW.md) for actual output evidence and [complete measured report](REPORT.md) for scores, plots and metric signatures.

## Conclusion

**gemma is the best-performing model by the predeclared chrF criterion in this experiment.** That conclusion is limited to this subset, these model variants and their different adaptation methods. For consequential use, retain source text and uncertainty, validate names and numbers, obtain qualified human review, and provide a correction process.

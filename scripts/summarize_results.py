"""Produce measured comparisons without inventing scores or ethical findings."""
from pathlib import Path
import sys
import statistics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project.common import read_json


def summarize(run):
    run = Path(run)
    names = ('basic', 'mt5', 'llama', 'gemma')
    rows = {name: read_json(run / 'results' / f'{name}.json') for name in names}
    ranked = sorted(rows, key=lambda name: (rows[name]['chrf'], rows[name]['bleu']), reverse=True)
    winner = ranked[0]
    fastest = min(rows, key=lambda name: rows[name]['seconds_per_sentence'])
    lines = ['# Comparison and conclusion', '',
             f'On the shared {rows[winner]["n_test"]}-sentence IITB English–Hindi test subset, **{winner}** has the highest chrF '
             f'({rows[winner]["chrf"]:.2f}) with BLEU {rows[winner]["bleu"]:.2f}. '
             f'The chrF ranking is {", ".join(ranked)}. '
             f'**{fastest}** has the lowest measured latency ({rows[fastest]["seconds_per_sentence"]:.3f} seconds per sentence).', '',
             'These are corpus-level overlap scores, not accuracy percentages or guarantees of semantic correctness. '
             'The best model in this experiment may still produce unsafe or unusable translations; consult the qualitative review.', '',
             '## Architecture and compute', '',
             'The basic Transformer and mT5 are encoder–decoder systems: the encoder reads the source, and the decoder generates target tokens using causal self-attention and cross-attention. '
             'LLaMA and Gemma are decoder-only systems: source instructions and the source sentence occupy the same causal context as generated target tokens. '
             'They use zero-shot instruction prompting here. The shared source sentences and greedy decoding control part of the comparison, '
             'but architecture is confounded by pretraining, model size, tokenization, precision and adaptation method.', '',
             '| Model | Parameters | Trainable during adaptation | Adaptation time (seconds) | Inference (seconds/sentence) | Output cap hits |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        timing = 'N/A: prompting' if row['training_seconds'] is None else f'{row["training_seconds"]:.1f}'
        lines.append(f'| {name} | {row["parameters"]:,} | {row["trainable_parameters"]:,} | {timing} | {row["seconds_per_sentence"]:.3f} | {row["capped_outputs"]}/{row["n_test"]} |')
    lines += ['', 'Training times are recorded wall-clock durations, including validation and saving; the original Transformer run spanned a long desktop session. '
              'Possible sleep/idle time was not instrumented, so its wall time cannot establish a pure compute-speed advantage or disadvantage. '
              'Inference timing uses GPU synchronization and excludes model loading and warmup. All reported primary timings must be checked for matching devices. '
              'CUDA peaks cover the invocation in which they were measured; the reused Transformer checkpoint has inference-only memory measurement.', '',
              '## Temperature observations', '',
              'The primary benchmark uses greedy decoding. The auxiliary experiment uses the same 50 test examples for all models, '
              'with three independent seeds at each positive temperature. Numbers below are means across those seeds; greedy has one run. '
              'These test-set observations describe sensitivity and are not used to select trained checkpoints.']
    for name in names:
        trials = read_json(run / 'results' / f'{name}_temperature.json')
        grouped = {temp: [t for t in trials if t['temperature'] == temp] for temp in sorted({t['temperature'] for t in trials})}
        means = {temp: statistics.mean(t['chrf'] for t in group) for temp, group in grouped.items()}
        best = max(means, key=means.get)
        line = '; '.join(f'T={temp:g}: chrF {value:.2f}' for temp, value in means.items())
        lines += ['', f'**{name}:** {line}. Highest observed subset chrF occurs at T={best:g}. '
                  f'The change from greedy to T=1 is {means[1.0] - means[0.0]:+.2f} chrF points. '
                  'This is descriptive and not evidence of statistical significance.']
    lines += ['', 'Higher temperature flattens token probabilities. It can change wording and increase variation, but it does not reliably improve translation fidelity. '
              'Check the saved translations for omitted facts, repetition and invented information rather than interpreting diversity as quality.', '',
              '## Limitations and responsible use', '',
              'Only 20,000 training pairs and 300 test pairs were used. The training and held-out domains differ, and the corpus contains imperfect references. '
              'A single reference penalizes valid paraphrases and may reward overlap with a flawed translation. For example, test ID '
              '`338385764865c774` mentions cleaning a park in English, while its Hindi reference describes correcting the poor condition of departments; '
              'this weak alignment limits the interpretation of an automatic score for that sentence. '
              'The shared 96-token output limit can cut off translations, and tokens correspond to different amounts of text across tokenizers. '
              'No significance test, broad fairness audit or professional translator assessment was conducted.', '',
              'See [qualitative review](QUALITATIVE_REVIEW.md) for actual output evidence and [complete measured report](REPORT.md) for scores, plots and metric signatures.', '',
              f'## Conclusion\n\n**{winner} is the best-performing model by the predeclared chrF criterion in this experiment.** '
              'That conclusion is limited to this subset, these model variants and their different adaptation methods. '
              'For consequential use, retain source text and uncertainty, validate names and numbers, obtain qualified human review, and provide a correction process.']
    (run / 'results/ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(run / 'results/ANALYSIS.md')


if __name__ == '__main__':
    summarize(sys.argv[1] if len(sys.argv) > 1 else 'runs/laptop')

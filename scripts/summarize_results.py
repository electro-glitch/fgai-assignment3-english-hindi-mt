"""Produce measured comparisons without inventing scores or ethical findings."""
from pathlib import Path
import sys
import statistics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project.common import read_json


def summarize(run):
    run = Path(run)
    config = read_json(run / 'config.json')
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
             'but architecture is confounded by pretraining, model size, tokenization, precision and adaptation method. '
             'The scratch implementation recomputes the decoder prefix each generation step, while the Hugging Face models use cached generation when supported; '
             'latency therefore also reflects implementation choices. Pretraining cost is excluded. Loss curves should be interpreted within each model because vocabularies and tokenization differ.', '',
             '| Model | Parameters | Trainable during adaptation | Adaptation time (seconds) | Inference (seconds/sentence) | Output cap hits |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in rows.items():
        timing = 'N/A: prompting' if row['training_seconds'] is None else f'{row["training_seconds"]:.1f}'
        lines.append(f'| {name} | {row["parameters"]:,} | {row["trainable_parameters"]:,} | {timing} | {row["seconds_per_sentence"]:.3f} | {row["capped_outputs"]}/{row["n_test"]} |')
    if rows['gemma']['parameters'] < rows['llama']['parameters'] and rows['gemma']['chrf'] > rows['llama']['chrf']:
        lines += ['', 'Gemma has fewer loaded parameters than LLaMA here, yet a higher chrF. Parameter count alone did not predict translation quality in this experiment.']
    if rows['mt5']['chrf'] < rows['basic']['chrf']:
        lines += ['', 'The mT5 adapter scored below the scratch Transformer despite its larger pretrained base. '
                  'Its validation loss decreased, but its free-running translations still showed repetition and omissions. '
                  'This demonstrates a limitation of this short, restricted LoRA adaptation, not an inherent weakness of mT5 or encoder–decoder models.']
    lines += ['', 'Output-cap hits can indicate either a genuinely long translation or degenerate repetition. '
              'The qualitative examples show repeated fragments in several capped outputs, so a cap hit should not automatically be interpreted as ordinary length truncation.']
    lines += ['', 'Training times are recorded wall-clock durations, including validation and saving; the original Transformer run spanned a long desktop session. '
              'Possible sleep/idle time was not instrumented, so its wall time cannot establish a pure compute-speed advantage or disadvantage. '
              'Inference timing uses GPU synchronization and excludes model loading and warmup. All reported primary timings must be checked for matching devices. '
              'CUDA peaks cover the invocation in which they were measured; the reused Transformer checkpoint has inference-only memory measurement.', '',
              '## Temperature observations', '',
              f'The primary benchmark uses greedy decoding. The auxiliary experiment uses the same {min(config["temperature_samples"], config["test_size"])} test examples for all models, '
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
              f'Only {config["train_size"]:,} training pairs and {config["test_size"]} test pairs were used. The training and held-out domains differ, and the corpus contains imperfect references. '
              'A single reference penalizes valid paraphrases and may reward overlap with a flawed translation. '
              'The shared 96-token output limit can cut off translations, and tokens correspond to different amounts of text across tokenizers. '
              'No significance test, broad fairness audit or professional translator assessment was conducted.', '',
              'See [qualitative review](QUALITATIVE_REVIEW.md) for actual output evidence and [complete measured report](REPORT.md) for scores, plots and metric signatures.', '',
              f'## Conclusion\n\n**{winner} is the best-performing model by the predeclared chrF criterion in this experiment.** '
              'That conclusion is limited to this subset, these model variants and their different adaptation methods. '
              'For consequential use, retain source text and uncertainty, validate names and numbers, obtain qualified human review, and provide a correction process.']
    (run / 'results/ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    from project.common import ROOT
    if run.resolve() == (ROOT / 'runs/laptop').resolve():
        path = ROOT / 'README.md'
        marker = '<!-- ANALYSIS_SUMMARY -->'
        text = path.read_text(encoding='utf-8').split(marker)[0].rstrip()
        import re
        status = ('## Current status\n\n'
                  f'All four model runs and temperature experiments are complete on the shared {rows[winner]["n_test"]}-sentence test set. '
                  f'**{winner} ranks first by chrF ({rows[winner]["chrf"]:.2f}), with BLEU {rows[winner]["bleu"]:.2f}.** '
                  'See the measured results below, the [score verification](runs/laptop/results/verification.json), and the '
                  '[qualitative review](runs/laptop/results/QUALITATIVE_REVIEW.md). '
                  'The translations still contain consequential errors; the reported winner is specific to this experiment. '
                  'Unit-test fixtures are separate from these real-data results.\n')
        text = re.sub(r'## Current status\n\n.*?(?=\n## Dataset)', lambda _: status, text, flags=re.S)
        summary = [marker, '', '## Measured comparison and temperature findings', '',
                   f'The best model by chrF is **{winner}** ({rows[winner]["chrf"]:.2f}); the fastest measured inference is **{fastest}** '
                   f'({rows[fastest]["seconds_per_sentence"]:.3f} seconds per sentence). '
                   'Size, pretraining, tokenization, precision, generation caching and adaptation methods differ, so this does not isolate an architecture effect.', '',
                   '| Model | Greedy chrF | T=0.3 chrF | T=0.7 chrF | T=1 chrF |', '|---|---:|---:|---:|---:|']
        for name in names:
            trials = read_json(run / 'results' / f'{name}_temperature.json')
            values = [statistics.mean(t['chrf'] for t in trials if t['temperature'] == temp) for temp in (0, .3, .7, 1)]
            summary.append('| ' + name + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
        summary += ['', 'Temperature numbers use the fixed 50-sentence subset, not the full 300-sentence benchmark. Positive temperatures average three seeds. '
                    'These observations do not establish statistical significance or justify tuning on the test set.', '',
                    '[Detailed comparison and conclusion](runs/laptop/results/ANALYSIS.md) · '
                    '[Actual translation evidence and ethical analysis](runs/laptop/results/QUALITATIVE_REVIEW.md) · '
                    '[Independent score recomputation](runs/laptop/results/verification.json)', '',
                    '**Conclusion:** use the measured winner only as the strongest model within this limited experiment. '
                    'The observed hallucinations, omissions and cultural/identity errors require further validation and qualified human review for consequential use.']
        path.write_text(text + '\n\n' + '\n'.join(summary) + '\n', encoding='utf-8')
    print(run / 'results/ANALYSIS.md')


if __name__ == '__main__':
    summarize(sys.argv[1] if len(sys.argv) > 1 else 'runs/laptop')

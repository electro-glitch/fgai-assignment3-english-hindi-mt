import csv
import statistics
from pathlib import Path
from .common import digest, read_json, save_json

MODELS = ('basic', 'mt5', 'llama', 'gemma')


def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = Path(run)
    folder = run / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run / 'data/manifest.json')
    save_json(folder / 'dataset_manifest.json', manifest)
    rows, predictions, missing = [], {}, []
    for name in MODELS:
        path = folder / f'{name}.json'
        if not path.exists():
            missing.append(name)
            continue
        row = read_json(path)
        if row['test_hash'] != manifest['test_hash'] or row['config_hash'] != manifest['config_hash']:
            raise ValueError(f'{name} used a different test set or configuration.')
        samples = read_json(run / 'predictions' / f'{name}.json')
        source = [{k: s[k] for k in ('en', 'hi', 'id')} for s in samples]
        if digest(source) != manifest['test_hash']:
            raise ValueError(f'{name} prediction alignment is invalid.')
        rows.append(row)
        predictions[name] = samples
    columns = ['model', 'bleu', 'chrf', 'training_seconds', 'inference_seconds',
               'seconds_per_sentence', 'parameters', 'trainable_parameters',
               'peak_cuda_allocated_mb', 'truncated_inputs', 'capped_outputs', 'device', 'adaptation']
    with (folder / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Measured experiment report', '',
             'Status: ' + ('INCOMPLETE; missing ' + ', '.join(missing) if missing else 'All four primary model runs available.'),
             '', f'Dataset revision: `{manifest["revision"]}`. Test fingerprint: `{manifest["test_hash"]}`.', '',
             '| Model | BLEU | chrF | Training seconds | Inference seconds | Parameters |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        training = 'N/A (prompting)' if row['training_seconds'] is None else f'{row["training_seconds"]:.1f}'
        lines.append(f'| {row["model"]} | {row["bleu"]:.2f} | {row["chrf"]:.2f} | {training} | {row["inference_seconds"]:.2f} | {row["parameters"]:,} |')
    if len(rows) == 4:
        winner = max(rows, key=lambda x: (x['chrf'], x['bleu']))
        lines += ['', f'Best measured model by predeclared primary metric chrF (BLEU breaks ties): **{winner["model"]}**.',
                  'This ranking applies only to this subset and protocol; no significance test or universal superiority is claimed.']
    else:
        lines += ['', 'No four-model winner can be identified until all required runs are complete.']
    lines += ['', '## Interpretation', '',
              'Compare BLEU and chrF together. chrF is the primary ranking metric because character overlap is useful for Hindi morphology. '
              'The scratch Transformer learns translation and representations only from this training subset; mT5 adapts multilingual pretrained representations; '
              'LLaMA and Gemma use instruction pretraining and zero-shot prompting. Size, tokenizer, pretraining and adaptation differ, so this is not a controlled architecture ablation.', '',
              'Inference uses batch size one, greedy decoding, one beam and the same output token cap. Tokens differ by tokenizer; decoder prompts add input overhead. '
              'Timing excludes loading and warmup but includes tokenization, generation and decoding, with CUDA synchronization. '
              'Training includes validation and checkpoint saves, excludes initial model/tokenizer loading, and uses the best validation-loss checkpoint. '
              'Prompting has no training time (N/A). Peak CUDA allocation covers only the current invocation: it includes training only if training occurred in that invocation. A reused checkpoint reports inference allocation only. It excludes non-PyTorch/display allocations. '
              'Review truncation and output-cap counts before comparing quality. Different devices invalidate direct speed comparisons.']
    temp_summary = []
    lines += ['', '## Sampling-temperature analysis', '',
              'Same fixed test subset at each temperature. Positive temperatures use three seeds; report population standard deviation. '
              'Temperature zero is greedy and has one run. Sampling uses the full distribution (top-p=1, top-k=0 for HF models); '
              'temperature changes output diversity, not model weights. Do not select checkpoints or tune temperatures against this test set.', '',
              '| Model | Temperature | Runs | BLEU mean ± SD | chrF mean ± SD |', '|---|---:|---:|---:|---:|']
    for name in MODELS:
        path = folder / f'{name}_temperature.json'
        if not path.exists():
            lines += ['', f'Temperature experiment pending: {name}.']
            continue
        trials = read_json(path)
        for temp in sorted(set(t['temperature'] for t in trials)):
            group = [t for t in trials if t['temperature'] == temp]
            stats = {'model': name, 'temperature': temp, 'runs': len(group)}
            for metric in ('bleu', 'chrf'):
                stats[metric + '_mean'] = statistics.mean(t[metric] for t in group)
                stats[metric + '_sd'] = statistics.pstdev(t[metric] for t in group)
            temp_summary.append(stats)
            lines.append(f'| {name} | {temp:g} | {len(group)} | {stats["bleu_mean"]:.2f} ± {stats["bleu_sd"]:.2f} | {stats["chrf_mean"]:.2f} ± {stats["chrf_sd"]:.2f} |')
    save_json(folder / 'temperature_summary.json', temp_summary)
    if temp_summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in MODELS:
            points = [r for r in temp_summary if r['model'] == name]
            if points:
                ax.errorbar([p['temperature'] for p in points], [p['chrf_mean'] for p in points],
                            yerr=[p['chrf_sd'] for p in points], marker='o', capsize=3, label=name)
        ax.set(xlabel='Sampling temperature (0 = greedy)', ylabel='chrF', title='Temperature sensitivity on fixed subset')
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / 'temperature.png', dpi=160)
        plt.close(fig)
        lines += ['', '![Temperature experiment](temperature.png)']
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, field, title in zip(axes, ('bleu', 'chrf', 'seconds_per_sentence'), ('BLEU', 'chrF', 'Seconds / sentence')):
            ax.bar([r['model'] for r in rows], [r[field] for r in rows], color='#286a9c')
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(folder / 'comparison.png', dpi=160)
        plt.close(fig)
        lines += ['', '![Measured comparison](comparison.png)']
    histories = [r for r in rows if r.get('history')]
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4), squeeze=False)
        for ax, row in zip(axes[0], histories):
            for split in ('train', 'validation'):
                ax.plot([h['epoch'] for h in row['history']], [h[split] for h in row['history']], marker='o', label=split)
            ax.set(title=row['model'], xlabel='Epoch', ylabel='Token cross-entropy')
            ax.legend()
        fig.tight_layout()
        fig.savefig(folder / 'training.png', dpi=160)
        plt.close(fig)
        lines += ['', '![Training history](training.png)']
    lines += ['', '## Responsible AI review', '',
              'Use ethics_review.csv to inspect aligned outputs. Automated digit-change and empty-output flags are screening aids, '
              'not proof of hallucination or harm. A Hindi speaker must review gender/caste/religion stereotypes, omitted negation, invented facts, '
              'honorifics and cultural references, medical/legal uncertainty, and handling of sensitive language. '
              'Record actual source/translation evidence and a correction. Do not claim model-specific ethical findings from overlap scores alone.', '',
              'Mitigations: diverse and documented training data; subgroup evaluation with adequate sample sizes; preserve names, numbers and uncertainty; '
              'human review for consequential translations; display limitations; avoid private data in public outputs; provide correction and escalation channels. '
              'Automatic translation must not be the sole basis for medical, legal, or safety decisions.', '',
              '## Generated samples', '', 'See samples.html for side-by-side outputs and predictions/*.json for the full test translations.']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Keep the submission README tied to the main run, never to smoke-test scores.
    from .common import ROOT
    if run.resolve() == (ROOT / 'runs/laptop').resolve():
        readme = ROOT / 'README.md'
        content = readme.read_text(encoding='utf-8')
        marker = '<!-- MEASURED_RESULTS -->'
        if marker in content:
            content = content.split(marker)[0].rstrip()
        readme.write_text(content + '\n\n' + marker + '\n\n## Latest measured results\n\n' +
                          '\n'.join(lines[2:lines.index('## Interpretation')]) +
                          '\n\n[Full report and analysis](runs/laptop/results/REPORT.md)\n', encoding='utf-8')
    if predictions:
        import html
        import re
        records = []
        first = next(iter(predictions.values()))
        tables = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Translation comparison</title>',
                  '<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:12px;vertical-align:top}th{background:#e9f2f8}td{max-width:300px;overflow-wrap:anywhere}</style>',
                  '<h1>English–Hindi translation comparison</h1><p>Actual generated outputs. Human review required.</p><table><tr><th>Source</th><th>Reference</th>']
        tables += [f'<th>{name}</th>' for name in predictions]
        tables.append('</tr>')
        for i, row in enumerate(first):
            values = [row['en'], row['hi']] + [samples[i]['prediction'] for samples in predictions.values()]
            tables.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
            for name, samples in predictions.items():
                output = samples[i]['prediction']
                digits = lambda text: re.findall(r'\d+', text)
                flags = []
                if digits(row['en']) != digits(output):
                    flags.append('number change: inspect manually')
                if not output.strip():
                    flags.append('empty translation')
                records.append({'id': row['id'], 'model': name, 'source': row['en'], 'reference': row['hi'],
                                'prediction': output, 'screening_flags': '; '.join(flags),
                                'human_category': '', 'human_evidence': '', 'suggested_correction': ''})
        tables.append('</table></html>')
        (folder / 'samples.html').write_text('\n'.join(tables), encoding='utf-8')
        # Do not overwrite a human review on regeneration.
        review_path = folder / 'ethics_review.csv'
        if not review_path.exists():
            with review_path.open('w', newline='', encoding='utf-8-sig') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    print(f'Report saved to {folder / "REPORT.md"}')

"""Recompute every reported score and enforce the common-test protocol."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project.common import read_json, save_json, digest
from project.experiment import score


def verify(run):
    run = Path(run)
    config = read_json(run / 'config.json')
    manifest = read_json(run / 'data/manifest.json')
    reference = read_json(run / 'data/test.json')
    assert digest(reference) == manifest['test_hash']
    records = []
    for name in ('basic', 'mt5', 'llama', 'gemma'):
        result = read_json(run / 'results' / f'{name}.json')
        samples = read_json(run / 'predictions' / f'{name}.json')
        assert [{k: r[k] for k in ('en', 'hi', 'id')} for r in samples] == reference, name
        assert result['n_test'] == len(reference), name
        assert result['test_hash'] == manifest['test_hash'], name
        assert result['config_hash'] == digest(config), name
        assert result['parameters'] > 0 and result['inference_seconds'] > 0, name
        recalculated = score([s['prediction'] for s in samples], [r['hi'] for r in reference])
        for metric in ('bleu', 'chrf'):
            assert abs(result[metric] - recalculated[metric]) < 1e-8, (name, metric)
        trials = read_json(run / 'results' / f'{name}_temperature.json')
        expected = {(0.0, config['seed'])} | {(t, seed) for t in config['temperatures'] for seed in config['temperature_seeds']}
        assert len(trials) == len(expected), name
        assert {(t['temperature'], t['seed']) for t in trials} == expected, name
        subset = reference[:config['temperature_samples']]
        for trial in trials:
            assert trial['n'] == len(subset)
            assert [{k: s[k] for k in ('en', 'hi', 'id')} for s in trial['samples']] == subset
            check = score([s['prediction'] for s in trial['samples']], [r['hi'] for r in subset])
            for metric in ('bleu', 'chrf'):
                assert abs(trial[metric] - check[metric]) < 1e-8
        records.append({'model': name, 'test_samples': len(samples), 'temperature_trials': len(trials),
                        'scores_recomputed': True, 'prediction_file_hash': digest(samples)})
    save_json(run / 'results/verification.json', {'status': 'passed', 'test_hash': manifest['test_hash'], 'models': records})
    print('All four models and all temperature trials verified against the same test set.')


if __name__ == '__main__':
    verify(sys.argv[1] if len(sys.argv) > 1 else 'runs/laptop')

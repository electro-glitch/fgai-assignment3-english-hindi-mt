"""Repeat primary timing after a power-state change; preserve the old measurement."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from project.common import read_json, save_json, seed_all, sync, digest
from project.experiment import load_decoder, predict
from project.data import load_splits


def retime(name, run):
    if name not in ('llama', 'gemma'):
        raise ValueError('This utility is for decoder-only models.')
    run = Path(run)
    config = read_json(run / 'config.json')
    old = read_json(run / 'results' / f'{name}.json')
    samples = read_json(run / 'predictions' / f'{name}.json')
    rows = load_splits(run)['test']
    assert digest(rows) == old['test_hash']
    seed_all(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('This timing repeat requires the same CUDA device.')
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, _ = load_decoder(name, config, run, 'cuda')
    predict(name, model, tokenizer, rows[:1], config, 'cuda')
    sync('cuda')
    start = time.perf_counter()
    predictions, stats = predict(name, model, tokenizer, rows, config, 'cuda')
    sync('cuda')
    elapsed = time.perf_counter() - start
    if predictions != [row['prediction'] for row in samples]:
        raise RuntimeError('Predictions changed during timing repeat; old results retained. Investigate before updating.')
    old['timing_note'] = f'Initial {name} run began on battery; charger was connected during this run. Not used for final speed comparison.'
    save_json(run / 'results' / f'{name}_initial_timing.json', old)
    updated = {**old, **stats, 'inference_seconds': elapsed, 'seconds_per_sentence': elapsed / len(rows),
               'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20,
               'memory_measurement_scope': 'Inference-only timing repeat',
               'timing_note': 'Repeated primary inference after user confirmed charger connection; all predictions exactly matched the original run.'}
    save_json(run / 'results' / f'{name}.json', updated)
    print(f'{name}: repeated primary inference {elapsed:.3f} seconds; predictions unchanged.')


if __name__ == '__main__':
    retime(sys.argv[1] if len(sys.argv) > 1 else 'gemma', sys.argv[2] if len(sys.argv) > 2 else 'runs/laptop')

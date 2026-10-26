"""Streaming reservoir sampling; official held-out splits stay held out."""
import random
import re
import unicodedata
from pathlib import Path
from .common import digest, read_json, save_json

DATASET = 'cfilt/iitb-english-hindi'


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def clean(row, max_chars):
    pair = {lang: normalize(row['translation'][lang]) for lang in ('en', 'hi')}
    if any(not value or len(value) > max_chars for value in pair.values()):
        return None
    if not re.search('[A-Za-z]', pair['en']) or not re.search('[\u0900-\u097F]', pair['hi']):
        return None
    if pair['en'] == pair['hi']:
        return None
    return pair


def reservoir(rows, count, seed, max_chars, banned_en=None, banned_hi=None):
    rng = random.Random(seed)
    selected, seen_en, seen_hi = [], set(), set()
    banned_en, banned_hi = banned_en or set(), banned_hi or set()
    eligible = scanned = 0
    for row in rows:
        scanned += 1
        pair = clean(row, max_chars)
        if pair is None:
            continue
        # Store hashes instead of all corpus strings: bounded memory overhead per row.
        en, hi = digest(pair['en']), digest(pair['hi'])
        if en in banned_en or hi in banned_hi or en in seen_en or hi in seen_hi:
            continue
        seen_en.add(en)
        seen_hi.add(hi)
        eligible += 1
        if len(selected) < count:
            selected.append(pair)
        else:
            index = rng.randrange(eligible)
            if index < count:
                selected[index] = pair
    if len(selected) < count:
        raise ValueError(f'Requested {count} unique pairs, found only {len(selected)}.')
    rng.shuffle(selected)
    return selected, seen_en, seen_hi, {'scanned': scanned, 'eligible': eligible, 'selected': count}


def prepare(config, run):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    run = Path(run)
    if (run / 'data/manifest.json').exists():
        manifest = read_json(run / 'data/manifest.json')
        if manifest['config_hash'] != digest(config):
            raise ValueError('Configuration changed; select a new --run directory.')
        print('Existing prepared dataset retained.')
        return
    published_manifest = run / 'results/dataset_manifest.json'
    revision = read_json(published_manifest)['revision'] if published_manifest.exists() else HfApi().dataset_info(DATASET).sha
    blocked_en, blocked_hi, stats, splits = set(), set(), {}, {}
    # Exclude ALL valid held-out source/target sentences from training, not just sampled ones.
    for i, split in enumerate(('test', 'validation', 'train')):
        print(f'Streaming {split} from {DATASET} at {revision} ...', flush=True)
        stream = load_dataset(DATASET, split=split, revision=revision, streaming=True)
        chosen, ens, his, stat = reservoir(stream, config[split + '_size'], config['seed'] + i,
                                           config['max_chars'], blocked_en, blocked_hi)
        splits[split], stats[split] = chosen, stat
        if split != 'train':
            blocked_en.update(ens)
            blocked_hi.update(his)
    for split, rows in splits.items():
        for row in rows:
            row['id'] = digest(row)[:16]
        save_json(run / 'data' / f'{split}.json', rows)
    save_json(run / 'data/manifest.json', {'dataset': DATASET, 'revision': revision,
              'config_hash': digest(config), 'splits': stats,
              'test_hash': digest(splits['test']), 'normalization': 'NFC; whitespace collapse; script/length filters; source and target deduplication'})
    save_json(run / 'config.json', config)


def load_splits(run):
    return {split: read_json(Path(run) / 'data' / f'{split}.json') for split in ('train', 'validation', 'test')}

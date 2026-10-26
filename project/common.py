import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_CACHE', str(ROOT / '.cache' / 'huggingface' / 'hub'))
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / '.cache' / 'huggingface' / 'datasets'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(6, os.cpu_count() or 1))


def device_for(request):
    import torch
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run setup.ps1 or choose --device cpu.')
    return 'cuda' if request == 'auto' and torch.cuda.is_available() else ('cpu' if request == 'auto' else request)


def sync(device):
    if device == 'cuda':
        import torch
        torch.cuda.synchronize()

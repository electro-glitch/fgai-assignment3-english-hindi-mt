import pytest
import torch
from project.basic import BasicTransformer, train_tokenizer, encode, pad
from project.data import normalize, reservoir
from project.experiment import score
from project.common import digest, save_json


def test_normalization_and_split_leakage():
    rows = [{'translation': {'en': en, 'hi': hi}} for en, hi in
            [('Hello', 'नमस्ते'), ('Hello', 'नमस्कार'), ('Good day', 'शुभ दिन'), ('bad', ''), ('123', '१२३')]]
    assert normalize('  Hello\n world ') == 'Hello world'
    selected, _, _, stats = reservoir(rows, 1, 42, 400, {digest('Hello')})
    assert selected == [{'en': 'Good day', 'hi': 'शुभ दिन'}]
    assert stats['eligible'] == 1
    with pytest.raises(ValueError):
        reservoir(rows, 10, 42, 400)


def test_causal_mask_and_padding():
    torch.manual_seed(42)
    model = BasicTransformer(40, d_model=16, heads=2, layers=1, feedforward=32, dropout=0).eval()
    source = torch.tensor([[5, 6, 3]])
    original = model(source, torch.tensor([[2, 8, 9]]))
    changed = model(source, torch.tensor([[2, 8, 12]]))
    assert torch.allclose(original[:, :2], changed[:, :2], atol=1e-6)
    padded = model(torch.tensor([[5, 6, 3, 0, 0]]), torch.tensor([[2, 8, 9]]))
    assert torch.allclose(original, padded, atol=1e-5)


def test_training_reduces_loss_and_generation_stops():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = BasicTransformer(20, d_model=16, heads=2, layers=1, feedforward=32, dropout=0)
    source, target, labels = torch.tensor([[5, 3]]), torch.tensor([[2, 7]]), torch.tensor([[7, 3]])
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(source, target).flatten(0, 1), labels.flatten())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * .2
    model.eval()
    assert model.generate(source, 8)[0].tolist() == [2, 7, 3]


def test_tokenizer_and_metrics(tmp_path):
    rows = [{'en': 'Hello world', 'hi': 'नमस्ते दुनिया'}]
    tokenizer = train_tokenizer(rows, tmp_path / 'tokenizer.json', 300)
    for text in rows[0].values():
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
    assert len(encode(tokenizer, 'a very long sentence', 3)) <= 3
    assert pad([[1, 3], [3]]).tolist() == [[1, 3], [3, 0]]
    refs = ['यह एक बहुत अच्छा अनुवाद है।']
    metrics = score(refs, refs)
    assert metrics['bleu'] == pytest.approx(100)
    assert metrics['chrf'] == pytest.approx(100)
    with pytest.raises(ValueError):
        score([], refs)


def test_report_rejects_misaligned_test(tmp_path):
    from project.report import report
    save_json(tmp_path / 'data/manifest.json', {'test_hash': 'correct', 'config_hash': 'config', 'revision': 'rev'})
    save_json(tmp_path / 'results/basic.json', {'test_hash': 'wrong', 'config_hash': 'config'})
    with pytest.raises(ValueError, match='different test'):
        report(tmp_path)


def test_report_uses_published_manifest_without_raw_dataset(tmp_path):
    from project.report import report
    save_json(tmp_path / 'results/dataset_manifest.json',
              {'test_hash': 'correct', 'config_hash': 'config', 'revision': 'rev'})
    report(tmp_path)
    assert 'INCOMPLETE' in (tmp_path / 'results/REPORT.md').read_text(encoding='utf-8')

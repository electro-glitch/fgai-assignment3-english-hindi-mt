import math
import torch
from torch import nn


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=192, heads=4, layers=3, feedforward=768, dropout=0.1, max_positions=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pos = torch.arange(max_positions).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * frequencies), torch.cos(pos * frequencies)
        self.register_buffer('positions', pe)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, heads, layers, layers, feedforward,
                                          dropout, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids):
        return self.dropout(self.embedding(ids) * math.sqrt(self.d_model) + self.positions[:ids.size(1)])

    def encode(self, source):
        return self.transformer.encoder(self.embed(source), src_key_padding_mask=source.eq(0))

    def decode(self, target, memory, source_mask):
        mask = torch.triu(torch.ones(target.size(1), target.size(1), device=target.device, dtype=torch.bool), 1)
        hidden = self.transformer.decoder(self.embed(target), memory, tgt_mask=mask,
                    tgt_key_padding_mask=target.eq(0), memory_key_padding_mask=source_mask)
        return self.output(hidden)

    def forward(self, source, target):
        return self.decode(target, self.encode(source), source.eq(0))

    @torch.inference_mode()
    def generate(self, source, max_new_tokens, temperature=0.0):
        memory = self.encode(source)
        target = torch.full((source.size(0), 1), 2, dtype=torch.long, device=source.device)
        done = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
        for _ in range(max_new_tokens):
            logits = self.decode(target, memory, source.eq(0))[:, -1].clone()
            logits[:, [0, 2]] = -float('inf')
            next_id = (torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
                       if temperature > 0 else logits.argmax(-1))
            next_id = torch.where(done, torch.zeros_like(next_id), next_id)
            target = torch.cat((target, next_id[:, None]), dim=1)
            done |= next_id.eq(3)
            if done.all():
                break
        return target


def train_tokenizer(rows, path, vocab_size):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((row[lang] for row in rows for lang in ('en', 'hi')), trainer)
    tokenizer.save(str(path))
    return tokenizer


def encode(tokenizer, text, limit):
    return tokenizer.encode(text).ids[:limit - 1] + [3]


def pad(sequences, device='cpu'):
    return nn.utils.rnn.pad_sequence([torch.tensor(s, dtype=torch.long) for s in sequences],
                                    batch_first=True, padding_value=0).to(device)

"""Offline architecture checks using tiny random models, never benchmark results."""
import torch
from transformers import (MT5Config, MT5ForConditionalGeneration, LlamaConfig,
                          LlamaForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM)
from peft import LoraConfig, TaskType, get_peft_model


def test_mt5_lora_gradients_and_save_reload(tmp_path):
    from peft import PeftModel
    torch.set_num_threads(2)
    config = MT5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                       num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, eos_token_id=1)
    base = MT5ForConditionalGeneration(config)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=2,
                           target_modules=['q', 'v']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    loss = model(input_ids=torch.tensor([[3, 4, 1]]), labels=torch.tensor([[6, 7, 1]])).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n, p in model.named_parameters() if 'lora_' not in n)
    model.save_pretrained(tmp_path)
    reloaded = PeftModel.from_pretrained(MT5ForConditionalGeneration(config), tmp_path)
    assert reloaded.generate(input_ids=torch.tensor([[3, 4, 1]]), max_new_tokens=3).shape[1] <= 4


def test_decoder_architectures_generate():
    for cls, config in [
        (LlamaForCausalLM, LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)),
        (Gemma3ForCausalLM, Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))]:
        model = cls(config).eval()
        generated = model.generate(torch.tensor([[2, 3]]), max_new_tokens=2, do_sample=False, pad_token_id=0)
        assert 2 < generated.shape[1] <= 4

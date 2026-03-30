"""Sanity checks — forward pass, shapes, loss computation."""

import torch
from omegaconf import OmegaConf

from minimaker.model import GPT


def _toy_cfg():
    return OmegaConf.create({
        "name": "test",
        "n_layers": 2,
        "n_heads": 2,
        "d_model": 64,
        "d_ff": 256,
        "vocab_size": 256,
        "max_seq_len": 32,
        "dropout": 0.0,
    })


def test_forward_shape():
    cfg = _toy_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets():
    cfg = _toy_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    t = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, t)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_param_count():
    cfg = _toy_cfg()
    model = GPT(cfg)
    assert model.param_count() > 0


def test_activation_checkpointing():
    cfg = _toy_cfg()
    model = GPT(cfg)
    model.enable_activation_checkpointing()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    t = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, t)
    loss.backward()  # should not crash


def test_weight_tying():
    cfg = _toy_cfg()
    model = GPT(cfg)
    assert model.tok_emb.weight is model.lm_head.weight

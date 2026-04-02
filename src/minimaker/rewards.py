"""Reward functions for GRPO — simple, composable, transparent."""

from __future__ import annotations

import re

import torch


def format_reward(
    responses: list[str],
    required_tags: tuple[str, str] = ("<answer>", "</answer>"),
) -> torch.Tensor:
    """Reward for containing required format tags. Returns 0 or 1."""
    open_tag, close_tag = required_tags
    scores = []
    for resp in responses:
        has_format = open_tag in resp and close_tag in resp
        if has_format:
            # Check tags are in correct order
            open_idx = resp.index(open_tag)
            close_idx = resp.index(close_tag)
            scores.append(1.0 if close_idx > open_idx else 0.0)
        else:
            scores.append(0.0)
    return torch.tensor(scores)


def math_reward(responses: list[str], answers: list[str]) -> torch.Tensor:
    """Reward for correct math answers. Extracts number from <answer> tags."""
    scores = []
    for resp, gold in zip(responses, answers):
        match = re.search(r"<answer>\s*(.+?)\s*</answer>", resp)
        if match:
            extracted = match.group(1).strip()
            scores.append(1.0 if extracted == gold.strip() else 0.0)
        else:
            scores.append(0.0)
    return torch.tensor(scores)


def length_reward(
    responses: list[str], target_len: int = 200, max_penalty: float = 0.5
) -> torch.Tensor:
    """Soft penalty for responses far from target length."""
    scores = []
    for resp in responses:
        deviation = abs(len(resp) - target_len) / target_len
        scores.append(max(1.0 - deviation * max_penalty, 0.0))
    return torch.tensor(scores)


def combined_reward(
    responses: list[str],
    answers: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Weighted combination of reward functions.

    Default weights: format=0.3, math=0.7 (if answers provided), else format=1.0.
    """
    if weights is None:
        weights = {"format": 0.3, "math": 0.7} if answers else {"format": 1.0}

    total = torch.zeros(len(responses))
    if "format" in weights:
        total += weights["format"] * format_reward(responses)
    if "math" in weights and answers is not None:
        total += weights["math"] * math_reward(responses, answers)
    if "length" in weights:
        total += weights["length"] * length_reward(responses)
    return total

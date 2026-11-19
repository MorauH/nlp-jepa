"""Optimizer parameter groups."""

from __future__ import annotations


def parameter_groups(model, weight_decay: float):
    """Split parameters into decayed and non-decayed groups.

    Weight decay must not be applied to biases or normalization scales.
    Decaying an RMSNorm weight drives it toward zero, which collapses the
    residual stream and, in the predictor, turns every candidate into a copy
    of the input latent.
    """
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

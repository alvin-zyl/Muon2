import math
import warnings
from typing import Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR

from .muon2 import Muon2

# Optimizer name in the config -> Muon2 variant.
MUON_FAMILY = {
    "muon2": "muon2",      # Muon^2
    "muon2_f": "muon2_f",  # Muon^2-F
    "muon": "muon",        # Muon baseline
}


def get_optimizer(opt_config: dict, lr=1e-3) -> Tuple[type, dict]:
    """Map an optimizer config to an optimizer class and its keyword arguments."""
    name = opt_config["name"]
    if opt_config.get("lr") is None:
        warnings.warn("You have not specified a learning rate. A default value of 1e-3 will be used.")

    if name == "adamw":
        return torch.optim.AdamW, {
            "lr": lr,
            "weight_decay": opt_config.get("weight_decay", 0),
            "betas": tuple(opt_config.get("betas", (0.9, 0.999))),
            "eps": opt_config.get("eps", 1e-8),
            "fused": True,
        }

    if name not in MUON_FAMILY:
        raise KeyError(f"Unknown optimizer name '{name}'. Choose from {['adamw', *MUON_FAMILY]}.")
    return Muon2, {
        "lr": lr,
        "weight_decay": opt_config.get("weight_decay", 0),
        "variant": MUON_FAMILY[name],
        "ns_steps": opt_config.get("ns_steps", 5),
        "momentum": opt_config.get("momentum", 0.95),
        "nesterov": opt_config.get("nesterov", True),
        "betas": tuple(opt_config.get("betas", (0.95, 0.95))),
        "eps": opt_config.get("eps", 1e-8),
    }


def get_scheduler(opt_config: dict, opt: torch.optim.Optimizer, total_iterations: int) -> LambdaLR:
    """
    lr_schedule:
      constant        : constant lr
      constant-linear : constant for the first `constant_fraction` of training, then linear decay
                        to 10% of the peak lr (used for all experiments in the paper)
      warm-up-cosine  : linear warm-up for `warm_up_fraction` of training, then cosine decay to 0
    """
    name = opt_config.get("lr_schedule", "constant")

    if name == "constant":
        lr_fun = lambda step: 1.0

    elif name == "constant-linear":
        num_constant_steps = int(opt_config.get("constant_fraction", 0.0) * total_iterations)

        def lr_fun(step):
            if step < num_constant_steps:
                return 1.0
            return max(0.1, 1.0 - (step - num_constant_steps) / (total_iterations - num_constant_steps))

    elif name == "warm-up-cosine":
        num_warmup_steps = int(opt_config.get("warm_up_fraction", 0.0) * total_iterations)

        def lr_fun(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, total_iterations - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    else:
        raise ValueError(f"Unknown learning rate schedule '{name}'.")

    return LambdaLR(opt, lr_lambda=lr_fun)

"""Muon^2: Boosting Muon via Adaptive Second-Moment Preconditioning.

Muon^2 applies Adam-style second-moment preconditioning to the momentum *before*
orthogonalization, which moves the singular values of the Newton-Schulz input out
of the "dead zone" near zero and lets fewer iterations reach a good polar factor:

    M_t = b1 * M_{t-1} + (1 - b1) * G_t
    V_t = b2 * V_{t-1} + (1 - b2) * G_t * G_t        # added by Muon^2
    O_t = NewtonSchulz(M_t / (sqrt(V_t) + eps), K)   # added by Muon^2: preconditioning
    W_{t+1} = W_t - lr * sqrt(m / n) * O_t

The same class also implements the Muon baseline (variant="muon"), so both
share one code path for the update and the AdamW fallback.
"""
import math

import torch

from .polar import newton_schulz

VARIANTS = ("muon2", "muon2_f", "muon")

# Parameters whose name contains one of these are always updated with AdamW.
ADAMW_PARAM_NAMES = ("embeddings", "embed_tokens", "wte", "lm_head", "wpe")


class Muon2(torch.optim.Optimizer):
    """
    2D weight matrices are updated with an orthogonalized step; every other
    parameter (embeddings, lm_head, biases, norms) is updated with AdamW using
    the same lr, weight decay, betas and eps.

    Parameters
    ----------
    named_params : iterable of (name, parameter), e.g. ``model.named_parameters()``
    variant : str
        "muon2"   : Muon^2, Adam-style preconditioning M / (sqrt(V) + eps) before orthogonalization.
        "muon2_f" : Muon^2-F, memory-efficient Muon^2 that keeps only Adafactor-style row/column
                    second-moment statistics and reconstructs V from their outer product.
        "muon"    : Muon, orthogonalized Nesterov momentum.
    ns_steps : int
        Number of Newton-Schulz iterations.
    momentum, nesterov :
        Momentum used by the "muon" variant.
    betas, eps :
        (b1, b2) and eps of the Muon^2 moments and of the AdamW fallback.

    The update of an m x n weight is scaled by sqrt(m / n).
    """

    def __init__(
        self,
        named_params,
        lr=1e-3,
        weight_decay=0.1,
        variant="muon2",
        ns_steps=5,
        momentum=0.95,
        nesterov=True,
        betas=(0.95, 0.95),
        eps=1e-8,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got '{variant}'.")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            variant=variant,
            ns_steps=ns_steps,
            momentum=momentum,
            nesterov=nesterov,
            betas=tuple(betas),
            eps=eps,
        )

        matrix_params, adamw_params = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(excl in name for excl in ADAMW_PARAM_NAMES):
                assert p.ndim == 2, f"{name}: expected a 2D weight, got shape {tuple(p.shape)}"
                matrix_params.append(p)
            else:
                adamw_params.append(p)

        super().__init__(matrix_params + adamw_params, defaults)

        for p in matrix_params:
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            variant = group["variant"]
            momentum = group["momentum"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            # ------------------------------------------------------------------
            #  Orthogonalized update for 2D weight matrices
            # ------------------------------------------------------------------
            for p in group["params"]:
                state = self.state[p]
                if not state["use_muon"] or p.grad is None:
                    continue
                g = p.grad

                # ---- Pre-orthogonalization: build the matrix to orthogonalize ----
                if variant == "muon2":
                    if "moment1" not in state:
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["moment1"].lerp_(g, 1 - beta1)
                    state["moment2"].lerp_(g.square(), 1 - beta2)
                    g = state["moment1"] / (state["moment2"].sqrt() + eps)

                elif variant == "muon2_f":
                    if "v_r" not in state:
                        state["v_r"] = torch.zeros(g.size(0), 1, dtype=torch.float32, device=g.device)
                        state["v_c"] = torch.zeros(1, g.size(1), dtype=torch.float32, device=g.device)
                        if beta1 > 0:
                            state["moment1"] = torch.zeros_like(g)
                    # Factored second moment (float32 for numerical stability)
                    g_sq = g.float().square()
                    state["v_r"].mul_(beta2).add_(g_sq.sum(dim=-1, keepdim=True), alpha=1 - beta2)
                    state["v_c"].mul_(beta2).add_(g_sq.sum(dim=-2, keepdim=True), alpha=1 - beta2)
                    v_r, v_c = state["v_r"], state["v_c"]   # [m, 1], [1, n]
                    v_hat = v_r * v_c / (v_r.sum() + eps)   # rank-1 reconstruction, [m, n]
                    if beta1 > 0:
                        state["moment1"].lerp_(g, 1 - beta1)
                        numerator = state["moment1"].float()
                    else:
                        numerator = g.float()
                    g = (numerator / (v_hat.sqrt() + eps)).to(g.dtype)

                else:  # muon: (Nesterov) momentum
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                # ---- Orthogonalization ----
                u = newton_schulz(g, group["ns_steps"])

                fan_out, fan_in = p.shape
                adjusted_lr = lr * math.sqrt(fan_out / fan_in)
                p.mul_(1 - lr * weight_decay)
                p.add_(u, alpha=-adjusted_lr)

            # ------------------------------------------------------------------
            #  AdamW for embeddings, lm_head and non-matrix parameters
            # ------------------------------------------------------------------
            for p in group["params"]:
                state = self.state[p]
                if state["use_muon"] or p.grad is None:
                    continue
                g = p.grad
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr / scale)

        return loss

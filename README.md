# Muon²: Boosting Muon via Adaptive Second-Moment Preconditioning

[![arXiv](https://img.shields.io/badge/arXiv-2604.09967-b31b1b.svg)](https://arxiv.org/abs/2604.09967)
[![Project page](https://img.shields.io/badge/Project-page-blue.svg)](https://alvin-zyl.github.io/muon2/)
[![EMNLP 2026 Oral](https://img.shields.io/badge/EMNLP%202026-Oral-green.svg)](https://arxiv.org/abs/2604.09967)

Official PyTorch implementation of
*Muon²: Boosting Muon via Adaptive Second-Moment Preconditioning* (EMNLP 2026, Oral).

Muon² applies Adam-style second-moment preconditioning to the momentum **before** Newton–Schulz
orthogonalization. It reaches lower perplexity than Muon with 40% fewer Newton–Schulz iterations,
and its memory-efficient variant **Muon²-F** keeps Muon's memory footprint.

## Method

Muon orthogonalizes the momentum with a few Newton–Schulz (NS) iterations, which push every singular
value towards 1. Tiny singular values need many iterations to get there: with 5 steps, values below
about 10⁻³ never reach the target range, and early in training a large share of Muon's normalized
momentum spectrum sits in this *dead zone*. Dividing the momentum element-wise by the square root of
its second moment shifts the spectrum about ten times higher and narrows it, so NS converges in fewer
steps and the update is better aligned with the exact polar factor.

For each weight matrix W ∈ ℝ^{m×n}:

| Step | Update | |
|---|---|---|
| 1 | Gₜ = ∇_W ℒ(Wₜ) | gradient |
| 2 | Mₜ = β₁ Mₜ₋₁ + (1 − β₁) Gₜ | momentum |
| 3 | Vₜ = β₂ Vₜ₋₁ + (1 − β₂) Gₜ ⊙ Gₜ | **added: second moment** |
| 4 | M̃ₜ = Mₜ ⊘ (√Vₜ + ε) | **added: preconditioning** |
| 5 | Oₜ = NewtonSchulz(M̃ₜ, K) | orthogonalize |
| 6 | Wₜ₊₁ = Wₜ − η √(m/n) Oₜ | update |

Only steps 3 and 4 differ from Muon. **Muon²-F** stores only Adafactor-style row and column statistics
of Gₜ ⊙ Gₜ and reconstructs Vₜ from their outer product, so it needs no extra m×n buffer.

In code ([`utils/optim/muon2.py`](utils/optim/muon2.py)):

```python
state["moment1"].lerp_(g, 1 - beta1)            # M_t
state["moment2"].lerp_(g.square(), 1 - beta2)   # V_t
g = state["moment1"] / (state["moment2"].sqrt() + eps)
u = newton_schulz(g, ns_steps)
p.add_(u, alpha=-lr * math.sqrt(fan_out / fan_in))
```

## Results

Validation perplexity (lower is better) of language-model pre-training; learning rates are tuned by a sweep.

| Model | Muon (3 NS) | Muon² (3 NS) | Muon²-F (3 NS) | Muon (5 NS) | Muon² (5 NS) | Muon²-F (5 NS) |
|---|---|---|---|---|---|---|
| GPT-Small (124M) | 32.70 | **28.12** | 28.30 | 29.51 | 27.95 | **27.93** |
| GPT-Base (362M)  | 24.69 | **20.39** | 21.17 | 21.47 | **19.96** | 20.58 |
| GPT-Large (774M) | 21.13 | **16.99** | 17.69 | 17.56 | **16.52** | 16.55 |
| LLaMA-60M        | 26.37 | **24.59** | 24.68 | 24.98 | **24.60** | 24.66 |
| LLaMA-350M       | 14.91 | **13.44** | 13.55 | 14.03 | 13.46 | **13.44** |
| LLaMA-1B         | 11.63 | **10.42** | 10.49 | 10.62 | **10.21** | **10.21** |
| LLaMA-13B        | 10.54 | **9.11** | – | 9.24 | **8.71** | – |
| MoE-7B-A1B       | 11.93 | **9.85** | – | 10.61 | **9.66** | – |

Muon² with 3 NS steps beats Muon with 5 NS steps at every scale, i.e. better models with 40% fewer NS iterations.

Comparison with other Muon variants (validation perplexity):

| Optimizer | GPT-Small (3 NS) | GPT-Small (5 NS) | GPT-Base (3 NS) | GPT-Base (5 NS) |
|---|---|---|---|---|
| Muon         | 32.70 | 29.51 | 24.69 | 21.47 |
| PolarExpress | 30.01 | 29.42 | 22.74 | 21.16 |
| Turbo-Muon   | 29.70 | 29.66 | 23.46 | 21.93 |
| NorMuon      | 30.35 | 28.40 | 23.33 | 21.27 |
| AdaMuon      | 31.20 | 29.30 | 26.07 | 22.42 |
| **Muon²**    | **28.12** | **27.95** | **20.39** | **19.96** |

See the [paper](https://arxiv.org/abs/2604.09967) and [project page](https://alvin-zyl.github.io/muon2/)
for the spectral analysis, memory and wall-clock measurements, and downstream evaluations.

## Setup

```bash
conda create -n muon2 python=3.10 -y
conda activate muon2
pip install -r requirements.txt
```

## Data

Tokenize a FineWeb sample with the GPT-2 tokenizer into binary shards of 10⁸ tokens:

```bash
export DATA_DIR=/path/to/data            # where shards are written and read (default: ./data)
python process_data.py --name fineweb10B   # GPT-Small and GPT-Base
python process_data.py --name fineweb100B  # GPT-Large (15.5B tokens > the 10B sample)
```

Shards are written to `$DATA_DIR/<name>-gpt2/`. The first shard (`val_000000.bin`) is the validation
set and the rest (`train_000001.bin`, …) are training data. `total_tokens_B` in a config limits
training to the first `10 × total_tokens_B` training shards.

## Training

Every experiment has its own config in `configs/<model>/<optimizer>_ns<steps>.yaml`:

```bash
# single node, 4 GPUs
torchrun --standalone --nproc_per_node=4 run.py --config configs/gpt_small/muon2_ns3.yaml

# multi-node with SLURM (edit the #SBATCH header for your cluster)
CONFIG=configs/gpt_base/muon2_ns5.yaml sbatch -N 8 scripts/slurm.sh
```

- `tokens_processed` is the global batch size in tokens, and gradient accumulation adapts to the number of
  GPUs. It must be divisible by `#GPUs × batch_size × context_length`. Our runs used 32 GPUs for
  GPT-Small/Base and 64 GPUs for GPT-Large.
- Each config uses the best learning rate from the sweep in the paper, {0.003, 0.005, 0.01, 0.02, 0.04}.
  `lr` is a list; to rerun the sweep, list several values and each one is trained as a separate run.
- The numbers in the results table are the lowest validation perplexity (exp of `val_losses`) during training.
- Per-step training loss, validation loss, learning rate and gradient norm are saved to
  `outputs/results/<model>/<config>/*.json`. Checkpoints go to `outputs/checkpoints/`.
  Runs are also logged to Weights & Biases; delete the `wandb` block in a config to disable it.

### Optimizers

| Optimizer | `name` | Configs |
|---|---|---|
| **Muon²** | `muon2` | `configs/gpt_{small,base,large}/muon2_ns{3,5}.yaml` |
| **Muon²-F** | `muon2_f` | `configs/gpt_{small,base,large}/muon2_f_ns{3,5}.yaml` |
| Muon (baseline) | `muon` | `configs/gpt_{small,base,large}/muon_ns{3,5}.yaml` |

All three share one implementation ([`Muon2`](utils/optim/muon2.py)). 2D weight matrices get the
orthogonalized update, scaled by √(m/n) for an m×n weight; embeddings, the LM head and all 1D
parameters are updated with AdamW using the same learning rate, weight decay, `betas` and `eps`.
Optimizer keys in the configs:

| Key | Meaning |
|---|---|
| `ns_steps` | number of Newton–Schulz iterations |
| `betas` | (β₁, β₂) of the Muon² moments and of AdamW (default `[0.95, 0.95]`) |
| `momentum`, `nesterov` | Nesterov momentum of the Muon baseline |
| `lr_schedule`, `constant_fraction` | `constant-linear`: constant lr for the first 40% of training, then linear decay to 0.1× |

### Using Muon² in your own training loop

```python
from utils.optim.muon2 import Muon2

optimizer = Muon2(model.named_parameters(), lr=0.02, weight_decay=0.1,
                  variant="muon2",   # or "muon2_f"
                  ns_steps=3)
```

Parameters whose names contain `wte`, `wpe`, `lm_head`, `embeddings` or `embed_tokens` are routed to
AdamW, so pass `named_parameters()`, not `parameters()`.

## Repository structure

```
run.py                  training entry point (loops over optimizers and learning rates in a config)
process_data.py         FineWeb download + GPT-2 tokenization into binary shards
configs/                one YAML per (model, optimizer, NS steps)
scripts/slurm.sh        multi-node launcher
utils/model.py          GPT-2 model
utils/train.py          training loop (gradient accumulation, DDP, evaluation, checkpointing)
utils/dataloader.py     sharded data loader
utils/optim/muon2.py    Muon² / Muon²-F and the Muon baseline
utils/optim/polar.py    Newton–Schulz orthogonalization
utils/optim/factory.py  config -> optimizer / lr scheduler
```

## Citation

```bibtex
@inproceedings{liu2026muon2,
  title     = {Muon$^2$: Boosting Muon via Adaptive Second-Moment Preconditioning},
  author    = {Liu, Ziyue and Zhang, Ruijie and Wang, Zhengyang and Zhao, Yequan and
               Su, Yupeng and Yang, Zi and Zhang, Zheng},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## Acknowledgements

The training framework is adapted from [MuonPlus](https://github.com/K1seki221/MuonPlus), which builds on
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), [build-nanogpt](https://github.com/karpathy/build-nanogpt)
and [llm.c](https://github.com/karpathy/llm.c). The Muon update follows [KellerJordan/Muon](https://github.com/KellerJordan/Muon).

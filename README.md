# TinyZero

TinyZero is a compact research codebase implementing lightweight planning and RL-style environments and models. It collects
algorithms, environment definitions, model implementations and helper scripts used for experimentation and small-scale training.

This repository is organized to be easy to inspect and extend: use it to reproduce experiments, prototype new environments, or
adapt the transformer-based models and planning algorithms for your own tasks.

## Highlights

- Simple, well-structured Python layout (algorithms, environments, models, utils, scripts, configs).
- Small training and evaluation scripts that can be used as examples or starting points.
- Multiple example environments including toy tasks (countdown, multiplication) and a Frozen Lake variant.
- Transformer model and FSDP wrapper included for distributed training experiments.

## Quickstart

Prerequisites
- Python 3.10+ (3.11+ recommended)
- A CUDA-enabled GPU for larger training runs (optional for smoke tests)
- Install Python dependencies:

```bash
python -m pip install -r "/Users/amankhan/Downloads/TinyZero new/TinyZero_RAGEN/requirements.txt"
```

Run a smoke test

```bash
# Run the lightweight smoke script to verify everything is working
python "/Users/amankhan/Downloads/TinyZero new/TinyZero_RAGEN/scripts/run_smoke.py"
```

Train a model (example)

```bash
python "/Users/amankhan/Downloads/TinyZero new/TinyZero_RAGEN/train.py" --config configs/rtx4090_100iter.yaml
```

Evaluate

```bash
python "/Users/amankhan/Downloads/TinyZero new/TinyZero_RAGEN/evaluate.py" --config configs/default.yaml
```

## Project layout

- `algorithms/` — planner and algorithm implementations (astarpo, ragen_astarpo, trajectory helpers).
- `environments/` — environment interfaces and tasks (countdown, frozenlake_env, multiplication).
- `models/` — model code and wrappers (transformer, fsdp_wrapper).
- `configs/` — YAML config files for runs, smoke tests, and hardware-specific presets.
- `scripts/` — convenience scripts for running modal jobs, smoke tests, and small experiments.
- `utils/` — small utilities (distributed setup, logging helpers).
- `logs/`, `logs_rtx4090/`, `checkpoints_*` — runtime artifacts produced by experiments (kept here from attachments).

Files of interest
- `train.py` — entry point for training experiments.
- `evaluate.py` — evaluation harness and metrics aggregation.
- `Dockerfile` / `modal.yaml` — container and cloud execution helpers.

## Usage tips

- Use the provided YAML files in `configs/` as starting points. They contain sensible defaults and hardware-aware settings.
- For quick development, run `scripts/run_smoke.py` which uses a small config and short checkpointing.
- If you plan to run longer training on a GPU machine, ensure your CUDA/cuDNN environment matches the Python versions and the installed packages.

## Clean up and housekeeping

The copy included runtime files (logs, checkpoints, and compiled `.pyc` files). If you want a tidy repository without artifacts, consider running:

```bash
find "/Users/amankhan/Downloads/TinyZero new/TinyZero_RAGEN" -name "__pycache__" -or -name "*.pyc" -or -name "logs*" -or -name "checkpoints*" | xargs rm -rf --
```

Or ask me and I can remove them for you while leaving the source code intact.

## Development & contributions

- Code style: keep functions small and well-documented. Tests aren't included by default — add small unit tests when adding new environments or model components.
- If you want, I can initialize a `git` repository in this folder and make an initial commit, add a `.gitignore`, and create a `CONTRIBUTING.md` template.

## Contact & support

If you need help running experiments, cleaning large artifacts, or preparing this repository for sharing (git init, squash history, or packaging), tell me what you'd like next and I will prepare it.

---

This README was generated and tailored to the current copy of the project. It aims to be friendly and practical while leaving implementation details in the code where appropriate.
# TinyZero_AstarPO — Reward-Augmented Generation with A* Policy Optimization

TinyZero_AstarPO integrates RAGEN (Reward-Augmented Generation) with an A* policy optimization (A*PO) approach in a compact codebase for rapid experimentation. Designed for coursework and lightweight research, it avoids complex infrastructure—helping users focus on algorithms rather than engineering overhead.

Why use this repository?
- Pedagogical: built for quick comprehension, reproducibility, and small-scale RL/LM experiments.
- Minimalist: pure PyTorch (with optional FSDP), no dependency on large RL/LLM frameworks.
- Extendible: add tasks, environments, or LM models easily for coursework or research.

Included components 
- `train.py` — Main entrypoint (TinyZeroTrainer). Collects trajectories, runs the selected algorithm (`ragen` or `astarpo`), logs metrics, and checkpoints models.
- `evaluate.py` — Evaluation tool for saved model checkpoints.
- `algorithms/astarpo.py` — All A*PO logic, V estimator, and utilities for trajectory scoring.
- `algorithms/ragen_astarpo.py` — RAGEN implemented over A*PO for integrated value computation and policy updates.
- `environments/` — Supports `countdown`, `multiplication`, and a Gym-compatible `FrozenLake` (`frozenlake_env.py`).
- `models/transformer.py` — A small transformer-style language model architecture used for policy and reference models.
- `configs/` — Example configs: `default.yaml` for standard runs, `smoke.yaml` for tiny quick-tests.
- `scripts/run_smoke.py` — Quick-start smoke test—runs a minimal model for one iteration locally.
- `Dockerfile`, `modal.yaml`, `modal_instructions.md` — Helpers to deploy on modal.com for scalable experiments.

What problem does this solve?
- Demonstrates how to implement RAGEN: train a generative policy by sampling candidate completions, scoring them with task rewards, and using those scores to update the policy.
- Shows how to replace PPO/GRPO with A*PO in the policy optimization loop, producing stable, easy-to-verify updates.
- Provides a compact environment + LM-in-the-loop setup so students can run RL-style experiments with transformer LMs on small benchmarks.

How the core algorithm works (short)
For each environment prompt, RAGEN:

1. Samples K candidate completions from the policy model.
2. Calculates task rewards for each completion.
3. Computes $V^*(x)$ for the prompt using A*PO’s log-sum-exp estimator:

	$$V^*(x) = \beta \log \mathbb{E}_{y\sim\pi_{ref}}\left[\exp\left(\frac{r(x,y)}{\beta}\right)\right]$$

4. Sets the advantage as $A^*(x,y) = r(x,y) - V^*(x)$.
5. Uses $A^*(x,y)$ as a regression target for the scaled log-ratio $\beta\log\left(\frac{\pi_\theta}{\pi_{ref}}\right)$.
6. Trains with MSE regression (plus a small entropy bonus), yielding a simple and robust learning signal for small benchmarks.

Quickstart (5-minute grader guide)

1. Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run a local smoke test:

```bash
python scripts/run_smoke.py
```

This smoke test builds the trainer, collects a few short trajectories, performs one RAGEN update, and prints a small stats dictionary. If imports are missing the script prints clear diagnostics.

3. Reproducing experiments (e.g., FrozenLake):
- Edit `configs/default.yaml` to set `algorithm.name`, `training.total_iterations`, and other hyperparameters.
- Run:

```bash
python train.py --config configs/default.yaml
```

Grader / professor quick-edits and review checklist
- Switch algorithm: change `algorithm.name` in `configs/default.yaml` between `ragen` and `astarpo`.
- Modify reward: edit `compute_reward` in `train.py` to change task shaping.
- Tweak hyperparameters: adjust `astarpo.beta`, `num_completions`, or `training.total_iterations` in the config.

Key files to inspect
- `algorithms/astarpo.py` — numeric details for V* estimation and stability (log-sum-exp implementation).
- `algorithms/ragen_astarpo.py` — RAGEN update loop and how it uses A*PO helpers.
- `train.py` — environment glue, trajectory collection, reward functions, and the main training loop.

Modal deployment and cost controls
- `Dockerfile`, `modal.yaml`, and `modal_instructions.md` are included to help run heavier experiments on modal.com.
- Use `configs/smoke.yaml` for development to keep costs low. Cap `training.total_iterations` and `astarpo.num_trajectories` to control GPU spend.

Tasks, evaluation, and expected results
- Built-in tasks: `countdown` and `multiplication` (text-based), plus Gym `FrozenLake` for RL baseline.
- Smoke test verifies the pipeline but does not reproduce the full paper; run longer experiments on modal to reproduce paper figures and collect metrics with `evaluate.py`.

How to get help or request a demo
- Ask for a live demo (10-minute slide deck + walkthrough).
- Request a Modal SDK script (`modal_submit.py`) and I will add it for you to run with your Modal credentials.
- Request a short FrozenLake experiment run (CPU or GPU) and I will produce a small metrics table and failure-case examples.

License
- MIT License. See `LICENSE` for details.

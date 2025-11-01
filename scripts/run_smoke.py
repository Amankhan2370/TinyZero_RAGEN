#!/usr/bin/env python3
"""
Small smoke runner to exercise the trainer for one iteration.

This script attempts to import the project's training code and run a
single training iteration using the lightweight `configs/smoke.yaml`.
It is intentionally tolerant: if heavy dependencies (torch, transformers)
aren't installed, it will print diagnostic information rather than failing.
"""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def main():
    try:
        from train import TinyZeroTrainer
    except Exception as e:
        print("Could not import TinyZeroTrainer:")
        traceback.print_exc()
        print("This smoke runner requires the repository's python dependencies (torch, transformers, gym).")
        print("If you want to run a full smoke test, install requirements with: pip install -r requirements.txt")
        return

    try:
        trainer = TinyZeroTrainer(config_path=str(
            ROOT / 'configs' / 'smoke.yaml'))
        print("Trainer constructed. Running one training iteration...")
        stats = trainer.train_iteration()
        print("Iteration stats:")
        print(stats)
    except Exception:
        print("Trainer run failed:")
        traceback.print_exc()


if __name__ == '__main__':
    main()

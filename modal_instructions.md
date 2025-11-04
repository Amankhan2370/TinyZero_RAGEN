# Deploying TinyZero_AstarPO on Modal (step-by-step)

This document walks through preparing and running the repository on modal.com. It assumes you have a modal account and the modal CLI or UI access.

Important safety notes
- Keep development runs on CPU or use `configs/smoke.yaml` to avoid unexpected GPU spend.
- For final experiments, set `distributed.use_fsdp: true` in your config and select a GPU instance with enough memory.
- Monitor modal's cost dashboard and terminate runs early if cost looks high.

1) Build a Docker image locally (optional)

If you want to build and test the Docker image locally before uploading to modal:

```bash
docker build -t tinyzero_astarpo:latest .
docker run --gpus all -it --rm tinyzero_astarpo:latest /bin/bash
# inside container, you can run: python scripts/run_smoke.py
```

2) Create a modal image / function

Modal supports building an image from your repo. You can either push a pre-built image to a container registry or use Modal's image build. The exact modal CLI commands depend on the modal SDK version, but here's a typical flow:

- Use the modal dashboard to create a new image and point it to your Git repo.
- Use the `modal.yaml` template in this repository as a starting point for run options.

3) Run a smoke test on modal

Use `configs/smoke.yaml` to run a cheap test (CPU or a very small GPU) on modal. This keeps costs low and verifies everything runs.

4) Run a controlled GPU experiment

- Edit `configs/default.yaml` and set `training.total_iterations` to a conservative value for your target budget.
- Flip `distributed.use_fsdp` to `true` only when you have proper multi-GPU setup.
- Launch the run using modal's job UI or CLI, using the `modal.yaml` template to request 1 GPU, 4 CPUs, and about 32GB RAM.

5) Cost control checklist
- Start with `configs/smoke.yaml`.
- Limit `training.total_iterations` and `astarpo.num_trajectories`.
- Use single-seed runs for debugging; increase seeds only for final results.
- Use modal preemption/stop features to terminate long runs.

6) Logs and checkpoints
- The trainer writes checkpoints to the configured `experiment.checkpoint_dir` in the config. Make sure the modal container mounts or uploads that directory to persistent storage if you need it after the instance terminates.

7) Troubleshooting
- If you run out of GPU memory, reduce model size in config (`model.hidden_size`, `num_layers`) or increase the instance size.
- If FSDP wrapping errors occur, ensure you're using a compatible PyTorch CUDA image and that `distributed.use_fsdp` matches the runtime setup.

If you'd like, I can produce a ready-to-run `modal` Python script using Modal's SDK for you to paste into your modal project (I can't run it here without your modal credentials). Tell me if you'd prefer that and which GPU type you want to target.

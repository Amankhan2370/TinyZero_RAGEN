import modal
import os
from pathlib import Path

# Use App instead of Stub (Modal API update)
app = modal.App("tinyzero-ragen")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.1",
    "transformers>=4.36.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0.1",
    "tensorboard>=2.14.0",
    "accelerate>=0.25.0",
    "einops>=0.7.0",
    "dataclasses-json>=0.6.0",
    "tqdm>=4.65.0",
    "wandb>=0.16.0",
    "omegaconf>=2.3.0",
    "gymnasium>=1.2.0"
).run_commands("apt-get update && apt-get install -y git")

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
)
def train_on_modal():
    import subprocess
    import sys
    
    # Clone your repository
    print("Cloning repository...")
    subprocess.run([
        "git", "clone", 
        "https://github.com/Amankhan2370/TinyZero_RAGEN.git",
        "/tmp/tinyzero"
    ], check=True)
    
    # Change to project directory
    os.chdir("/tmp/tinyzero")
    
    print("Starting training on Modal GPU...")
    
    # Run training with smoke config for faster results
    subprocess.run([
        "python", "train.py", 
        "--config", "configs/smoke.yaml"
    ], check=True)
    
    print("Training completed!")
    return {"status": "completed"}

@app.local_entrypoint()
def main():
    print("Deploying to Modal NEU workspace...")
    print("Starting training on Modal GPU...")
    result = train_on_modal.remote()
    print(f"Training completed! Status: {result['status']}")
"""
Helper to submit a smoke run to Modal.

This script attempts to use the Modal Python SDK if available. If the SDK
is not installed or you prefer the CLI, the script prints the equivalent
`modal` CLI commands to build the image and run the smoke job.

Usage (recommended):
  # ensure you're logged in with `modal login` or have MODAL_API_KEY set
  python modal_submit.py --config configs/smoke.yaml --cpu-only

Notes:
 - This script does not require the Modal SDK; it will print CLI steps when
   the SDK is not installed.
 - The Dockerfile and modal.yaml in this repo are configured to build a
   PyTorch image; adjust resources in `modal.yaml` before launching full runs.
"""
import argparse
import shlex
import subprocess
import sys


def print_cli_instructions(config_path: str, cpu_only: bool):
    print("""
Use these commands on a machine with the `modal` CLI installed and authenticated.

1) Build an image from the repo (local build):

   modal image build --local -t tinyzero-astarpo:latest .

2) Run the image with the smoke config (CPU-only run):

   modal run --name tinyzero-smoke \
     --image tinyzero-astarpo:latest \
     --cpu 1 --memory 4G \
     -- bash -lc "./scripts/run_modal.sh {config}"

Replace `{config}` with the config path.
""".format(config=config_path))


def try_modal_sdk_run(config_path: str, cpu_only: bool):
    try:
        import modal
    except Exception:
        print("Modal SDK not available in this environment. Falling back to CLI instructions.\n")
        print_cli_instructions(config_path, cpu_only)
        return

    image = modal.Image.build_from_dockerfile(".")

    stub = modal.Stub("tinyzero-astarpo")

    @stub.function(image=image, cpu=1 if cpu_only else 4, memory="4G" if cpu_only else "32G")
    def run_smoke():
        import subprocess
        subprocess.run(
            ["bash", "-lc", f"./scripts/run_modal.sh {config_path}"], check=True)

    print("Submitting job to Modal via SDK...")
    with stub.run():
        run_smoke.call()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Prefer CPU-only resources")
    args = parser.parse_args()

    try_modal_sdk_run(args.config, args.cpu_only)


if __name__ == "__main__":
    main()

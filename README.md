# function-calling-fine-tuning

Local preparation repo for xLAM-based function-calling fine-tuning on Qwen 3 with NVIDIA NeMo AutoModel.

## Pinned Components

- Repo commit: record the exact run commit with `git rev-parse HEAD`
- AutoModel release: `v0.3.0`
- AutoModel container: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- Model: `Qwen/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Dataset: `Salesforce/xlam-function-calling-60k`
- Dataset revision: `26d14ebfe18b1f7b524bd39b404b50af5dc97866`
- Dataset license: `cc-by-4.0`
- Python dependencies: pinned in [requirements-dev.txt](/home/val/Documents/py-projects/function-calling-fine-tuning/requirements-dev.txt)

## Layout

- Smoke configs: [configs/exp00_smoke](/home/val/Documents/py-projects/function-calling-fine-tuning/configs/exp00_smoke)
- Local scripts: [scripts](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts)
- Core package: [src/function_calling_ft](/home/val/Documents/py-projects/function-calling-fine-tuning/src/function_calling_ft)
- Tests: [tests](/home/val/Documents/py-projects/function-calling-fine-tuning/tests)
- Public manifests: [data/manifests](/home/val/Documents/py-projects/function-calling-fine-tuning/data/manifests)

## Local Workflow

1. Download the gated xLAM dataset:
   `./.venv/bin/python scripts/download_xlam.py`
2. Inspect a deterministic 100-row sample:
   `./.venv/bin/python scripts/inspect_xlam.py`
3. Build the 200-example smoke split:
   `./.venv/bin/python scripts/select_smoke_sample.py`
4. Normalize and validate it:
   `./.venv/bin/python scripts/normalize_xlam.py`
   `./.venv/bin/python scripts/validate_examples.py`
5. Run the full local gate:
   `make preflight`

## EC2 + Docker

- Base image: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- Repo Dockerfile: [Dockerfile](/home/val/Documents/py-projects/function-calling-fine-tuning/Dockerfile)
- Host bootstrap: [scripts/bootstrap_instance.sh](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts/bootstrap_instance.sh)
- Container smoke orchestration: [scripts/smoke_run.sh](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts/smoke_run.sh)

Mount checkpoints and results onto persistent storage. Do not rely on state inside a container started with `--rm`.

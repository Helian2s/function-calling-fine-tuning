# function-calling-fine-tuning

Local preparation repo for xLAM-based function-calling fine-tuning on Qwen 3 with NVIDIA NeMo AutoModel.

## Pinned Components

- Repo commit: record the exact run commit with `git rev-parse HEAD`
- AutoModel container: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- AutoModel container digest: `sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2`
- Runtime-reported `nemo_automodel` version: `0.2.0rc0`
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
- AutoModel CLI: `automodel finetune llm -c configs/exp00_smoke/smoke_qlora.yaml`
- Runtime constants: [configs/common/exp00.env](/home/val/Documents/py-projects/function-calling-fine-tuning/configs/common/exp00.env)
- Repo Dockerfile: [Dockerfile](/home/val/Documents/py-projects/function-calling-fine-tuning/Dockerfile)
- Host bootstrap: [scripts/bootstrap_instance.sh](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts/bootstrap_instance.sh)
- Container runner: [scripts/run_automodel_container.sh](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts/run_automodel_container.sh)
- Container smoke orchestration: [scripts/smoke_run.sh](/home/val/Documents/py-projects/function-calling-fine-tuning/scripts/smoke_run.sh)

Persistent host paths under `/mnt/workspace` are mounted to `/workspace` inside
the container. Checkpoints must land under `/workspace/checkpoints`; results,
logs, and run-info must land under their matching `/workspace/*` mounts. Do not
rely on state inside a container started with `--rm`.

## C4 Version Correction

- Previously expected AutoModel version: `0.3.0`
- Runtime-observed AutoModel version: `0.2.0rc0`
- Container tag: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- Container digest: `sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2`
- CLI validation: passed
- Recipe target resolution: passed
- C4 disposition: pass with corrected version evidence

The earlier expectation that this image contained AutoModel `0.3.0` was
incorrect. The actual container was validated directly: `nemo_automodel`
imports successfully, the AutoModel CLI is available, Experiment 0 LoRA and
QLoRA target classes resolve, persistent checkpoint paths validate, and
template and loss-mask checks pass. The container must not be changed solely to
match the previously expected package-version string; runtime execution results
are authoritative for Experiment 0.

Useful C0 commands:

```bash
make preflight
make smoke-preflight
make smoke-baseline
make smoke-train
make smoke-reload-check
make smoke-evaluate
make smoke-run
scripts/run_automodel_container.sh --pull --login-ngc make smoke-run
scripts/publish_exp00_source_bundle.sh --dry-run
scripts/audit_launch_template.sh
```

`smoke-run` does not upload artifacts or shut the host down. Use
`scripts/sync_results.sh --dry-run` or the installed
`sudo /usr/local/sbin/ft-exp00-shutdown-and-sync --dry-run` helper to inspect
the final upload plan.

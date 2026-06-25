# function-calling-fine-tuning

Local preparation repo for xLAM-based function-calling fine-tuning on Qwen 3 with NVIDIA NeMo AutoModel.

## Pinned Components

- Repo commit: record the exact run commit with `git rev-parse HEAD`
- AutoModel container: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- AutoModel container digest: `sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2`
- Runtime-reported `nemo_automodel` version: `0.2.0rc0`
- Model: `Qwen/Qwen3-1.7B`
- Model revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Dataset: `Salesforce/xlam-function-calling-60k`
- Dataset revision: `26d14ebfe18b1f7b524bd39b404b50af5dc97866`
- Dataset license: `cc-by-4.0`
- Python dependencies: pinned in [requirements-dev.txt](/home/val/Documents/py-projects/function-calling-fine-tuning/requirements-dev.txt)

## Experiment 0 Model Switch

Experiment 0 now uses `Qwen/Qwen3-1.7B`. The original `Qwen/Qwen3-8B`
baseline achieved a strong result on the initial full 40-case function-calling
test, leaving limited headroom for observing measurable fine-tuning
improvement. `Qwen/Qwen3-1.7B` preserves the modern Qwen3 architecture while
providing more adaptation headroom and reducing GPU cost. This change does not
imply that fine-tuning is guaranteed to improve the smaller model; Experiment 0
still primarily validates the end-to-end training, checkpoint, reload, and
evaluation pipeline.

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
5. Build the optional 1,000-record stratified baseline benchmark:
   `./.venv/bin/python scripts/select_stratified_eval_sample.py`
6. Run the full local gate:
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

## Experiment Storage Policy

Before each EC2 experiment stage, publish the current clean Git commit to S3:

```bash
scripts/sync_source_to_s3.sh --execute
```

This builds a revisioned source archive, uploads it to
`s3://finetuning-lab-1-037678282394-us-west-2-an/finetuning/source-bundles/`,
and updates the convenience `exp00-source.tar.gz` alias by default.

Storage rules:

| Artifact group | Git | Workspace EBS | S3 |
| --- | --- | --- | --- |
| Source code, configs, scripts, tests | yes | yes | yes, as source bundle |
| Frozen smoke dataset | no | yes, copied at bootstrap | yes, canonical copy |
| Dataset manifests and checksums | yes | yes | yes |
| Results and predictions | no | yes | yes, before stopping EC2 |
| Logs | no | yes | yes, before stopping EC2 |
| Run-info and environment reports | no | yes | yes, before stopping EC2 |
| Base model and Hugging Face cache | no | yes | no |
| Docker and NGC caches | no | yes | no |
| Intermediate training checkpoints | no | yes | no, unless explicitly requested |
| Final LoRA/QLoRA adapter | no | yes | yes |

The frozen smoke dataset prefix is small, about 454 KiB
(`464592` bytes), so S3 remains the canonical source and bootstrap copies it to
the workspace EBS volume for each prepared host.

After a stage creates artifacts, sync them before stopping the instance. For the
C6 baseline subset:

```bash
scripts/sync_results.sh --stage baseline
```

For the 1,000-case stratified baseline benchmark:

```bash
scripts/sync_results.sh --stage baseline-1000
```

For a completed training and full-evaluation run:

```bash
scripts/sync_results.sh --stage final
```

The final-stage sync uploads the final adapter directory
`/mnt/workspace/checkpoints/exp-00/smoke-qlora` to S3. It does not upload base
LLM weights, Hugging Face caches, Docker layers, NGC caches, or intermediate
checkpoint directories.

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
make smoke-baseline-1000
make smoke-train
make smoke-reload-check
make smoke-evaluate
make smoke-run
scripts/run_automodel_container.sh --pull --login-ngc make smoke-run
scripts/resolve_exp00_config.py
scripts/sync_source_to_s3.sh --dry-run
scripts/publish_eval_dataset.sh --dry-run
scripts/publish_exp00_source_bundle.sh --dry-run
scripts/audit_launch_template.sh
```

`smoke-run` does not shut the host down. Use `scripts/sync_results.sh
--stage baseline --dry-run`, `scripts/sync_results.sh --stage final --dry-run`,
or the installed `sudo /usr/local/sbin/ft-exp00-shutdown-and-sync --dry-run`
helper to inspect the upload plan before stopping the instance.

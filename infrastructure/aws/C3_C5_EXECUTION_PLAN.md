# C3-C5 Host Execution Plan

This runbook is for the existing Experiment 0 EC2 instance:

- Instance ID: `i-0c769a18f50fd1fe6`
- Region: `us-west-2`
- Launch template: `ft-exp00-g6e-v1`, version `2`
- Project path on host: `/mnt/workspace/project`
- Pinned container: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- NGC SSM parameter: `/finetuning/ngc/api-key`
- Hugging Face SSM parameter: `/finetuning/huggingface/token`

The instance must be stopped at the end. Do not run model training in this
plan. Do not print or persist secret values.

## C4 Version Correction

- Previously expected AutoModel version: `0.3.0`
- Runtime-observed AutoModel version: `0.2.0rc0`
- Container tag: `nvcr.io/nvidia/nemo-automodel:25.11.00`
- Container digest: `sha256:c4f613005518d520c2ac3d9206d95617a2385f86cf8aa09582aad8d35957e2f2`
- CLI validation: passed
- Recipe target resolution: passed
- C4 disposition: pass with corrected version evidence

The earlier expectation that this image contained AutoModel `0.3.0` was
incorrect. The actual container was validated directly:

- `nemo_automodel` imports successfully
- the AutoModel CLI is available
- the Experiment 0 LoRA and QLoRA target classes resolve
- persistent checkpoint paths validate
- template and loss-mask checks pass

The container must not be changed solely to match the previously expected
package-version string. Runtime execution results are authoritative for
Experiment 0.

## C3 - Authenticate to NGC and Pull the Pinned Container

Run on the EC2 instance through SSM:

1. Export the SSM parameter names:
   - `NGC_API_KEY_SSM_PARAMETER=/finetuning/ngc/api-key`
   - `HF_TOKEN_SSM_PARAMETER=/finetuning/huggingface/token`
2. Retrieve the NGC key into a shell variable only through
   `scripts/run_automodel_container.sh --login-ngc --pull`.
3. Log in to `nvcr.io` with `docker login --password-stdin`.
4. Pull `nvcr.io/nvidia/nemo-automodel:25.11.00`.
5. Record the immutable image digest to
   `/mnt/workspace/run-info/container_image.txt`.
6. Log out of `nvcr.io`.
7. Unset the NGC key variable.

Acceptance:

- Pull exits 0.
- `container_image.txt` includes the image ref and a repo digest.
- No NGC key appears in command output.

## C4 - Verify AutoModel and Container Dependencies

Run a short Python inspection inside the pinned container using the persistent
mounts configured by `scripts/run_automodel_container.sh`.

Inspect and write a JSON report to:

`/mnt/workspace/run-info/c4_container_report.json`

Required checks:

- `nvidia-smi` works inside the container.
- PyTorch imports and reports a version.
- `torch.version.cuda` is recorded.
- `torch.cuda.is_available()` is true.
- GPU name and memory are recorded.
- `nemo_automodel` imports and version is recorded when available.
- `transformers`, `datasets`, `peft`, and `bitsandbytes` import and versions
  are recorded.
- `function_calling_ft` imports from `/workspace/project/src`.
- Persistent mount paths exist inside the container:
  - `/workspace/project`
  - `/workspace/data`
  - `/workspace/checkpoints`
  - `/workspace/results`
  - `/workspace/logs`
  - `/workspace/run-info`
- Smoke config validation confirms checkpoint paths are under
  `/workspace/checkpoints`.

Acceptance:

- Inspection command exits 0.
- Report exists on persistent storage.
- No dependency import required for training fails.

## C5 - Verify Prompt Template and Loss Mask

Run inside the pinned container:

1. Validate both smoke configs:
   - `configs/exp00_smoke/smoke_qlora.yaml`
   - `configs/exp00_smoke/smoke_lora.yaml`
2. Inspect five real rendered examples:
   - `scripts/inspect_template.py --normalized-dir /workspace/data --cache-dir /root/.cache/huggingface --count 5`
3. Inspect loss masks for five real smoke examples:
   - `scripts/inspect_loss_mask.py --normalized-dir /workspace/data --cache-dir /root/.cache/huggingface --smoke-count 5`

Acceptance:

- Tool definitions appear before user requests.
- Function names are unchanged.
- Arguments render as JSON objects.
- Python-style single-quoted dictionaries are absent.
- Thinking mode is disabled.
- Assistant tool-call tokens receive loss.
- User and tool-schema tokens are masked.
- Not all tokens are masked.
- No inspected example is unexpectedly truncated.

## Stop Policy

After C3-C5 pass or fail:

1. Stop `i-0c769a18f50fd1fe6`.
2. Wait until the instance state is `stopped`.
3. Confirm the public IP is released.
4. Report all C3, C4, and C5 step statuses.

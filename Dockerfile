FROM nvcr.io/nvidia/nemo-automodel:25.11.00

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    PYTHONPATH=/workspace/function-calling-fine-tuning/src

WORKDIR /workspace/function-calling-fine-tuning

COPY Makefile README.md pyproject.toml requirements-dev.txt ./
COPY configs ./configs
COPY data/manifests ./data/manifests
COPY data/processed/README.md ./data/processed/README.md
COPY data/raw/README.md ./data/raw/README.md
COPY data/smoke/README.md ./data/smoke/README.md
COPY scripts ./scripts
COPY src ./src
COPY tests ./tests

RUN mkdir -p \
    /workspace/.cache/huggingface \
    /workspace/data \
    /workspace/outputs \
    /workspace/results

RUN python3 -m pip install -r requirements-dev.txt && \
    python3 -m pip install -e .

CMD ["/bin/bash"]

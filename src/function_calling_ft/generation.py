from __future__ import annotations

import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class GenerationTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ...


@dataclass(frozen=True)
class GenerationPrompt:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    input_ids: tuple[int, ...]
    thinking_mode_supported: bool

    @property
    def prompt_token_count(self) -> int:
        return len(self.input_ids)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON",
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object",
                )
            records.append(record)

    return records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True),
            )
            file.write("\n")


def build_inference_messages(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError("Record is missing a messages list")

    inference_messages = [dict(message) for message in messages]

    if inference_messages:
        final_message = inference_messages[-1]
        if (
            final_message.get("role") == "assistant"
            and final_message.get("tool_calls") is not None
        ):
            inference_messages.pop()

    if not inference_messages:
        raise ValueError("No inference messages remain after target removal")

    return inference_messages


def _tokenizer_kwargs(
    tokenizer: GenerationTokenizer,
    *,
    enable_thinking: bool,
) -> tuple[dict[str, Any], bool]:
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supports_enable_thinking = (
        "enable_thinking" in signature.parameters or supports_kwargs
    )
    kwargs: dict[str, Any] = {}

    if supports_enable_thinking:
        kwargs["enable_thinking"] = enable_thinking

    return kwargs, supports_enable_thinking


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "get") and callable(value.get):
        maybe_input_ids = value.get("input_ids")
        if maybe_input_ids is not None:
            value = maybe_input_ids

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, list)
        and value
        and isinstance(value[0], list)
    ):
        value = value[0]

    return tuple(int(token_id) for token_id in value)


def build_generation_prompt(
    tokenizer: GenerationTokenizer,
    record: dict[str, Any],
    *,
    enable_thinking: bool = False,
) -> GenerationPrompt:
    messages = build_inference_messages(record)
    tools_value = record.get("tools", [])
    tools = tools_value if isinstance(tools_value, list) else []
    tokenizer_kwargs, thinking_supported = _tokenizer_kwargs(
        tokenizer,
        enable_thinking=enable_thinking,
    )
    tokenized = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        **tokenizer_kwargs,
    )

    return GenerationPrompt(
        messages=messages,
        tools=tools,
        input_ids=normalize_token_ids(tokenized),
        thinking_mode_supported=thinking_supported,
    )


def validate_adapter_path(adapter_path: Path) -> Path:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    candidates = [
        path.parent
        for path in adapter_path.rglob("adapter_config.json")
        if path.is_file()
    ]

    if adapter_path.is_file():
        raise ValueError(f"Adapter path must be a directory: {adapter_path}")

    if (adapter_path / "adapter_config.json").is_file():
        candidates = [adapter_path]

    unique_candidates = sorted(set(candidates))

    if not unique_candidates:
        raise FileNotFoundError(
            "No adapter_config.json found under adapter path: "
            f"{adapter_path}",
        )

    if len(unique_candidates) > 1:
        joined = ", ".join(str(path) for path in unique_candidates)
        raise ValueError(
            "Adapter path contains multiple adapter_config.json files: "
            f"{joined}",
        )

    resolved = unique_candidates[0]
    weight_files = [
        *resolved.glob("adapter_model*.safetensors"),
        *resolved.glob("adapter_model*.bin"),
    ]

    if not weight_files:
        raise FileNotFoundError(
            "Adapter directory is missing adapter_model weights: "
            f"{resolved}",
        )

    return resolved


def validate_adapter_base_model(
    adapter_path: Path,
    expected_model_name: str,
) -> Path:
    resolved = validate_adapter_path(adapter_path)
    config_path = resolved / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Adapter config is not valid JSON: {config_path}",
        ) from exc

    actual_model_name = config.get("base_model_name_or_path")
    if actual_model_name and actual_model_name != expected_model_name:
        raise ValueError(
            "Adapter base_model_name_or_path does not match expected model: "
            f"{actual_model_name!r} != {expected_model_name!r}",
        )

    return resolved


def set_generation_seed(seed: int) -> None:
    random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        from transformers import set_seed
    except ModuleNotFoundError:
        return

    set_seed(seed)


def _device_for_model(model: Any, requested_device: str | None) -> Any:
    if requested_device:
        return requested_device

    device = getattr(model, "device", None)
    if device is not None:
        return device

    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return "cpu"

    return getattr(first_parameter, "device", "cpu")


def _generate_one(
    *,
    tokenizer: GenerationTokenizer,
    model: Any,
    prompt: GenerationPrompt,
    max_new_tokens: int,
    device: str | None,
) -> tuple[str, int]:
    import torch

    model_device = _device_for_model(model, device)
    input_ids = torch.tensor(
        [list(prompt.input_ids)],
        dtype=torch.long,
        device=model_device,
    )
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None:
        pad_token_id = eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    token_ids = normalize_token_ids(output_ids)
    generated_ids = token_ids[len(prompt.input_ids) :]
    raw_generation = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return raw_generation, len(generated_ids)


def generate_prediction_records(
    *,
    records: Sequence[dict[str, Any]],
    tokenizer: GenerationTokenizer,
    model: Any,
    model_name: str,
    model_revision: str,
    adapter_path: str | None,
    seed: int,
    max_new_tokens: int,
    device: str | None = None,
) -> list[dict[str, Any]]:
    set_generation_seed(seed)
    predictions: list[dict[str, Any]] = []

    for record in records:
        record_id = str(record.get("id", ""))
        metadata = record.get("metadata")
        source_id = (
            metadata.get("source_id")
            if isinstance(metadata, dict)
            else None
        )
        base_record = {
            "id": record_id,
            "source_id": source_id,
            "model_name": model_name,
            "model_revision": model_revision,
            "adapter_path": adapter_path,
            "raw_generation": "",
            "prompt_token_count": 0,
            "generated_token_count": 0,
            "generation_error": None,
        }

        try:
            prompt = build_generation_prompt(
                tokenizer,
                record,
                enable_thinking=False,
            )
            raw_generation, generated_count = _generate_one(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            predictions.append(
                {
                    **base_record,
                    "raw_generation": raw_generation,
                    "prompt_token_count": prompt.prompt_token_count,
                    "generated_token_count": generated_count,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-record errors.
            predictions.append(
                {
                    **base_record,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                }
            )

    return predictions


def load_transformers_model(
    *,
    model_name: str,
    model_revision: str,
    adapter_path: Path | None,
    cache_dir: Path | None,
    load_in_4bit: bool,
    torch_dtype: str,
) -> tuple[Any, Any, str | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=False,
    )
    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": False,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "4-bit loading requires transformers BitsAndBytesConfig",
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    resolved_adapter_path: str | None = None

    if adapter_path is not None:
        resolved_adapter = validate_adapter_base_model(
            adapter_path,
            model_name,
        )
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Adapter reload requires the peft package",
            ) from exc
        model = PeftModel.from_pretrained(model, str(resolved_adapter))
        resolved_adapter_path = str(resolved_adapter)

    model.eval()
    return tokenizer, model, resolved_adapter_path

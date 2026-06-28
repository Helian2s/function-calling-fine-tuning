from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024 / 1024 / 1024, 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _CudaMemoryTrace:
    def __init__(self, output: Path, *, max_events: int) -> None:
        self.output = output
        self.max_events = max_events
        self.events: list[dict[str, Any]] = []
        self.summary: dict[str, dict[str, Any]] = {}

    def _memory(self) -> dict[str, int | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"allocated_bytes": None, "reserved_bytes": None}
            return {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }
        except Exception:
            return {"allocated_bytes": None, "reserved_bytes": None}

    def record(
        self,
        *,
        phase: str,
        name: str,
        started_at: float,
        before: dict[str, int | None],
    ) -> None:
        after = self._memory()
        ended_at = time.monotonic()
        allocated_before = before.get("allocated_bytes")
        reserved_before = before.get("reserved_bytes")
        allocated_after = after.get("allocated_bytes")
        reserved_after = after.get("reserved_bytes")
        allocated_delta = (
            allocated_after - allocated_before
            if allocated_after is not None and allocated_before is not None
            else None
        )
        reserved_delta = (
            reserved_after - reserved_before
            if reserved_after is not None and reserved_before is not None
            else None
        )
        key = f"{phase}:{name}"
        aggregate = self.summary.setdefault(
            key,
            {
                "phase": phase,
                "name": name,
                "count": 0,
                "max_allocated_after_bytes": None,
                "max_reserved_after_bytes": None,
                "max_allocated_delta_bytes": None,
                "max_reserved_delta_bytes": None,
                "total_duration_seconds": 0.0,
            },
        )
        aggregate["count"] += 1
        aggregate["total_duration_seconds"] += ended_at - started_at
        for field, value in (
            ("max_allocated_after_bytes", allocated_after),
            ("max_reserved_after_bytes", reserved_after),
            ("max_allocated_delta_bytes", allocated_delta),
            ("max_reserved_delta_bytes", reserved_delta),
        ):
            if value is None:
                continue
            previous = aggregate[field]
            aggregate[field] = value if previous is None else max(previous, value)

        if len(self.events) < self.max_events:
            self.events.append(
                {
                    "phase": phase,
                    "name": name,
                    "duration_seconds": ended_at - started_at,
                    "allocated_before_bytes": allocated_before,
                    "allocated_after_bytes": allocated_after,
                    "allocated_delta_bytes": allocated_delta,
                    "reserved_before_bytes": reserved_before,
                    "reserved_after_bytes": reserved_after,
                    "reserved_delta_bytes": reserved_delta,
                },
            )

    def write(self) -> None:
        payload = {
            "schema_version": "1.0",
            "pid": os.getpid(),
            "argv": sys.argv,
            "max_raw_events": self.max_events,
            "raw_event_count": len(self.events),
            "events": self.events,
            "summary": dict(sorted(self.summary.items())),
        }
        _write_json(self.output, payload)


def _install_cuda_memory_trace() -> _CudaMemoryTrace | None:
    output = os.environ.get("FCFT_TORCH_MEMORY_TRACE_OUTPUT")
    if not output:
        return None

    try:
        import torch
    except Exception:
        return None

    trace = _CudaMemoryTrace(
        Path(output),
        max_events=int(os.environ.get("FCFT_TORCH_MEMORY_TRACE_MAX_EVENTS", "5000")),
    )

    original_call_impl = torch.nn.Module._call_impl

    def traced_call_impl(self: Any, *args: Any, **kwargs: Any) -> Any:
        name = type(self).__name__
        should_trace = name in {
            "Linear",
            "Embedding",
            "Qwen3Attention",
            "Qwen3DecoderLayer",
            "Qwen3MLP",
        } or "Lora" in name or "LoRA" in name
        if not should_trace:
            return original_call_impl(self, *args, **kwargs)
        before = trace._memory()
        started = time.monotonic()
        try:
            return original_call_impl(self, *args, **kwargs)
        finally:
            trace.record(
                phase="forward_module",
                name=name,
                started_at=started,
                before=before,
            )

    torch.nn.Module._call_impl = traced_call_impl

    original_autograd_backward = torch.autograd.backward

    def traced_autograd_backward(*args: Any, **kwargs: Any) -> Any:
        before = trace._memory()
        started = time.monotonic()
        try:
            return original_autograd_backward(*args, **kwargs)
        finally:
            trace.record(
                phase="backward",
                name="torch.autograd.backward",
                started_at=started,
                before=before,
            )

    torch.autograd.backward = traced_autograd_backward

    original_tensor_backward = torch.Tensor.backward

    def traced_tensor_backward(self: Any, *args: Any, **kwargs: Any) -> Any:
        before = trace._memory()
        started = time.monotonic()
        try:
            return original_tensor_backward(self, *args, **kwargs)
        finally:
            trace.record(
                phase="backward",
                name="torch.Tensor.backward",
                started_at=started,
                before=before,
            )

    torch.Tensor.backward = traced_tensor_backward

    original_optimizer_step = torch.optim.Optimizer.step

    def traced_optimizer_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        before = trace._memory()
        started = time.monotonic()
        try:
            return original_optimizer_step(self, *args, **kwargs)
        finally:
            trace.record(
                phase="optimizer_step",
                name=type(self).__name__,
                started_at=started,
                before=before,
            )

    torch.optim.Optimizer.step = traced_optimizer_step

    original_torch_save = torch.save

    def traced_torch_save(*args: Any, **kwargs: Any) -> Any:
        before = trace._memory()
        started = time.monotonic()
        try:
            return original_torch_save(*args, **kwargs)
        finally:
            trace.record(
                phase="checkpoint_save",
                name="torch.save",
                started_at=started,
                before=before,
            )

    torch.save = traced_torch_save

    safetensors_torch: Any | None
    try:
        import safetensors.torch as safetensors_torch
    except Exception:
        safetensors_torch = None
    if safetensors_torch is not None:
        original_save_file = safetensors_torch.save_file

        def traced_save_file(*args: Any, **kwargs: Any) -> Any:
            before = trace._memory()
            started = time.monotonic()
            try:
                return original_save_file(*args, **kwargs)
            finally:
                trace.record(
                    phase="checkpoint_save",
                    name="safetensors.torch.save_file",
                    started_at=started,
                    before=before,
                )

        safetensors_torch.save_file = traced_save_file

    atexit.register(trace.write)
    return trace


def _install_qlora_peft_state_dict_patch() -> None:
    if os.environ.get("FCFT_PATCH_QLORA_PEFT_STATE_DICT") != "1":
        return

    try:
        import torch
        from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
    except Exception:
        return

    report_path_text = os.environ.get("FCFT_QLORA_PEFT_STATE_DICT_PATCH_REPORT")
    report_path = Path(report_path_text) if report_path_text else None
    original_state_dict = ModelState.state_dict

    def has_4bit_state(model_parts: list[Any]) -> bool:
        for model in model_parts:
            for module in model.modules():
                if type(module).__name__ == "Linear4bit":
                    return True
            for _, parameter in model.named_parameters(remove_duplicate=False):
                if type(parameter).__name__ == "Params4bit":
                    return True
        return False

    def adapter_only_state_dict(self: Any) -> dict[str, Any]:
        if (
            not getattr(self, "is_peft", False)
            or getattr(self, "is_init_step", False)
            or not has_4bit_state(self.model)
        ):
            return original_state_dict(self)

        state_dict: dict[str, torch.Tensor] = {}
        for model in self.model:
            for name, parameter in model.named_parameters(remove_duplicate=False):
                if ".lora_A." not in name and ".lora_B." not in name:
                    continue
                key = name
                if not key.startswith("base_model.model."):
                    key = "base_model.model." + key
                state_dict[key] = parameter.detach().cpu().contiguous()

        if not state_dict:
            return original_state_dict(self)

        if report_path is not None:
            _write_json(
                report_path,
                {
                    "schema_version": "1.0",
                    "enabled": True,
                    "reason": "nf4_qlora_peft_adapter_only_checkpoint",
                    "tensor_count": len(state_dict),
                    "keys_preview": sorted(state_dict)[:8],
                    "total_numel": sum(int(tensor.numel()) for tensor in state_dict.values()),
                },
            )
        return state_dict

    ModelState.state_dict = adapter_only_state_dict


def _write_torch_cuda_memory_report() -> None:
    output = os.environ.get("FCFT_TORCH_MEMORY_OUTPUT")
    if not output:
        return

    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "argv": sys.argv,
        "cuda_available": False,
        "devices": [],
        "peak_allocated_vram_bytes": None,
        "peak_reserved_vram_bytes": None,
        "peak_allocated_vram_gb": None,
        "peak_reserved_vram_gb": None,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on target env.
        payload["error"] = f"torch import failed: {exc!r}"
        _write_json(Path(output), payload)
        return

    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
        if payload["cuda_available"]:
            try:
                torch.cuda.synchronize()
            except Exception as exc:  # pragma: no cover - hardware-dependent.
                payload["synchronize_error"] = repr(exc)

            devices: list[dict[str, Any]] = []
            for index in range(torch.cuda.device_count()):
                try:
                    allocated = int(torch.cuda.max_memory_allocated(index))
                    reserved = int(torch.cuda.max_memory_reserved(index))
                    device = {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "peak_allocated_vram_bytes": allocated,
                        "peak_reserved_vram_bytes": reserved,
                        "peak_allocated_vram_gb": _gb(allocated),
                        "peak_reserved_vram_gb": _gb(reserved),
                    }
                except Exception as exc:  # pragma: no cover - hardware-dependent.
                    device = {"index": index, "error": repr(exc)}
                devices.append(device)

            payload["devices"] = devices
            allocated_values = [
                device["peak_allocated_vram_bytes"]
                for device in devices
                if isinstance(device.get("peak_allocated_vram_bytes"), int)
            ]
            reserved_values = [
                device["peak_reserved_vram_bytes"]
                for device in devices
                if isinstance(device.get("peak_reserved_vram_bytes"), int)
            ]
            peak_allocated = max(allocated_values) if allocated_values else None
            peak_reserved = max(reserved_values) if reserved_values else None
            payload["peak_allocated_vram_bytes"] = peak_allocated
            payload["peak_reserved_vram_bytes"] = peak_reserved
            payload["peak_allocated_vram_gb"] = _gb(peak_allocated)
            payload["peak_reserved_vram_gb"] = _gb(peak_reserved)
    except Exception as exc:  # pragma: no cover - last-resort telemetry guard.
        payload["error"] = repr(exc)

    _write_json(Path(output), payload)


atexit.register(_write_torch_cuda_memory_report)
_install_qlora_peft_state_dict_patch()
_install_cuda_memory_trace()

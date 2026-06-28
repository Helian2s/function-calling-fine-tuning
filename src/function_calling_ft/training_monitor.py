from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
LOSS_PATTERN = re.compile(
    rf"\b(?:loss|train_loss|training_loss)\b\s*(?::|=)?\s*"
    rf"(?P<value>nan|inf|-inf|{NUMBER_PATTERN})",
    re.IGNORECASE,
)
LR_PATTERN = re.compile(
    rf"\b(?:lr|learning_rate)\b\s*(?::|=)?\s*(?P<value>{NUMBER_PATTERN})",
    re.IGNORECASE,
)
STEP_PATTERN = re.compile(
    r"\b(?:step|global_step|iteration)\b\s*(?::|=)?\s*(?P<value>\d+)",
    re.IGNORECASE,
)
STEP_TIME_PATTERN = re.compile(
    rf"\b(?:step_time|step time|time/step|iter_time|iteration_time)\b"
    rf"\s*[=:]\s*(?P<value>{NUMBER_PATTERN})\s*(?:s|sec|seconds)?\b",
    re.IGNORECASE,
)
GRAD_NORM_PATTERN = re.compile(
    rf"\bgrad_norm\b\s*(?::|=)?\s*(?P<value>{NUMBER_PATTERN})",
    re.IGNORECASE,
)
TRAINABLE_PARAMS_PATTERN = re.compile(
    r"\btrainable(?:_|\s+)?(?:params?|parameters)\b"
    r"(?!\s+percentage)[^0-9]*(?P<value>[0-9][0-9,]*)",
    re.IGNORECASE,
)
TOTAL_PARAMS_PATTERN = re.compile(
    r"\b(?:all|total)(?:_|\s+)?(?:params?|parameters)\b[^0-9]*(?P<value>[0-9][0-9,]*)",
    re.IGNORECASE,
)
FROZEN_PARAMS_PATTERN = re.compile(
    r"\bfrozen(?:_|\s+)?(?:params?|parameters)\b[^0-9]*(?P<value>[0-9][0-9,]*)",
    re.IGNORECASE,
)
OOM_PATTERN = re.compile(
    r"(cuda.*out of memory|outofmemory|out of memory|cuda oom)",
    re.IGNORECASE,
)
TIMESTAMP_PATTERN = re.compile(
    r"^(?P<value>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
)


def _parse_number(value: str) -> float:
    lowered = value.lower()
    if lowered == "nan":
        return math.nan
    if lowered == "inf":
        return math.inf
    if lowered == "-inf":
        return -math.inf
    return float(value)


def _parse_int(value: str) -> int:
    return int(value.replace(",", ""))


def _parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_PATTERN.search(line)
    if not match:
        return None
    return datetime.strptime(match.group("value"), "%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class ParsedTrainingLine:
    losses: tuple[float, ...] = ()
    learning_rates: tuple[float, ...] = ()
    steps: tuple[int, ...] = ()
    step_times_seconds: tuple[float, ...] = ()
    grad_norms: tuple[float, ...] = ()
    trainable_parameter_count: int | None = None
    total_parameter_count: int | None = None
    frozen_parameter_count: int | None = None
    oom_event: bool = False


def parse_training_line(line: str) -> ParsedTrainingLine:
    losses = tuple(_parse_number(match.group("value")) for match in LOSS_PATTERN.finditer(line))
    learning_rates = tuple(_parse_number(match.group("value")) for match in LR_PATTERN.finditer(line))
    steps = tuple(_parse_int(match.group("value")) for match in STEP_PATTERN.finditer(line))
    step_times = tuple(
        _parse_number(match.group("value")) for match in STEP_TIME_PATTERN.finditer(line)
    )
    grad_norms = tuple(
        _parse_number(match.group("value")) for match in GRAD_NORM_PATTERN.finditer(line)
    )

    trainable_match = TRAINABLE_PARAMS_PATTERN.search(line)
    total_match = TOTAL_PARAMS_PATTERN.search(line)
    frozen_match = FROZEN_PARAMS_PATTERN.search(line)

    return ParsedTrainingLine(
        losses=losses,
        learning_rates=learning_rates,
        steps=steps,
        step_times_seconds=step_times,
        grad_norms=grad_norms,
        trainable_parameter_count=(
            _parse_int(trainable_match.group("value")) if trainable_match else None
        ),
        total_parameter_count=_parse_int(total_match.group("value")) if total_match else None,
        frozen_parameter_count=(
            _parse_int(frozen_match.group("value")) if frozen_match else None
        ),
        oom_event=OOM_PATTERN.search(line) is not None,
    )


def summarize_training_signals(
    *,
    losses: list[float],
    learning_rates: list[float],
    steps: list[int],
    step_times_seconds: list[float],
) -> dict[str, Any]:
    finite_losses = [loss for loss in losses if math.isfinite(loss)]
    finite_step_times = [value for value in step_times_seconds if math.isfinite(value)]

    return {
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(finite_losses) if finite_losses else None,
        "max_loss": max(finite_losses) if finite_losses else None,
        "loss_history": losses,
        "losses_are_finite": bool(losses) and len(finite_losses) == len(losses),
        "learning_rate_history": learning_rates,
        "initial_learning_rate": learning_rates[0] if learning_rates else None,
        "final_learning_rate": learning_rates[-1] if learning_rates else None,
        "step_history": steps,
        "global_step": max(steps) if steps else None,
        "step_time_seconds_history": step_times_seconds,
        "average_step_time_seconds": (
            sum(finite_step_times) / len(finite_step_times) if finite_step_times else None
        ),
    }


def parse_existing_training_log(path: Path) -> dict[str, Any]:
    losses: list[float] = []
    validation_losses: list[float] = []
    learning_rates: list[float] = []
    steps: list[int] = []
    step_times_seconds: list[float] = []
    grad_norms: list[float] = []
    last_step_timestamp: datetime | None = None
    trainable_parameter_count: int | None = None
    total_parameter_count: int | None = None
    frozen_parameter_count: int | None = None
    oom_event_count = 0

    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = parse_training_line(line)
            is_validation_line = "[val]" in line
            if is_validation_line:
                validation_losses.extend(parsed.losses)
            else:
                losses.extend(parsed.losses)
                learning_rates.extend(parsed.learning_rates)
                steps.extend(parsed.steps)
                grad_norms.extend(parsed.grad_norms)
                if parsed.steps:
                    timestamp = _parse_timestamp(line)
                    if timestamp is not None:
                        if last_step_timestamp is not None:
                            step_times_seconds.append(
                                (timestamp - last_step_timestamp).total_seconds(),
                            )
                        last_step_timestamp = timestamp
            step_times_seconds.extend(parsed.step_times_seconds)
            trainable_parameter_count = (
                parsed.trainable_parameter_count
                if parsed.trainable_parameter_count is not None
                else trainable_parameter_count
            )
            total_parameter_count = (
                parsed.total_parameter_count
                if parsed.total_parameter_count is not None
                else total_parameter_count
            )
            frozen_parameter_count = (
                parsed.frozen_parameter_count
                if parsed.frozen_parameter_count is not None
                else frozen_parameter_count
            )
            if parsed.oom_event:
                oom_event_count += 1

    if frozen_parameter_count is None and (
        total_parameter_count is not None
        and trainable_parameter_count is not None
    ):
        frozen_parameter_count = total_parameter_count - trainable_parameter_count

    trainable_parameter_ratio = (
        trainable_parameter_count / total_parameter_count
        if trainable_parameter_count is not None and total_parameter_count
        else None
    )
    nonzero_grad_norm_count = sum(
        1 for value in grad_norms if math.isfinite(value) and value > 0
    )
    adapter_gradient_status = (
        "observed_nonzero_grad_norm"
        if nonzero_grad_norm_count > 0
        else "not_observed"
    )
    base_model_trainability_status = (
        "frozen_by_ratio"
        if trainable_parameter_ratio is not None and trainable_parameter_ratio <= 0.10
        else "unknown"
    )

    return {
        **summarize_training_signals(
            losses=losses,
            learning_rates=learning_rates,
            steps=steps,
            step_times_seconds=step_times_seconds,
        ),
        "validation_loss_history": validation_losses,
        "initial_validation_loss": validation_losses[0] if validation_losses else None,
        "final_validation_loss": validation_losses[-1] if validation_losses else None,
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "trainable_parameter_ratio": trainable_parameter_ratio,
        "grad_norm_history": grad_norms,
        "nonzero_grad_norm_count": nonzero_grad_norm_count,
        "adapter_gradient_status": adapter_gradient_status,
        "base_model_trainability_status": base_model_trainability_status,
        "oom_event_count": oom_event_count,
    }

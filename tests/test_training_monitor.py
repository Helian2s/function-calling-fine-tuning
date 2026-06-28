from __future__ import annotations

import math
import json
from pathlib import Path

from scripts.run_training_with_monitor import (
    SITE_CUSTOMIZE_DIR,
    child_environment,
    read_torch_memory_summary,
)
from function_calling_ft.training_monitor import (
    parse_existing_training_log,
    parse_training_line,
    summarize_training_signals,
)


def test_parse_training_line_extracts_loss_lr_step_and_params() -> None:
    parsed = parse_training_line(
        "step=7 loss=1.25 lr=1.0e-4 step_time=2.5s "
        "grad_norm=12.5 trainable params: 12,345 || all params: 1,000,000",
    )

    assert parsed.losses == (1.25,)
    assert parsed.learning_rates == (1.0e-4,)
    assert parsed.steps == (7,)
    assert parsed.step_times_seconds == (2.5,)
    assert parsed.grad_norms == (12.5,)
    assert parsed.trainable_parameter_count == 12345
    assert parsed.total_parameter_count == 1000000


def test_parse_training_line_extracts_automodel_parameter_wording() -> None:
    trainable = parse_training_line("Trainable parameters: 17,432,576")
    total = parse_training_line("Total parameters: 2,049,172,480")

    assert trainable.trainable_parameter_count == 17432576
    assert total.total_parameter_count == 2049172480


def test_parse_training_line_ignores_trainable_percentage_as_count() -> None:
    parsed = parse_training_line("Trainable parameters percentage: 0.85%")

    assert parsed.trainable_parameter_count is None


def test_parse_training_line_detects_non_finite_loss_and_oom() -> None:
    parsed = parse_training_line("step=2 train_loss=NaN CUDA out of memory")

    assert len(parsed.losses) == 1
    assert math.isnan(parsed.losses[0])
    assert parsed.oom_event


def test_parse_training_log_extracts_automodel_step_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "training.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-26 03:59:11 | INFO | root | Trainable parameters: 17,432,576",
                "2026-06-26 03:59:11 | INFO | root | Total parameters: 2,049,172,480",
                "2026-06-26 03:59:21 | INFO | root | step 0 | epoch 0 | loss 1.0030 | grad_norm 199.0 | lr 4.00e-05",
                "2026-06-26 03:59:22 | INFO | root | step 1 | epoch 0 | loss 0.5750 | grad_norm 190.0 | lr 7.00e-05",
                "2026-06-26 03:59:32 | INFO | root | [val] name \"default\" | step 9 | epoch 0 | loss 1.0574 | lr 8.45e-05",
            ],
        ),
        encoding="utf-8",
    )

    summary = parse_existing_training_log(log_path)

    assert summary["loss_history"] == [1.003, 0.575]
    assert summary["validation_loss_history"] == [1.0574]
    assert summary["learning_rate_history"] == [4.0e-5, 7.0e-5]
    assert summary["step_history"] == [0, 1]
    assert summary["step_time_seconds_history"] == [1.0]
    assert summary["trainable_parameter_count"] == 17432576
    assert summary["total_parameter_count"] == 2049172480
    assert summary["frozen_parameter_count"] == 2031739904
    assert summary["trainable_parameter_ratio"] == 17432576 / 2049172480
    assert summary["grad_norm_history"] == [199.0, 190.0]
    assert summary["nonzero_grad_norm_count"] == 2
    assert summary["adapter_gradient_status"] == "observed_nonzero_grad_norm"
    assert summary["base_model_trainability_status"] == "frozen_by_ratio"


def test_summarize_training_signals_reports_finite_loss_status() -> None:
    summary = summarize_training_signals(
        losses=[2.0, 1.5],
        learning_rates=[1.0e-4, 9.0e-5],
        steps=[1, 2],
        step_times_seconds=[3.0, 5.0],
    )

    assert summary["initial_loss"] == 2.0
    assert summary["final_loss"] == 1.5
    assert summary["losses_are_finite"] is True
    assert summary["global_step"] == 2
    assert summary["average_step_time_seconds"] == 4.0


def test_child_environment_injects_torch_memory_probe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "training_torch_memory.json"

    env = child_environment(output)

    assert env["FCFT_TORCH_MEMORY_OUTPUT"] == str(output)
    assert env["PYTHONPATH"].split(":")[0] == str(SITE_CUSTOMIZE_DIR)


def test_read_torch_memory_summary_reports_peaks(tmp_path: Path) -> None:
    report_path = tmp_path / "training_torch_memory.json"
    report_path.write_text(
        json.dumps(
            {
                "peak_allocated_vram_gb": 11.25,
                "peak_reserved_vram_gb": 12.5,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = read_torch_memory_summary(report_path)

    assert summary["torch_memory_probe_present"] is True
    assert summary["peak_allocated_vram_gb"] == 11.25
    assert summary["peak_reserved_vram_gb"] == 12.5

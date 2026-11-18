"""计算验证入口的门禁行为（§9、§12、TP-19）。

真正的 forward/backward 需要 FFmpeg 解码与真实权重，放在镜像内执行；这里钉住
那些**不能靠"跑起来"发现**的规则：CUDA 不可用必须 BLOCKED、步数必须为正、
设备字符串不能被静默改写。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit, pytest.mark.no_db]

from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.training import EXIT_BLOCKED, EXIT_DATA, exit_code_for  # noqa: E402
from embodied_dataops.training.validate import (  # noqa: E402
    _Checks,
    _assess_level,
    _resolve_device,
    run_act_validation,
)


def _passing_checks() -> _Checks:
    """8 项局部检查全过——旧逻辑到这一步就直接给 requested_level 了。"""
    checks = _Checks()
    for name in (
        "processor_stats_frozen",
        "reader_built",
        "policy_built",
        "loss_finite",
        "gradients_present",
        "gradients_finite",
        "optimizer_step",
        "checkpoint_roundtrip",
    ):
        checks.record(name, True)
    return checks


def _complete_machine_record() -> dict:
    return {
        "device": {
            "name": "NVIDIA A10",
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "driver": "550.90.07",
            "os": "Linux-6.x-x86_64",
            "cuda_version": "12.8",
            "capability": "8.6",
            "precision": "fp32",
        },
        "command": "embodied-dataops training-validate --device cuda:0 --steps 10",
    }


def test_a_non_positive_step_budget_is_refused(tmp_path):
    with pytest.raises(ApiError) as excinfo:
        run_act_validation(
            bundle=tmp_path / "bundle",
            source_root=tmp_path / "source",
            device="cpu",
            steps=0,
            report_dir=tmp_path / "reports",
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_cuda_request_without_cuda_is_blocked_not_downgraded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ApiError) as excinfo:
        _resolve_device("cuda:0")
    assert excinfo.value.code is ErrorCode.CUDA_UNAVAILABLE
    assert exit_code_for(excinfo.value) == EXIT_BLOCKED


def test_a_missing_cuda_index_is_blocked(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ApiError) as excinfo:
        _resolve_device("cuda:3")
    assert excinfo.value.code is ErrorCode.CUDA_UNAVAILABLE


def test_an_unsupported_device_type_is_a_contract_error():
    with pytest.raises(ApiError) as excinfo:
        _resolve_device("mps")
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert exit_code_for(excinfo.value) == 2


def test_a_malformed_device_string_is_a_contract_error():
    with pytest.raises(ApiError) as excinfo:
        _resolve_device("cuda:not-a-number")
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_cpu_is_accepted_unchanged():
    device = _resolve_device("cpu")
    assert device.type == "cpu"


def test_missing_decode_dependencies_are_blocked_not_failed():
    assert exit_code_for(ErrorCode.DEPENDENCY_UNAVAILABLE) == EXIT_BLOCKED


def test_data_defects_map_to_exit_code_4():
    for code in (
        ErrorCode.EMPTY_REQUIRED_WINDOWS,
        ErrorCode.EMPTY_TRAIN_WINDOWS,
        ErrorCode.NONFINITE_INPUT,
        ErrorCode.FILE_DECODE_ERROR,
        ErrorCode.SOURCE_HASH_MISMATCH,
    ):
        assert exit_code_for(code) == EXIT_DATA, code


def test_model_numerics_map_to_exit_code_5():
    for code in (ErrorCode.MODEL_NUMERICS_FAILED, ErrorCode.PROCESSOR_STATS_MISMATCH):
        assert exit_code_for(code) == 5, code


def test_eight_passing_checks_alone_do_not_yield_gpu_verified():
    """这正是 f01-gpu 报告的问题：8/8 True 被写成了 GPU_VERIFIED。"""
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=_passing_checks(),
        report={"device": {"name": "NVIDIA A10", "uuid": None, "driver": "unknown"}},
        val_eval=None,
        steps=10,
        chunk_size=4,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved is None
    assert assessment.level_status == "PARTIAL"
    joined = " | ".join(assessment.blocking)
    assert "val-split eval" in joined
    assert "uuid" in joined
    assert "driver" in joined
    assert "command" in joined
    assert "os" in joined


def test_complete_evidence_yields_gpu_verified_and_chunk_deviation_becomes_a_note():
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=_passing_checks(),
        report=_complete_machine_record(),
        val_eval={"split": "val", "windows": 49, "finite": True},
        steps=10,
        chunk_size=4,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved == "GPU_VERIFIED"
    assert assessment.level_status == "VERIFIED"
    assert assessment.blocking == ()
    # §13 允许调整 chunk，但必须留下可追溯的说明，不能静默替换首候选。
    assert any("chunk_size=4" in note for note in assessment.notes)


def test_a_short_gpu_run_cannot_claim_gpu_verified():
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=_passing_checks(),
        report=_complete_machine_record(),
        val_eval={"split": "val", "windows": 49, "finite": True},
        steps=3,
        chunk_size=16,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved is None
    assert assessment.level_status == "PARTIAL"
    assert any("optimizer steps" in item for item in assessment.blocking)


def test_a_non_finite_val_eval_blocks_the_level():
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=_passing_checks(),
        report=_complete_machine_record(),
        val_eval={"split": "val", "windows": 49, "finite": False},
        steps=10,
        chunk_size=16,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved is None
    assert any("non-finite" in item for item in assessment.blocking)


def test_a_missing_recipe_digest_blocks_the_level():
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=_passing_checks(),
        report=_complete_machine_record(),
        val_eval={"split": "val", "windows": 49, "finite": True},
        steps=10,
        chunk_size=16,
        recipe_digest=None,
    )
    assert assessment.achieved is None
    assert any("recipe digest" in item for item in assessment.blocking)


def test_cpu_level_keeps_its_existing_meaning():
    assessment = _assess_level(
        requested_level="CPU_VERIFIED",
        checks=_passing_checks(),
        report={"device": {"requested": "cpu", "cuda_available": False}},
        val_eval=None,
        steps=1,
        chunk_size=4,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved == "CPU_VERIFIED"
    assert assessment.level_status == "VERIFIED"
    assert any("val split" in note for note in assessment.notes)


def test_failed_checks_are_reported_as_failed_not_partial():
    checks = _passing_checks()
    checks.fail("loss_finite")
    assessment = _assess_level(
        requested_level="GPU_VERIFIED",
        checks=checks,
        report=_complete_machine_record(),
        val_eval={"split": "val", "windows": 49, "finite": True},
        steps=10,
        chunk_size=16,
        recipe_digest="sha256:abc",
    )
    assert assessment.achieved is None
    assert assessment.level_status == "FAILED"

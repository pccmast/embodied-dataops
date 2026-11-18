"""指定模型的真实计算验证（EDO-TRAIN-PREP-1.1 §9、§13）。

``run_act_validation`` 是首期**唯一**受支持的项目训练验证入口。它拒绝走
``lerobot-train --dataset.repo_id=...``：那条路会绕过 split、有效窗口和冻结统计量，
最后得到的"训练成功"证明不了本合同里的任何一条。

验证顺序固定且不可跳过：

1. 重验 bundle 的文件摘要与 ``artifact_id``（发布后可能已被篡改，TP-16）；
2. 逐文件重验 ``--source-root``（换一个同路径的目录同样要被抓到）；
3. 按 split 建受限 adapter，DataLoader 只能看见本 split 的冻结窗口（TP-11）；
4. 从冻结配置造 ACTConfig / ACTPolicy，加载冻结 processor 并断言 stats 未漂移（TP-12）；
5. 真实 forward / backward / optimizer.step，检查 loss 与**实际产生的**梯度有限（TP-14）；
6. checkpoint 与 processor 保存后重载，eval 输出在固定容差内一致（TP-15）；
7. 按 §13 判定等级：``status`` 只说明本次局部检查是否全过，``achieved_level`` 还要求
   满足 §13 的通过清单——val-split eval、完整机器记录、步数与配方。二者分离是刻意的：
   8 项检查全过不等于 A10 通过，把它写成 ``GPU_VERIFIED`` 就是在制造一个不存在的证据。

CPU 不可用或设备不可用一律 **BLOCKED**，绝不静默回退——把 CUDA 失败当 CPU 通过
是在制造一个不存在的证据。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.artifact import (
    LoadedBundle,
    load_bundle,
    load_processor_step,
    verify_source_root,
)
from embodied_dataops.training.dataset import PreparedACTDataset, open_prepared_dataset
from embodied_dataops.training.processors import (
    PROCESSOR_EPS,
    ActProcessors,
    build_act_processors,
    processor_stats_fingerprint,
    verify_processor_stats,
)

__all__ = [
    "VERIFICATION_SCHEMA",
    "ValidationBudget",
    "run_act_validation",
]

VERIFICATION_SCHEMA = "training-verification/1.1"
#: 规格 §8.2 的工程候选容差；调整它属于新版本而不是运行时覆盖。
ATOL = 1e-5
RTOL = 1e-4
#: §13 的 A10 通过清单：首个候选 10 个优化器步骤、chunk=16。
GPU_PASS_STEPS = 10
GPU_PASS_CHUNK_SIZE = 16
#: §13 要求报告记录的机器信息；缺任一项都不能算 GPU_VERIFIED。
REQUIRED_MACHINE_KEYS = ("name", "uuid", "driver", "os", "cuda_version", "capability", "precision")
#: 这些占位值等同于"没记录"，不能拿来满足 §13。
_PLACEHOLDER_VALUES = (None, "", "unknown", "Unknown", "none", "None")


@dataclass(frozen=True)
class ValidationBudget:
    device: str
    steps: int
    batch_size: int = 1
    seed: int = 20261031

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "optimizer_steps": int(self.steps),
            "batch_size": int(self.batch_size),
            "seed": int(self.seed),
        }


@dataclass
class _Checks:
    """逐项检查结果；任何一项 False 都会让整体状态变成 FAIL。"""

    results: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, ok: bool, **details: Any) -> None:
        self.results[name] = bool(ok)
        if details:
            self.details[name] = details

    def fail(self, name: str, **details: Any) -> None:
        self.record(name, False, **details)

    @property
    def ok(self) -> bool:
        return all(self.results.values())


def _resolve_device(requested: str) -> Any:
    import torch  # noqa: PLC0415

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "the requested device is not a valid torch device string",
            details={"device": requested, "error": str(exc)},
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ApiError(
                ErrorCode.CUDA_UNAVAILABLE,
                "CUDA was requested but is not available; refusing to fall back to CPU",
                details={"requested_device": requested},
            )
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise ApiError(
                ErrorCode.CUDA_UNAVAILABLE,
                "the requested CUDA device index does not exist",
                details={"requested_device": requested, "device_count": torch.cuda.device_count()},
            )
        return torch.device("cuda", index)
    if device.type != "cpu":
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "only cpu and cuda devices are supported",
            details={"requested_device": requested},
        )
    return device


def _nvidia_smi(field: str) -> str | None:
    """从 nvidia-smi 取 torch 不暴露的字段（驱动版本、UUID）。

    取不到就返回 None，不编造 "unknown"——编造的占位值会让 §13 的机器记录
    看起来完整，而实际上什么都没记。
    """
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(  # noqa: S603 - 固定参数，无 shell
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = result.stdout.strip().splitlines()
    if result.returncode != 0 or not values:
        return None
    return values[0].strip() or None


def _device_record(device: Any, *, precision: str) -> dict[str, Any]:
    import platform  # noqa: PLC0415

    import torch  # noqa: PLC0415

    record: dict[str, Any] = {
        "requested": str(device),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "os": platform.platform(),
        "precision": precision,
    }
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        record.update(
            {
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "capability": f"{properties.major}.{properties.minor}",
                "cuda_version": str(torch.version.cuda),
            }
        )
        # torch 2.10 起 driver_version 未必存在；取不到就查 nvidia-smi，都没有就留 None。
        driver = getattr(torch.cuda, "driver_version", None)
        if not isinstance(driver, str) or not driver or driver.lower() == "unknown":
            driver = _nvidia_smi("driver_version")
        record["driver"] = driver
        uuid = getattr(properties, "uuid", None)
        record["uuid"] = str(uuid) if uuid else _nvidia_smi("uuid")
    return record


def _policy_features(loaded: LoadedBundle) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: PLC0415

    features = loaded.resolved.features
    input_features = {
        features.state_key: PolicyFeature(type=FeatureType.STATE, shape=(features.state_dim,))
    }
    for camera in features.cameras:
        shape = features.image_shapes.get(camera)
        if not shape:
            raise ApiError(
                ErrorCode.FEATURE_INCOMPATIBLE,
                "the resolved recipe has no image shape for a camera",
                details={"camera": camera},
            )
        input_features[f"observation.images.{camera}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=tuple(int(size) for size in shape)
        )
    output_features = {
        features.action_key: PolicyFeature(type=FeatureType.ACTION, shape=(features.action_dim,))
    }
    return input_features, output_features


def _build_policy(loaded: LoadedBundle, device: Any, seed: int) -> Any:
    import torch  # noqa: PLC0415
    from lerobot.policies.act.configuration_act import ACTConfig  # noqa: PLC0415
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: PLC0415

    resolved = loaded.resolved
    input_features, output_features = _policy_features(loaded)
    torch.manual_seed(int(seed))

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=int(resolved.temporal.chunk_size),
        n_action_steps=int(resolved.temporal.n_action_steps),
        n_obs_steps=1,
        vision_backbone=str(resolved.target.vision_backbone),
        # 配方显式 null 时不下载预训练骨干：离线 smoke 不能偷偷联网取权重。
        pretrained_backbone_weights=resolved.target.pretrained_backbone_weights,
        device=str(device),
    )
    return ACTPolicy(config).to(device), config


def _load_processors(loaded: LoadedBundle) -> ActProcessors:
    """从 bundle 重载冻结 processor，并断言生效统计量与冻结值完全一致。"""
    directory = loaded.path / "processors"
    pre = load_processor_step(directory, "pre.json")
    post = load_processor_step(directory, "post.json")

    rebuilt = build_act_processors(loaded.resolved, loaded.stats)
    processors = ActProcessors(
        pre=pre,
        post=post,
        stats_payload=rebuilt.stats_payload,
        config_sha256=str(loaded.manifest.get("digests", {}).get("processor")),
        action_key=loaded.resolved.features.action_key,
    )
    verify_processor_stats(processors, loaded.stats, resolved=loaded.resolved)
    for step in (processors.pre, processors.post):
        eps = float(getattr(step, "eps", -1.0))
        if eps != PROCESSOR_EPS:
            raise ApiError(
                ErrorCode.PROCESSOR_STATS_MISMATCH,
                "a reloaded processor does not use eps=0",
                details={"observed_eps": eps, "expected_eps": PROCESSOR_EPS},
            )
    return processors


def _batch_iterator(dataset: PreparedACTDataset, *, batch_size: int, device: Any) -> Any:
    """无限循环的确定性批次流：shuffle 关闭，顺序由冻结窗口决定。"""
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)

    def _move(batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved

    while True:
        produced = False
        for batch in loader:
            produced = True
            yield _move(batch)
        if not produced:  # pragma: no cover - 空 split 已在准备阶段被拒
            raise ApiError(ErrorCode.EMPTY_REQUIRED_WINDOWS, "the split produced no batch")


def _finite(tensor: Any) -> bool:
    import torch  # noqa: PLC0415

    return bool(torch.isfinite(tensor).all().item())


def _train_steps(
    policy: Any,
    optimizer: Any,
    processors: ActProcessors,
    batches: Any,
    *,
    steps: int,
    checks: _Checks,
) -> dict[str, Any]:

    losses: list[float] = []
    for step in range(int(steps)):
        batch = next(batches)
        normalised = processors.normalise(batch)
        loss, loss_dict = policy(normalised)
        if not _finite(loss):
            checks.fail("loss_finite", step=step, loss=repr(float(loss)))
            raise ApiError(
                ErrorCode.MODEL_NUMERICS_FAILED,
                "the training loss is not finite",
                details={"step": step, "loss": float(loss)},
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        checked_gradients = 0
        for name, parameter in policy.named_parameters():
            if parameter.grad is None:
                continue
            checked_gradients += 1
            if not _finite(parameter.grad):
                checks.fail("gradients_finite", parameter=name, step=step)
                raise ApiError(
                    ErrorCode.MODEL_NUMERICS_FAILED,
                    "a gradient produced by the real backward pass is not finite",
                    details={"parameter": name, "step": step},
                )
        if checked_gradients == 0:
            checks.fail("gradients_present", step=step)
            raise ApiError(
                ErrorCode.MODEL_NUMERICS_FAILED,
                "the backward pass produced no gradient on any trainable parameter",
                details={"step": step},
            )
        optimizer.step()
        losses.append(float(loss.detach().item()))
        checks.details.setdefault("losses", []).append(losses[-1])

    checks.record("loss_finite", True)
    checks.record("gradients_finite", True)
    checks.record("gradients_present", True)
    checks.record("optimizer_step", True)
    return {
        "steps": int(steps),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "note": "a decreasing loss is not a pass criterion for a short smoke run",
    }


def _reload_check(policy: Any, batch: dict[str, Any], workdir: Path, checks: _Checks) -> dict[str, Any]:
    """checkpoint 保存 → 重载 → eval 相同输入输出一致（TP-15）。"""
    import torch  # noqa: PLC0415
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: PLC0415

    saved = workdir / "policy"
    saved.mkdir(parents=True, exist_ok=True)
    policy.eval()
    with torch.no_grad():
        before = policy.predict_action_chunk(batch).detach().clone()
    if not _finite(before):
        checks.fail("eval_output_finite")
        raise ApiError(
            ErrorCode.MODEL_NUMERICS_FAILED,
            "the eval forward produced a non-finite action chunk",
        )

    policy.save_pretrained(saved)
    reloaded = ACTPolicy.from_pretrained(saved)
    device = next(policy.parameters()).device
    reloaded = reloaded.to(device)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded.predict_action_chunk(batch).detach().clone()

    try:
        torch.testing.assert_close(after, before, atol=ATOL, rtol=RTOL)
    except AssertionError as exc:
        checks.fail("checkpoint_roundtrip", error=str(exc)[:400])
        raise ApiError(
            ErrorCode.MODEL_NUMERICS_FAILED,
            "the reloaded checkpoint does not reproduce the eval output",
            details={"atol": ATOL, "rtol": RTOL},
        ) from exc

    checks.record("checkpoint_roundtrip", True)
    return {
        # 只用相对名字：绝对临时路径既不可复现，也不该进产物身份。
        "checkpoint": saved.name,
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs_diff": float((after - before).abs().max().item()),
    }


@dataclass(frozen=True)
class _LevelAssessment:
    """等级判定的三项结论：达成值、状态、以及为什么没达成。"""

    achieved: str | None
    level_status: str
    blocking: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _assess_level(
    *,
    requested_level: str,
    checks: _Checks,
    report: dict[str, Any],
    val_eval: dict[str, Any] | None,
    steps: int,
    chunk_size: int,
    recipe_digest: str | None,
) -> _LevelAssessment:
    """把 §13 的通过清单变成可核实的门槛，而不是只看 8 项局部检查。

    ``GPU_VERIFIED`` 的关键词是"完整报告"：没有 val-split eval、机器记录缺 UUID 或
    驱动、或只跑 3 步，都不允许拿到这个等级。做不到就写 ``PARTIAL`` 并列出缺口——
    这样下游读报告的人和写文档的人会看到同一个结论。
    """
    notes: list[str] = []
    if val_eval is None:
        notes.append("no eval was executed on the val split")

    if requested_level != "GPU_VERIFIED":
        # CPU_VERIFIED 的既有语义就是"8 项计算检查全过"，证据缺口只记录不降级。
        if not checks.ok:
            return _LevelAssessment(None, "FAILED", (), tuple(notes))
        return _LevelAssessment(requested_level, "VERIFIED", (), tuple(notes))

    blocking: list[str] = []
    machine = report.get("device") or {}
    for key in REQUIRED_MACHINE_KEYS:
        if machine.get(key) in _PLACEHOLDER_VALUES:
            blocking.append(f"machine record is missing {key}")
    if report.get("command") in _PLACEHOLDER_VALUES:
        blocking.append("machine record is missing command")
    if not recipe_digest:
        blocking.append("machine record is missing the recipe digest")

    if val_eval is None:
        blocking.append("no val-split eval was executed (§13 requires at least one)")
    elif not val_eval.get("finite"):
        blocking.append("the val-split eval produced non-finite actions")

    if int(steps) < GPU_PASS_STEPS:
        blocking.append(f"only {int(steps)} optimizer steps; §13 requires {GPU_PASS_STEPS}")

    if int(chunk_size) != GPU_PASS_CHUNK_SIZE:
        # §13 允许调整 chunk，但必须形成可追溯的新配方身份，而不是悄悄换掉首候选。
        notes.append(
            f"chunk_size={int(chunk_size)} deviates from §13's first candidate "
            f"{GPU_PASS_CHUNK_SIZE}; recorded under a distinct recipe identity"
        )

    if not checks.ok:
        return _LevelAssessment(None, "FAILED", tuple(blocking), tuple(notes))
    if blocking:
        return _LevelAssessment(None, "PARTIAL", tuple(blocking), tuple(notes))
    return _LevelAssessment(requested_level, "VERIFIED", (), tuple(notes))


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    import json  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def run_act_validation(
    *,
    bundle: str | Path,
    source_root: str | Path,
    device: str = "cpu",
    steps: int = 1,
    report_dir: str | Path | None = None,
    batch_size: int = 1,
    seed: int = 20261031,
    report_name: str | None = None,
    workdir: str | Path | None = None,
    record_timestamp: bool = True,
    redact_paths: bool = False,
    command: str | None = None,
    precision: str = "fp32",
) -> dict[str, Any]:
    """对已发布的 bundle 执行真实 ACT 计算验证，并写出外部验证记录。"""
    if int(steps) <= 0:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "steps must be a positive number of optimizer steps",
            details={"steps": int(steps)},
        )

    loaded = load_bundle(bundle)
    verify_source_root(source_root, loaded.source_layout())

    budget = ValidationBudget(device=device, steps=int(steps), batch_size=int(batch_size), seed=int(seed))
    requested_level = "GPU_VERIFIED" if str(device).startswith("cuda") else "CPU_VERIFIED"

    report_root = Path(report_dir) if report_dir else loaded.path.parent / "verification"
    name = report_name or ("gpu-validation.json" if requested_level == "GPU_VERIFIED" else "cpu-validation.json")
    report_path = report_root / name

    checks = _Checks()
    created_workdir: Path | None = None
    recipe_digest = (loaded.manifest.get("recipe") or {}).get("resolved_recipe_sha256")
    chunk_size = int(loaded.resolved.temporal.chunk_size)
    # 只有 val 才有 eval 资格；val 缺席时退回 train 只为跑通 reload，不计入等级。
    val_eval: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "schema_version": VERIFICATION_SCHEMA,
        "requested_level": requested_level,
        "achieved_level": None,
        "level_status": None,
        # status 只说局部检查全过，等级另算；两者分离，避免 8/8 被读成 A10 通过。
        "status": "FAIL",
        "bundle": {
            # bundle 内部的报告必须可复现：临时构建路径不能进产物。
            "path": "<bundle>" if redact_paths else str(loaded.path),
            "preparation_id": loaded.preparation_id,
            "artifact_id": loaded.artifact_id,
        },
        "source_root": "<source-root>" if redact_paths else str(Path(source_root)),
        "budget": budget.as_dict(),
        "command": command,
        "recipe": {"resolved_recipe_sha256": recipe_digest, "chunk_size": chunk_size},
        "eval": None,
        "limitations": [],
    }
    if record_timestamp:
        # 写进 bundle 的验证报告必须逐字节可复现，所以运行时间只留在外部记录里。
        report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        resolved_device = _resolve_device(device)
        torch = __import__("torch")
        torch.manual_seed(int(seed))
        report["device"] = _device_record(resolved_device, precision=precision)

        processors = _load_processors(loaded)
        checks.record(
            "processor_stats_frozen",
            True,
            fingerprint=processor_stats_fingerprint(processors.pre),
        )

        split_windows = loaded.windows_for("train")
        if not split_windows:
            raise ApiError(
                ErrorCode.EMPTY_TRAIN_WINDOWS,
                "the bundle has no train window to validate",
            )

        dataset = open_prepared_dataset(
            split="train",
            windows=split_windows,
            resolved=loaded.resolved,
            source_root=source_root,
            repo_id=str((loaded.manifest.get("source") or {}).get("dataset_slug") or "bundle/train"),
        )
        checks.record("reader_built", True, windows=len(dataset))

        policy, config = _build_policy(loaded, resolved_device, int(seed))
        checks.record("policy_built", True, parameters=int(sum(p.numel() for p in policy.parameters())))

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=float(config.optimizer_lr),
            weight_decay=float(config.optimizer_weight_decay),
        )
        batches = _batch_iterator(dataset, batch_size=budget.batch_size, device=resolved_device)

        policy.train()
        report["training"] = _train_steps(
            policy, optimizer, processors, batches, steps=budget.steps, checks=checks
        )

        policy.eval()
        val_windows = loaded.windows_for("val")
        if val_windows:
            val_dataset = open_prepared_dataset(
                split="val",
                windows=val_windows,
                resolved=loaded.resolved,
                source_root=source_root,
                repo_id=str((loaded.manifest.get("source") or {}).get("dataset_slug") or "bundle/val"),
            )
            val_batch = processors.normalise(
                next(_batch_iterator(val_dataset, batch_size=budget.batch_size, device=resolved_device))
            )
            with torch.no_grad():
                val_actions = policy.predict_action_chunk(val_batch)
            if not _finite(val_actions):
                checks.fail("val_eval_finite")
                raise ApiError(
                    ErrorCode.MODEL_NUMERICS_FAILED,
                    "the val-split eval produced a non-finite action chunk",
                )
            val_eval = {
                "split": "val",
                "windows": len(val_dataset),
                "finite": True,
                "shape": [int(size) for size in val_actions.shape],
            }
            eval_batch = val_batch
        else:
            # 没有 val 窗口时仍要验证 reload，但这张 batch 来自 train，不能算 §13 的 eval。
            eval_batch = processors.normalise(
                next(_batch_iterator(dataset, batch_size=budget.batch_size, device=resolved_device))
            )
        report["eval"] = val_eval
        if workdir is not None:
            workdir_path = Path(workdir)
            # 同一目标目录重复验证时必须从干净状态开始，否则上次的 checkpoint 会混进来。
            if workdir_path.exists():
                shutil.rmtree(workdir_path, ignore_errors=True)
            workdir_path.mkdir(parents=True, exist_ok=True)
        else:
            # 默认落在系统临时目录：几百 MB 的 checkpoint 不属于"验证记录"。
            workdir_path = Path(tempfile.mkdtemp(prefix="edo-validate-"))
            created_workdir = workdir_path
        report["reload"] = _reload_check(policy, eval_batch, workdir_path, checks)

        if resolved_device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(resolved_device))
            report["device"]["peak_memory_bytes"] = peak

        report["checks"] = checks.results
        report["check_details"] = checks.details
        report["status"] = "PASS" if checks.ok else "FAIL"
        assessment = _assess_level(
            requested_level=requested_level,
            checks=checks,
            report=report,
            val_eval=val_eval,
            steps=budget.steps,
            chunk_size=chunk_size,
            recipe_digest=recipe_digest,
        )
        report["achieved_level"] = assessment.achieved
        report["level_status"] = assessment.level_status
        report["level_basis"] = {
            "blocking": list(assessment.blocking),
            "notes": list(assessment.notes),
        }
        report["limitations"] = [*assessment.blocking, *assessment.notes]
        if not checks.ok:
            raise ApiError(
                ErrorCode.MODEL_NUMERICS_FAILED,
                "one or more validation checks failed",
                details={"checks": checks.results},
            )
        return report
    except ApiError as exc:
        blocked = exc.code in (
            ErrorCode.CUDA_UNAVAILABLE,
            ErrorCode.MODEL_UNAVAILABLE,
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            ErrorCode.MODEL_CHECKSUM_MISMATCH,
        )
        report["status"] = "BLOCKED" if blocked else "FAIL"
        report["achieved_level"] = None
        report["level_status"] = "BLOCKED" if blocked else "FAILED"
        report["level_basis"] = {"blocking": [f"{exc.code.value}: {exc.message}"], "notes": []}
        report["checks"] = checks.results
        report["error"] = {"code": exc.code.value, "message": exc.message, "details": exc.details}
        raise
    finally:
        _write_report(report_path, report)
        if created_workdir is not None:
            shutil.rmtree(created_workdir, ignore_errors=True)

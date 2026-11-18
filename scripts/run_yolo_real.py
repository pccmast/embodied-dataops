"""Real YOLO11n inference evidence (acceptance V21-10).

Runs the fixed detector over frames taken from a **real** public dataset video that is
already present under ``public-data/`` and writes a machine-readable report.  Nothing
here is simulated: the weights SHA-256, the ultralytics version, the source video hash,
the per-frame detections and the latencies are all measured at run time.

A missing weight file or a missing source video produces ``BLOCKED`` with a reason, never
a fabricated PASS (SPEC_V2.1 §7.1, §9).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPOSITORY_ROOT / "var" / "models" / "yolo11n.pt"
DEFAULT_VIDEO = (
    REPOSITORY_ROOT
    / "public-data"
    / "utokyo_xarm_pick_and_place"
    / "videos"
    / "observation.images.image"
    / "chunk-000"
    / "file-000.mp4"
)
DEFAULT_OUT = REPOSITORY_ROOT / "reports" / "v21" / "yolo-real.json"

#: Fixed detector contract (SPEC_V2.1 §2, CONFIG_DECISIONS_V21).
IMGSZ = 640
CONF = 0.35
IOU = 0.5
MAX_DET = 100
#: How many distinct frames to infer on; the acceptance threshold is ten.
FRAMES_REQUIRED = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_smi(field: str) -> str | None:
    """从 nvidia-smi 取 torch 不暴露的字段（驱动版本、UUID）。

    容器内 nvidia-smi 可用才取；取不到就不写这个键，不编造 "unknown"。
    """
    import subprocess  # noqa: PLC0415 -- 只在需要设备记录时导入

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
    value = result.stdout.strip().splitlines()
    if result.returncode != 0 or not value:
        return None
    return value[0].strip() or None


def _gpu_record(index: int) -> dict[str, Any]:
    """实际设备记录：名称、UUID、算力、显存、驱动、torch/CUDA 版本。"""
    import torch  # noqa: PLC0415

    properties = torch.cuda.get_device_properties(index)
    record: dict[str, Any] = {
        "index": index,
        "name": str(properties.name),
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        record["uuid"] = str(uuid)
    driver = _nvidia_smi("driver_version")
    if driver:
        record["driver_version"] = driver
    return record


def _resolve_device(requested: str | None, require_cuda: bool) -> Any:
    """解析设备；``--require-cuda`` 时绝不退回 CPU。"""
    import torch  # noqa: PLC0415

    target = requested or ("cuda:0" if require_cuda else "cpu")
    try:
        device = torch.device(target)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device string {target!r}: {exc}") from exc

    if require_cuda and device.type != "cuda":
        raise ValueError(f"--require-cuda was given but the resolved device is {device.type}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"cuda:{index} does not exist (device_count={torch.cuda.device_count()})")
        return torch.device("cuda", index)
    return torch.device("cpu")


def _grab_frames(video: Path, count: int) -> tuple[list[Any], dict[str, Any]]:
    """用 PyAV 按帧号抽样解码。

    这里**不用 cv2.VideoCapture**：pypi 的 opencv wheel 内置 FFmpeg 通常不带 AV1
    解码器，而本数据集的视频是 AV1（libdav1d）。用 cv2 会静默解出 0 帧，
    看起来像"视频坏了"，其实是解码能力缺失。PyAV 是项目 dev extra 的显式依赖，
    走系统/自带 FFmpeg，能把真实解码器名写进证据里。
    """
    import av  # noqa: PLC0415 -- optional dependency, imported only when running

    frames: list[Any] = []
    indexes: list[int] = []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {video}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        total = int(stream.frames or 0)
        codec = stream.codec_context.name
        step = max(1, int(round(fps))) if fps > 0 else 1

        for wanted in (position * step for position in range(count)):
            if total and wanted >= total:
                break
            if fps > 0:
                # 用微秒偏移 seek，再往后解到目标帧；顺序解码 7490 帧没必要。
                container.seek(int(round(wanted / fps * 1_000_000)), stream=stream, backward=True)
            found = None
            for frame in container.decode(stream):
                if fps <= 0 or frame.pts is None:
                    found = frame
                    break
                if round(float(frame.pts * stream.time_base) * fps) >= wanted:
                    found = frame
                    break
            if found is None:
                break
            frames.append(found.to_ndarray(format="bgr24"))
            indexes.append(wanted)

    meta = {
        "fps": fps,
        "declared_frames": total,
        "decoder": codec,
        "sampled_frame_indexes": indexes,
        "sampled": len(frames),
    }
    return frames, meta


def run(
    weights: Path,
    video: Path,
    out: Path,
    device_requested: str | None = None,
    require_cuda: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "test_id": "V21-10",
        "title": "real YOLO11n inference",
        "generated_at": datetime.now(UTC).isoformat(),
        "params": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "max_det": MAX_DET},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }

    try:
        device = _resolve_device(device_requested, require_cuda)
    except (RuntimeError, ValueError) as exc:
        # 设备问题一律 BLOCKED：写 PASS 等于声称在从未用过的设备上跑过。
        report.update(status="BLOCKED", reason=f"device unavailable: {exc}")
        return report
    report["environment"]["device"] = str(device)
    if device.type == "cuda":
        report["environment"]["gpu"] = _gpu_record(device.index or 0)

    if not weights.is_file():
        report.update(status="BLOCKED", reason=f"weights not found at {weights.as_posix()}")
        return report
    if not video.is_file():
        report.update(status="BLOCKED", reason=f"source video not found at {video.as_posix()}")
        return report

    report["weights"] = {
        "path": weights.relative_to(REPOSITORY_ROOT).as_posix(),
        "byte_size": weights.stat().st_size,
        "sha256": _sha256(weights),
    }
    report["source"] = {
        "path": video.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": _sha256(video),
        "dataset": "lerobot/utokyo_xarm_pick_and_place@da56c23e1b6d32c101c300b85dc54288b53ea63c",
        "data_origin": "PUBLIC_REAL_ROBOT",
    }

    try:
        import ultralytics  # noqa: PLC0415
        from ultralytics import YOLO  # noqa: PLC0415

        report["environment"]["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # pragma: no cover - dependency present in this environment
        report.update(status="BLOCKED", reason=f"ultralytics unavailable: {type(exc).__name__}")
        return report

    try:
        frames, frame_meta = _grab_frames(video, FRAMES_REQUIRED)
    except ImportError as exc:
        report.update(status="BLOCKED", reason=f"video decoder unavailable: {type(exc).__name__}")
        return report
    except Exception as exc:  # noqa: BLE001 - 解码失败要写进证据，不是崩掉
        report.update(status="BLOCKED", reason=f"video decode failed: {type(exc).__name__}: {exc}")
        return report
    report["frames"] = frame_meta
    if len(frames) < FRAMES_REQUIRED:
        report.update(
            status="BLOCKED",
            reason=f"only {len(frames)} frames could be decoded, need {FRAMES_REQUIRED}",
        )
        return report

    model = YOLO(str(weights))
    per_frame: list[dict[str, Any]] = []
    latencies: list[float] = []
    for position, (frame_index, frame) in enumerate(zip(frame_meta["sampled_frame_indexes"], frames, strict=True)):
        started = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            device=str(device),
            verbose=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        boxes = results[0].boxes
        detections: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            names = results[0].names
            for class_id, score, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            ):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": names.get(int(class_id), str(int(class_id))),
                        "score": round(float(score), 6),
                        "xyxy": [round(float(value), 3) for value in xyxy],
                    }
                )
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "position": position,
                "detection_count": len(detections),
                "detections": detections,
            }
        )

    report["frames_detail"] = per_frame
    report["metrics"] = {
        "frames_inferred": len(per_frame),
        "total_detections": sum(item["detection_count"] for item in per_frame),
        "frames_with_detections": sum(1 for item in per_frame if item["detection_count"]),
        "latency_ms_p50": round(statistics.median(latencies), 2),
        "latency_ms_p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2),
        "device": str(device),
        "batch_size": 1,
    }
    report["status"] = "PASS"
    report["reason"] = (
        f"{len(per_frame)} real frames inferred with real weights on {device}; "
        f"{report['metrics']['total_detections']} detections"
    )
    _ = out
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real YOLO11n inference evidence (V21-10).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for inference: cpu (default), cuda, cuda:0",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="refuse to run unless inference happens on CUDA (reports BLOCKED, never falls back to cpu)",
    )
    args = parser.parse_args(argv)

    report = run(
        args.weights,
        args.video,
        args.out,
        device_requested=args.device,
        require_cuda=args.require_cuda,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{report['status']}] {report.get('reason', '')}")
    print(f"report: {args.out}")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

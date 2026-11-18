"""Robust FFmpeg / ffprobe / PyAV wrapper (SPEC 2, 5.5, 12.1).

Design rules enforced here:

* binary discovery order is ``FFMPEG_BIN`` / ``FFPROBE_BIN`` environment
  variables, then ``PATH``, then the interpreter bundled by
  ``imageio-ffmpeg`` (only ffmpeg; a sibling ``ffprobe`` next to it is used when
  present);
* when no ``ffprobe`` exists, probing and PTS extraction fall back to **PyAV**
  (exact presentation timestamps; this is the supported local path because
  ``imageio-ffmpeg`` bundles ffmpeg only), then to parsing ``ffmpeg -i`` for
  constant-frame-rate containers.  Every degraded path is recorded explicitly
  (``probed_with`` / ``derived_from_cfr``) so the importer can keep the
  limitation in its evidence;
* every command is an **argv list** with a timeout, ``-nostdin`` and no shell,
  so user-controlled paths are never interpolated into a shell string;
* a decode/encode problem is surfaced as :class:`ApiError` with a stable
  :class:`~embodied_dataops.errors.ErrorCode` and a *sanitized* stderr tail
  (source paths are replaced by ``<source>`` so internal paths never leak into
  API error details -- SPEC 12.1);
* video helpers import :mod:`embodied_dataops.canonical.video` for the 640x480
  letterbox math, so the canonical frame definition has exactly one owner.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS, TARGET_FPS
from embodied_dataops.canonical.video import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    apply_letterbox,
    compute_letterbox,
)
from embodied_dataops.errors import ApiError, ErrorCode

#: One command may never run longer than the import budget (SPEC 6.1).
COMMAND_TIMEOUT_SECONDS = 600.0
#: Nearest-neighbour tolerance for the 10 Hz grid (SPEC 5.5).
FRAME_TOLERANCE_NS = 50_000_000


class FFmpegUnavailable(ApiError):
    """The host has no usable FFmpeg/ffprobe/PyAV media backend."""

    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.DEPENDENCY_UNAVAILABLE, message)


# ---------------------------------------------------------------------------
# binary / backend discovery
# ---------------------------------------------------------------------------

_binary_cache: dict[str, str | None] = {}
#: Reentrant: resolving ``ffprobe`` may recursively resolve its sibling
#: ``ffmpeg`` inside the same call, so a plain ``Lock`` would deadlock.
_binary_lock = threading.RLock()


def reset_binary_cache() -> None:
    """Test hook: forget resolved binaries (e.g. after changing env vars)."""
    with _binary_lock:
        _binary_cache.clear()


def _env_binary(name: str) -> str | None:
    value = os.environ.get(name, "").strip().strip('"')
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_file() else None


def _imageio_ffmpeg_binary() -> str | None:
    try:  # optional dependency: imageio-ffmpeg ships an ffmpeg executable only
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return candidate if candidate and Path(candidate).is_file() else None


def _sibling_ffprobe(ffmpeg_exe: str | None) -> str | None:
    if not ffmpeg_exe:
        return None
    suffix = ".exe" if os.name == "nt" else ""
    candidate = Path(ffmpeg_exe).with_name(f"ffprobe{suffix}")
    return str(candidate) if candidate.is_file() else None


def _resolve(kind: str) -> str | None:
    with _binary_lock:
        if kind in _binary_cache:
            return _binary_cache[kind]
        resolved: str | None = None
        if kind == "ffmpeg":
            resolved = _env_binary("FFMPEG_BIN") or shutil.which("ffmpeg") or _imageio_ffmpeg_binary()
        else:  # ffprobe
            resolved = _env_binary("FFPROBE_BIN") or shutil.which("ffprobe")
            if resolved is None:
                resolved = _sibling_ffprobe(_resolve("ffmpeg"))
        _binary_cache[kind] = resolved
        return resolved


def ffmpeg_available() -> bool:
    return _resolve("ffmpeg") is not None


def ffprobe_available() -> bool:
    return _resolve("ffprobe") is not None


def pyav_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:  # pragma: no cover - optional dependency
        return False


def ffmpeg_path() -> str:
    path = _resolve("ffmpeg")
    if path is None:
        raise FFmpegUnavailable(
            "ffmpeg binary not found: set FFMPEG_BIN, put ffmpeg on PATH, or install "
            "imageio-ffmpeg (it bundles a static ffmpeg build)"
        )
    return path


def ffprobe_path() -> str:
    path = _resolve("ffprobe")
    if path is None:
        raise FFmpegUnavailable(
            "ffprobe binary not found: set FFPROBE_BIN or put ffprobe on PATH "
            "(probing falls back to PyAV / ffmpeg -i when it is absent)"
        )
    return path


_FFMPEG_VERSION_RE = re.compile(r"ffmpeg version n?(\d+)\.(\d+)")
_ffmpeg_version_cache: tuple[int, int] | None = None
_ffmpeg_version_checked = False


def ffmpeg_version() -> tuple[int, int] | None:
    """Return the ``(major, minor)`` version of the resolved ffmpeg, when parsable."""
    global _ffmpeg_version_cache, _ffmpeg_version_checked
    if _ffmpeg_version_checked:
        return _ffmpeg_version_cache
    _ffmpeg_version_checked = True
    try:
        completed = subprocess.run(
            [ffmpeg_path(), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = _FFMPEG_VERSION_RE.search(completed.stdout or "")
        if match:
            _ffmpeg_version_cache = (int(match.group(1)), int(match.group(2)))
    except Exception:  # pragma: no cover - unresolvable binary
        _ffmpeg_version_cache = None
    return _ffmpeg_version_cache


def fps_passthrough_args() -> list[str]:
    """Select the "keep source PTS" flags supported by this ffmpeg build.

    ``-fps_mode`` was added in ffmpeg 5.1; older builds (e.g. Ubuntu 22.04's
    4.4.2) only understand the deprecated ``-vsync 0``.  Without this the decode
    command dies with "Unrecognized option 'fps_mode'" and every frame-based
    check reports a bogus ``FILE_DECODE_ERROR``.
    """
    version = ffmpeg_version()
    if version is not None and version >= (5, 1):
        return ["-fps_mode", "passthrough"]
    return ["-vsync", "0"]


def media_backend_summary() -> dict[str, Any]:
    """Audit evidence: which media backends this host provides."""
    return {
        "ffmpeg": _resolve("ffmpeg"),
        "ffprobe": _resolve("ffprobe"),
        "ffmpeg_version": ffmpeg_version(),
        "pyav": pyav_available(),
        "imageio_ffmpeg": bool(_imageio_ffmpeg_binary()),
    }


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------


def _sanitize(text: str, path: Path | None = None, *, limit: int = 800) -> str:
    """Remove local paths from a stderr tail before it becomes error evidence."""
    cleaned = text or ""
    candidates: set[str] = set()
    if path is not None:
        candidates.update({str(path), str(path.resolve()), path.name, path.as_posix()})
    for token in sorted(candidates, key=len, reverse=True):
        if token:
            cleaned = cleaned.replace(token, "<source>")
    cleaned = re.sub(r"[A-Za-z]:\\[^\s'\"<>]+", "<path>", cleaned)
    cleaned = re.sub(r"(?<![\w<>])/(?:[\w.\-]+/){2,}[\w.\-]+", "<path>", cleaned)
    cleaned = " ".join(line.strip() for line in cleaned.strip().splitlines() if line.strip())
    return cleaned[-limit:]


def _run(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    stdin: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    timeout = COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
    try:
        return subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL if stdin is None else stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ApiError(
            ErrorCode.TIMEOUT,
            "media command exceeded the time budget",
            details={"timeout_seconds": timeout, "command": Path(argv[0]).name},
        ) from exc
    except OSError as exc:  # pragma: no cover - executable vanished mid-run
        raise FFmpegUnavailable(f"failed to execute {Path(argv[0]).name}: {exc.strerror}") from exc


class _KillOnTimeout:
    """Watchdog that kills a streamed child process if it exceeds the budget."""

    def __init__(self, process: subprocess.Popen[bytes], timeout: float) -> None:
        self.process = process
        self.timeout = timeout
        self.expired = False
        self._timer: threading.Timer | None = None

    def _fire(self) -> None:
        self.expired = True
        try:
            self.process.kill()
        except Exception:  # pragma: no cover - process already gone
            pass

    def __enter__(self) -> _KillOnTimeout:
        self._timer = threading.Timer(self.timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._timer is not None:
            self._timer.cancel()


def _pyav_error_payload(exc: BaseException, path: Path) -> dict[str, Any]:
    return {"reason": _sanitize(f"{type(exc).__name__}: {exc}", path)}


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoProbe:
    """Container/stream facts needed by importers and normalizers."""

    duration_ns: int
    width: int
    height: int
    codec_name: str
    pix_fmt: str
    has_audio: bool
    format_name: str
    avg_frame_rate: float | None = None
    r_frame_rate: float | None = None
    nb_frames: int | None = None
    bit_rate: int | None = None
    probed_with: str = "ffprobe"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1_000_000_000

    @property
    def is_cfr(self) -> bool:
        if self.avg_frame_rate is None or self.r_frame_rate is None:
            return False
        return abs(self.avg_frame_rate - self.r_frame_rate) < 0.01

    @property
    def fps(self) -> float | None:
        return self.avg_frame_rate or self.r_frame_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_ns": self.duration_ns,
            "width": self.width,
            "height": self.height,
            "codec_name": self.codec_name,
            "pix_fmt": self.pix_fmt,
            "has_audio": self.has_audio,
            "format_name": self.format_name,
            "avg_frame_rate": self.avg_frame_rate,
            "r_frame_rate": self.r_frame_rate,
            "nb_frames": self.nb_frames,
            "bit_rate": self.bit_rate,
            "probed_with": self.probed_with,
        }


def _rate_to_float(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        if isinstance(value, str) and "/" in value:
            fraction = Fraction(value)
            return float(fraction) if fraction.denominator else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _duration_text_to_ns(text: str) -> int:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    return int(round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1_000_000_000))


def _probe_with_ffprobe(path: Path, *, timeout: float) -> VideoProbe:
    argv = [
        ffprobe_path(),
        "-v",
        "error",
        "-hide_banner",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = _run(argv, timeout=timeout)
    if completed.returncode != 0:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video stream cannot be read",
            details={"reason": _sanitize(completed.stderr.decode("utf-8", "replace"), path)},
        )
    try:
        payload = json.loads(completed.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Video metadata is not parseable") from exc
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source has no video stream")
    audio = any(s.get("codec_type") == "audio" for s in streams)
    fmt = payload.get("format") or {}
    duration_ns = 0
    try:
        duration_ns = int(round(float(fmt.get("duration")) * 1_000_000_000))
    except (TypeError, ValueError):
        pass
    if duration_ns <= 0:
        try:
            duration_ns = int(round(float(video.get("duration")) * 1_000_000_000))
        except (TypeError, ValueError):
            duration_ns = 0
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source video reports no frame size")
    nb_frames: int | None = None
    if video.get("nb_frames") not in (None, "", "N/A"):
        try:
            nb_frames = int(video["nb_frames"])
        except (TypeError, ValueError):
            nb_frames = None
    return VideoProbe(
        duration_ns=duration_ns,
        width=width,
        height=height,
        codec_name=str(video.get("codec_name") or ""),
        pix_fmt=str(video.get("pix_fmt") or ""),
        has_audio=audio,
        format_name=str(fmt.get("format_name") or ""),
        avg_frame_rate=_rate_to_float(video.get("avg_frame_rate")),
        r_frame_rate=_rate_to_float(video.get("r_frame_rate")),
        nb_frames=nb_frames,
        bit_rate=int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        probed_with="ffprobe",
        raw=payload,
    )


def _probe_with_pyav(path: Path) -> VideoProbe:
    """Exact container/stream probe without the ffprobe binary."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source has no video stream")
            stream = container.streams.video[0]
            codec_context = stream.codec_context
            duration_ns = 0
            if container.duration is not None:
                duration_ns = int(round(container.duration / av.time_base * 1_000_000_000))
            if duration_ns <= 0 and stream.duration is not None and stream.time_base is not None:
                duration_ns = int(round(float(stream.duration * stream.time_base) * 1_000_000_000))
            width = int(codec_context.width or 0)
            height = int(codec_context.height or 0)
            if width <= 0 or height <= 0:
                raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source video reports no frame size")
            bit_rate = None
            if container.bit_rate:
                bit_rate = int(container.bit_rate)
            return VideoProbe(
                duration_ns=duration_ns,
                width=width,
                height=height,
                codec_name=str(getattr(codec_context, "name", "") or ""),
                pix_fmt=str(getattr(codec_context, "pix_fmt", "") or ""),
                has_audio=bool(len(container.streams.audio)),
                format_name=str(getattr(container.format, "name", "") or ""),
                avg_frame_rate=_rate_to_float(getattr(stream, "average_rate", None)),
                r_frame_rate=_rate_to_float(getattr(stream, "base_rate", None)),
                nb_frames=int(stream.frames) if getattr(stream, "frames", 0) else None,
                bit_rate=bit_rate,
                probed_with="pyav",
            )
    except ApiError:
        raise
    except Exception as exc:  # av.error.* / OSError / EOFError
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video container cannot be opened",
            details=_pyav_error_payload(exc, path),
        ) from exc


def _probe_with_ffmpeg(path: Path, *, timeout: float) -> VideoProbe:
    """Fallback when neither ffprobe nor PyAV exists: parse ``ffmpeg -i``."""
    argv = [ffmpeg_path(), "-hide_banner", "-nostdin", "-v", "info", "-i", str(path)]
    completed = _run(argv, timeout=timeout)
    text = completed.stderr.decode("utf-8", "replace")
    if re.search(r"Invalid data found|moov atom not found|could not find codec parameters", text, re.I):
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video container cannot be decoded",
            details={"reason": _sanitize(text, path)},
        )
    width_match = None
    for line in text.splitlines():
        if "Video:" in line:
            size_match = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if size_match:
                width_match = size_match
                break
    if width_match is None:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Source has no readable video stream",
            details={"reason": _sanitize(text, path)},
        )
    width, height = int(width_match.group(1)), int(width_match.group(2))
    duration_ns = _duration_text_to_ns(text)
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    fps = float(fps_match.group(1)) if fps_match else None
    codec_match = re.search(r"Video:\s*([A-Za-z0-9_\-]+)", text)
    return VideoProbe(
        duration_ns=duration_ns,
        width=width,
        height=height,
        codec_name=codec_match.group(1) if codec_match else "",
        pix_fmt="",
        has_audio="Audio:" in text,
        format_name="",
        avg_frame_rate=fps,
        r_frame_rate=fps,
        nb_frames=None,
        probed_with="ffmpeg-i",
    )


def probe(path: str | Path, *, timeout: float | None = None) -> VideoProbe:
    """Probe one local video file; raises ``FILE_DECODE_ERROR`` when unreadable."""
    target = Path(path)
    if not target.is_file():
        raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "Source video object is missing")
    if ffprobe_available():
        return _probe_with_ffprobe(target, timeout=COMMAND_TIMEOUT_SECONDS if timeout is None else timeout)
    pyav_error: ApiError | None = None
    if pyav_available():
        try:
            return _probe_with_pyav(target)
        except ApiError as exc:  # keep the PyAV evidence, try ffmpeg -i next
            pyav_error = exc
    if ffmpeg_available():
        try:
            return _probe_with_ffmpeg(
                target, timeout=COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
            )
        except ApiError:
            if pyav_error is not None:
                raise pyav_error from None
            raise
    if pyav_error is not None:
        raise pyav_error from None
    raise FFmpegUnavailable(
        "no media backend available: set FFPROBE_BIN/FFMPEG_BIN, install ffmpeg, or install PyAV"
    )


# ---------------------------------------------------------------------------
# full decode check (SPEC 7.1 FILE_DECODE_ERROR)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeCheck:
    ok: bool
    frame_count: int
    returncode: int = 0
    stderr_tail: str = ""
    timed_out: bool = False
    duration_ns: int = 0
    backend: str = "ffmpeg"
    expected_frame_count: int | None = None
    frame_count_match: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "decoded_frame_count": self.frame_count,
            "expected_frame_count": self.expected_frame_count,
            "frame_count_match": self.frame_count_match,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "backend": self.backend,
        }


#: ``ffmpeg`` can exit 0 while concealing a truncated/corrupt stream, so the
#: declared frame count plus these stderr signatures are checked as well.  A
#: "full decode check" that only looked at the exit code would accept a
#: half-truncated H.264 file (exactly the F07 fixture).
_DECODE_ERROR_PATTERNS = re.compile(
    r"error|invalid|corrupt|truncat|partial file|moov atom|missing picture|"
    r"nal unit|failed to|no sequence|decode_slice_header|out of range|"
    r"reference picture missing|concealing",
    re.IGNORECASE,
)


def _decode_error_signature(stderr: str) -> str | None:
    """Return the first decode-error line of a stderr tail (already sanitized)."""
    if not stderr:
        return None
    for line in stderr.splitlines():
        if _DECODE_ERROR_PATTERNS.search(line):
            return line.strip()[:300]
    return None


def _combine_decode_result(
    check: DecodeCheck,
    *,
    expected_frames: int | None,
) -> DecodeCheck:
    match: bool | None = None
    if expected_frames is not None and expected_frames > 0:
        match = check.frame_count == expected_frames
    signature = _decode_error_signature(check.stderr_tail)
    ok = check.ok and not check.timed_out and not signature
    if check.ok and match is False:
        ok = False
    return DecodeCheck(
        ok=ok,
        frame_count=check.frame_count,
        returncode=check.returncode,
        stderr_tail=check.stderr_tail,
        timed_out=check.timed_out,
        duration_ns=check.duration_ns,
        backend=check.backend,
        expected_frame_count=expected_frames if expected_frames and expected_frames > 0 else None,
        frame_count_match=match,
    )


def _full_decode_with_pyav(path: Path) -> DecodeCheck:
    import av

    count = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return DecodeCheck(False, 0, returncode=-1, stderr_tail="no video stream", backend="pyav")
            for _frame in container.decode(video=0):
                count += 1
    except Exception as exc:  # av.error.* / OSError / EOFError
        return DecodeCheck(
            False,
            count,
            returncode=-1,
            stderr_tail=_sanitize(f"{type(exc).__name__}: {exc}", path),
            backend="pyav",
        )
    return DecodeCheck(count > 0, count, returncode=0 if count else -1, backend="pyav")


def full_decode_check(
    path: str | Path,
    *,
    timeout: float | None = None,
    duration_ns: int = 0,
    expected_frames: int | None = None,
) -> DecodeCheck:
    """Decode **every** frame and count them.

    Preferred path is ``ffmpeg -f null -`` with ``-progress`` frame counting;
    without an ffmpeg binary PyAV decodes the whole stream instead.  A header
    probe alone is never treated as stream-integrity proof (SPEC 2): a
    truncated file can still carry valid headers and even partially decode, so
    decode-error signatures and the declared frame count are verified too.
    """
    target = Path(path)
    budget = COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
    if ffmpeg_available():
        argv = [
            ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-progress",
            "pipe:1",
            "-nostats",
            "-i",
            str(target),
            "-f",
            "null",
            "-",
        ]
        try:
            completed = _run(argv, timeout=budget)
        except ApiError as exc:
            if exc.code is ErrorCode.TIMEOUT:
                return DecodeCheck(
                    ok=False,
                    frame_count=0,
                    returncode=-1,
                    stderr_tail="decode timed out",
                    timed_out=True,
                    duration_ns=duration_ns,
                )
            raise
        stdout = completed.stdout.decode("utf-8", "replace")
        frame_count = 0
        for match in re.finditer(r"^frame=(\d+)$", stdout, re.MULTILINE):
            frame_count = int(match.group(1))
        raw = DecodeCheck(
            ok=completed.returncode == 0 and frame_count > 0,
            frame_count=frame_count,
            returncode=completed.returncode,
            stderr_tail=_sanitize(completed.stderr.decode("utf-8", "replace"), target),
            duration_ns=duration_ns,
            backend="ffmpeg",
        )
        return _combine_decode_result(raw, expected_frames=expected_frames)
    if pyav_available():
        check = _full_decode_with_pyav(target)
        raw = DecodeCheck(
            ok=check.ok,
            frame_count=check.frame_count,
            returncode=check.returncode,
            stderr_tail=check.stderr_tail,
            duration_ns=duration_ns,
            backend=check.backend,
        )
        return _combine_decode_result(raw, expected_frames=expected_frames)
    raise FFmpegUnavailable(
        "no media backend available to verify full video decoding: install ffmpeg or PyAV"
    )


def require_decodable(
    path: str | Path, *, timeout: float | None = None, expected_frames: int | None = None
) -> DecodeCheck:
    check = full_decode_check(path, timeout=timeout, expected_frames=expected_frames)
    if not check.ok:
        if check.frame_count == 0:
            reason = "media decode produced no frames"
        elif check.timed_out:
            reason = "media decode exceeded the time budget"
        else:
            reason = check.stderr_tail[:400] or "media stream is corrupt"
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR, "Video cannot be fully decoded", details={"reason": reason}
        )
    return check


# ---------------------------------------------------------------------------
# frame timeline + 10 Hz grid selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameTimeline:
    """Presentation timestamps of every decodable video frame (source clock)."""

    pts_ns: tuple[int, ...]
    derived_from_cfr: bool = False
    backend: str = "ffprobe"

    @property
    def frame_count(self) -> int:
        return len(self.pts_ns)

    @property
    def duration_ns(self) -> int:
        if not self.pts_ns:
            return 0
        return max(0, self.pts_ns[-1] - self.pts_ns[0])

    def monotonic_violations(self, limit: int = 20) -> list[int]:
        violations: list[int] = []
        for index in range(1, len(self.pts_ns)):
            if self.pts_ns[index] <= self.pts_ns[index - 1]:
                violations.append(index)
                if len(violations) >= limit:
                    break
        return violations


def _parse_ffprobe_frame_times(stdout: str) -> list[int]:
    values: list[int] = []
    for line in stdout.splitlines():
        token = line.split(",", 1)[0].strip()
        if not token or token.upper() in {"N/A", "NAN"}:
            continue
        try:
            seconds = float(token)
        except ValueError:
            continue
        values.append(int(round(seconds * 1_000_000_000)))
    return values


def _timeline_with_pyav(path: Path, *, timeout: float) -> FrameTimeline:
    import av

    deadline = time.monotonic() + timeout
    values: list[int] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source has no video stream")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if time.monotonic() > deadline:
                    raise ApiError(ErrorCode.TIMEOUT, "frame timestamp extraction exceeded the time budget")
                if frame.pts is None:
                    continue
                time_base = frame.time_base if frame.time_base is not None else stream.time_base
                if time_base is None:
                    continue
                values.append(int(round(float(frame.pts * time_base) * 1_000_000_000)))
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video frame timestamps cannot be read",
            details=_pyav_error_payload(exc, path),
        ) from exc
    if not values:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Video reports no decodable frames")
    return FrameTimeline(pts_ns=tuple(values), backend="pyav")


def frame_timeline(
    path: str | Path,
    *,
    probe_result: VideoProbe | None = None,
    frame_count: int | None = None,
    timeout: float | None = None,
) -> FrameTimeline:
    """Read per-frame presentation timestamps (authoritative PTS, never filenames)."""
    target = Path(path)
    budget = COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
    if ffprobe_available():
        argv = [
            ffprobe_path(),
            "-v",
            "error",
            "-hide_banner",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(target),
        ]
        completed = _run(argv, timeout=budget)
        if completed.returncode != 0:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Video frame timestamps cannot be read",
                details={"reason": _sanitize(completed.stderr.decode("utf-8", "replace"), target)},
            )
        values = _parse_ffprobe_frame_times(completed.stdout.decode("utf-8", "replace"))
        if values:
            return FrameTimeline(pts_ns=tuple(values), backend="ffprobe")
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Video reports no decodable frames")
    if pyav_available():
        return _timeline_with_pyav(target, timeout=budget)
    # -- degraded path: no ffprobe, no PyAV ----------------------------------
    probe_result = probe_result or probe(target, timeout=budget)
    if not probe_result.is_cfr or probe_result.fps is None:
        raise FFmpegUnavailable(
            "ffprobe or PyAV is required to read variable-frame-rate presentation timestamps"
        )
    count = frame_count if frame_count is not None else probe_result.nb_frames
    if not count:
        count = full_decode_check(target, timeout=budget).frame_count
    fps = probe_result.fps
    return FrameTimeline(
        pts_ns=tuple(int(round(index * 1_000_000_000 / fps)) for index in range(count)),
        derived_from_cfr=True,
        backend="cfr-derived",
    )


@dataclass(frozen=True)
class GridSelection:
    """Mapping of the canonical 10 Hz grid onto source frame indexes."""

    source_indexes: list[int]
    grid_t_ns: list[int]
    source_t_ns: list[int]
    abs_errors_ns: list[int]
    missing_grid_t_ns: list[int]
    duplicate_selections: int
    origin_ns: int
    tolerance_ns: int = FRAME_TOLERANCE_NS

    @property
    def sample_count(self) -> int:
        return len(self.source_indexes)

    @property
    def missing_count(self) -> int:
        return len(self.missing_grid_t_ns)

    @property
    def max_abs_error_ns(self) -> int:
        return max(self.abs_errors_ns) if self.abs_errors_ns else 0

    def p95_abs_error_ns(self) -> float:
        if not self.abs_errors_ns:
            return 0.0
        ordered = sorted(self.abs_errors_ns)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return float(ordered[index])

    def alignment_dict(self) -> dict[str, Any]:
        return {
            "grid_points": self.sample_count + self.missing_count,
            "valid_sample_count": self.sample_count,
            "missing_sample_count": self.missing_count,
            "missing_frame_grid_t_ns": self.missing_grid_t_ns[:200],
            "duplicate_selection_count": self.duplicate_selections,
            "max_abs_sync_error_ms": self.max_abs_error_ns / 1e6,
            "p95_abs_sync_error_ms": self.p95_abs_error_ns() / 1e6,
            "origin_ns": self.origin_ns,
            "tolerance_ns": self.tolerance_ns,
        }


def select_grid_frames(
    timeline: FrameTimeline,
    *,
    interval_ns: int = SAMPLE_INTERVAL_NS,
    tolerance_ns: int = FRAME_TOLERANCE_NS,
    max_duration_ns: int = 600 * 1_000_000_000,
) -> GridSelection:
    """Pick the nearest source frame for every 10 Hz grid point.

    Episode-relative ``t=0`` is the first source frame PTS.  A grid point is
    only valid when the nearest frame is within ``tolerance_ns``; otherwise it
    is recorded as missing (a quality finding, never zero-filled).  Ties pick
    the earlier frame, matching :mod:`embodied_dataops.canonical.timebase`.
    """
    import bisect

    pts = list(timeline.pts_ns)
    if not pts:
        return GridSelection([], [], [], [], [], 0, 0, tolerance_ns)
    origin = pts[0]
    relative = [value - origin for value in pts]
    last_k = min(
        int(relative[-1] // interval_ns),
        max(0, max_duration_ns // interval_ns - 1),
    )
    indexes: list[int] = []
    grid: list[int] = []
    source: list[int] = []
    errors: list[int] = []
    missing: list[int] = []
    duplicates = 0
    previous = -1
    for k in range(last_k + 1):
        target = k * interval_ns
        position = bisect.bisect_left(relative, target)
        if position <= 0:
            chosen = 0
        elif position >= len(relative):
            chosen = len(relative) - 1
        else:
            before, after = relative[position - 1], relative[position]
            chosen = position - 1 if (target - before) <= (after - target) else position
        error = abs(relative[chosen] - target)
        if error > tolerance_ns:
            missing.append(target)
            continue
        if chosen == previous:
            duplicates += 1
        previous = chosen
        indexes.append(chosen)
        grid.append(target)
        source.append(relative[chosen])
        errors.append(error)
    return GridSelection(
        source_indexes=indexes,
        grid_t_ns=grid,
        source_t_ns=source,
        abs_errors_ns=errors,
        missing_grid_t_ns=missing,
        duplicate_selections=duplicates,
        origin_ns=origin,
        tolerance_ns=tolerance_ns,
    )


# ---------------------------------------------------------------------------
# raw frame decoding / encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    rgb: np.ndarray

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def _iter_raw_frames_ffmpeg(
    target: Path,
    *,
    probe_result: VideoProbe,
    wanted: set[int] | None,
    fps: float | None,
    timeout: float,
) -> Iterator[DecodedFrame]:
    width, height = probe_result.width, probe_result.height
    frame_size = width * height * 3
    argv = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(target),
        "-map",
        "0:v:0",
        *fps_passthrough_args(),
    ]
    if fps is not None:
        argv += ["-vf", f"fps={fps:g}"]
    argv += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )
    assert process.stdout is not None
    watchdog = _KillOnTimeout(process, timeout)
    completed_normally = False
    try:
        with watchdog:
            index = 0
            while True:
                raw = process.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                if wanted is None or index in wanted:
                    yield DecodedFrame(index=index, rgb=frame.copy())
                index += 1
            completed_normally = True
    finally:
        # An early close (``GeneratorExit``) must kill the child instead of
        # blocking forever on stderr; a normally exhausted decode is allowed to
        # finish on its own.
        if not completed_normally:
            try:
                process.kill()
            except Exception:  # pragma: no cover - already gone
                pass
        try:
            process.stdout.close()
        except Exception:  # pragma: no cover
            pass
        stderr = process.stderr.read() if process.stderr else b""
        returncode = process.wait(timeout=30)
    if completed_normally and returncode != 0:
        if watchdog.expired:
            raise ApiError(ErrorCode.TIMEOUT, "video decode exceeded the time budget")
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video decoding failed",
            details={"reason": _sanitize(stderr.decode("utf-8", "replace"), target)},
        )


def _iter_raw_frames_pyav(
    target: Path,
    *,
    wanted: set[int] | None,
    fps: float | None,
    timeout: float,
) -> Iterator[DecodedFrame]:
    import av

    deadline = time.monotonic() + timeout
    try:
        with av.open(str(target)) as container:
            if not container.streams.video:
                raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source has no video stream")
            stream = container.streams.video[0]
            index = 0
            next_target_ns = 0
            interval_ns = int(1_000_000_000 / fps) if fps else 0
            for frame in container.decode(stream):
                if time.monotonic() > deadline:
                    raise ApiError(ErrorCode.TIMEOUT, "video decode exceeded the time budget")
                if fps:
                    time_base = frame.time_base if frame.time_base is not None else stream.time_base
                    if frame.pts is None or time_base is None:
                        continue
                    current_ns = int(float(frame.pts * time_base) * 1_000_000_000)
                    if current_ns < next_target_ns:
                        continue
                    while next_target_ns <= current_ns:
                        next_target_ns += interval_ns
                rgb = frame.to_ndarray(format="rgb24")
                if wanted is None or index in wanted:
                    yield DecodedFrame(index=index, rgb=rgb)
                index += 1
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Video decoding failed",
            details=_pyav_error_payload(exc, target),
        ) from exc


def iter_raw_frames(
    path: str | Path,
    *,
    probe_result: VideoProbe | None = None,
    frame_indexes: Iterable[int] | None = None,
    fps: float | None = None,
    timeout: float | None = None,
) -> Iterator[DecodedFrame]:
    """Stream RGB24 frames (ffmpeg pipe first, PyAV fallback).

    ``frame_indexes`` filters materialized frames (every frame is still decoded,
    which keeps the check honest); ``fps`` resamples for preview tooling.
    """
    target = Path(path)
    budget = COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
    wanted = set(int(i) for i in frame_indexes) if frame_indexes is not None else None
    if ffmpeg_available():
        probe_result = probe_result or probe(target, timeout=budget)
        yield from _iter_raw_frames_ffmpeg(
            target, probe_result=probe_result, wanted=wanted, fps=fps, timeout=budget
        )
        return
    if pyav_available():
        yield from _iter_raw_frames_pyav(target, wanted=wanted, fps=fps, timeout=budget)
        return
    raise FFmpegUnavailable(
        "no media backend available for frame decoding: install ffmpeg or PyAV"
    )


def iter_frames(
    path: str | Path, *, fps: float | None = None, timeout: float | None = None
) -> Iterator[np.ndarray]:
    """Yield RGB frames (optionally resampled at ``fps``) for preview tooling."""
    for frame in iter_raw_frames(path, fps=fps, timeout=timeout):
        yield frame.rgb


def encode_canvas_frames_h264(
    frames: Iterable[np.ndarray],
    destination: str | Path,
    *,
    fps: int = TARGET_FPS,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
    crf: int = 18,
    timeout: float | None = None,
) -> int:
    """Encode 640x480 RGB frames as an H.264 yuv420p MP4; returns frame count."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg_path(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(max(1, int(fps))),
        # Deterministic encode: one encoder thread and bitexact output so the
        # canonical video bytes (and therefore canonical_sha256) are stable
        # across re-normalizations and fixture regenerations.
        "-threads",
        "1",
        "-flags",
        "+bitexact",
        "-fflags",
        "+bitexact",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        "-r",
        str(int(fps)),
        str(destination),
    ]
    budget = COMMAND_TIMEOUT_SECONDS if timeout is None else timeout
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    written = 0
    watchdog = _KillOnTimeout(process, budget)
    try:
        with watchdog:
            for frame in frames:
                array = np.asarray(frame)
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                if array.ndim != 3 or array.shape[0] != height or array.shape[1] != width:
                    raise ApiError(
                        ErrorCode.INTERNAL_ERROR,
                        "canonical frame does not match the 640x480 RGB canvas",
                        details={"shape": list(array.shape), "expected": [height, width, 3]},
                    )
                process.stdin.write(array.tobytes())
                written += 1
            process.stdin.close()
    except Exception:
        try:
            process.kill()
        except Exception:  # pragma: no cover
            pass
        process.wait(timeout=30)
        raise
    stderr = process.stderr.read() if process.stderr else b""
    returncode = process.wait(timeout=120)
    if returncode != 0:
        if watchdog.expired:
            raise ApiError(ErrorCode.TIMEOUT, "canonical video encoding exceeded the time budget")
        raise ApiError(
            ErrorCode.FILE_DECODE_ERROR,
            "Canonical video encoding failed",
            details={"reason": _sanitize(stderr.decode("utf-8", "replace"))},
        )
    if written == 0:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Canonical video would contain no frames")
    return written


@dataclass(frozen=True)
class LetterboxTranscodeResult:
    frame_count: int
    frame_transforms: list[dict[str, Any]]
    selection: GridSelection
    width: int = CANVAS_WIDTH
    height: int = CANVAS_HEIGHT
    fps: int = TARGET_FPS
    decoded_frame_count: int = 0
    timestamps_derived_from_cfr: bool = False

    @property
    def duration_ns(self) -> int:
        return self.frame_count * SAMPLE_INTERVAL_NS

    def transform_summary(self) -> dict[str, Any]:
        variants: list[dict[str, Any]] = []
        for record in self.frame_transforms:
            if record not in variants:
                variants.append(record)
                if len(variants) >= 10:
                    break
        return {
            "frame_count": len(self.frame_transforms),
            "uniform": len(variants) == 1,
            "variants": variants,
            "frames_with_bars": sum(
                1 for record in self.frame_transforms if record.get("pad_left") or record.get("pad_top")
            ),
        }


def _encode_selected_frames(
    canvas_frames: Iterator[np.ndarray],
    destination: Path,
    *,
    expected_frames: int,
    timeout: float | None,
) -> int:
    written = encode_canvas_frames_h264(canvas_frames, destination, timeout=timeout)
    if written != expected_frames:
        raise ApiError(
            ErrorCode.INTERNAL_ERROR,
            "canonical video frame count does not match the selected 10 Hz samples",
            details={"written": written, "expected": expected_frames},
        )
    return written


def transcode_to_h264_10fps_640x480_letterbox(
    source: str | Path,
    destination: str | Path,
    *,
    tolerance_ns: int = FRAME_TOLERANCE_NS,
    probe_result: VideoProbe | None = None,
    timeline: FrameTimeline | None = None,
    timeout: float | None = None,
) -> LetterboxTranscodeResult:
    """VFR/CFR -> canonical 10 fps 640x480 H.264 with per-frame letterbox records.

    Frames are selected on the 10 Hz grid with a nearest-neighbour tolerance
    (SPEC 5.5); each selected frame is letterboxed independently so a source
    with changing resolution still yields exact per-frame transforms.
    """
    source = Path(source)
    probe_result = probe_result or probe(source, timeout=timeout)
    timeline = timeline or frame_timeline(source, probe_result=probe_result, timeout=timeout)
    violations = timeline.monotonic_violations()
    if violations:
        raise ValueError(f"source presentation timestamps are not monotonic at frame {violations[0]}")
    selection = select_grid_frames(timeline, tolerance_ns=tolerance_ns)
    if selection.sample_count == 0:
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Source has no frames on the 10 Hz grid")
    transforms = _encode_selection(
        source,
        Path(destination),
        selection=selection,
        probe_result=probe_result,
        timeout=timeout,
    )
    return LetterboxTranscodeResult(
        frame_count=len(transforms),
        frame_transforms=transforms,
        selection=selection,
        decoded_frame_count=timeline.frame_count,
        timestamps_derived_from_cfr=timeline.derived_from_cfr,
    )


def _encode_selection(
    source: Path,
    destination: Path,
    *,
    selection: GridSelection,
    probe_result: VideoProbe,
    timeout: float | None,
) -> list[dict[str, Any]]:
    wanted = set(selection.source_indexes)
    occurrences: dict[int, int] = {}
    for index in selection.source_indexes:
        occurrences[index] = occurrences.get(index, 0) + 1
    transforms: list[dict[str, Any]] = []

    def canvas_frames() -> Iterator[np.ndarray]:
        for decoded in iter_raw_frames(
            source, probe_result=probe_result, frame_indexes=wanted, timeout=timeout
        ):
            transform = compute_letterbox(decoded.width, decoded.height)
            canvas = apply_letterbox(decoded.rgb, transform)
            for _ in range(occurrences.get(decoded.index, 0)):
                transforms.append(transform.as_record())
                yield canvas

    _encode_selected_frames(
        canvas_frames(), destination, expected_frames=selection.sample_count, timeout=timeout
    )
    return transforms


def thumbnail(
    source: str | Path,
    destination: str | Path,
    *,
    at_ns: int = 0,
    width: int = 320,
    timeout: float | None = None,
) -> Path:
    """Extract a single RGB JPEG thumbnail (no audio) for previews."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if ffmpeg_available():
        argv = [
            ffmpeg_path(),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0, at_ns) / 1_000_000_000:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={int(width)}:-2",
            "-q:v",
            "3",
            str(destination),
        ]
        completed = _run(argv, timeout=timeout)
        if completed.returncode != 0 or not destination.is_file():
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Thumbnail extraction failed",
                details={"reason": _sanitize(completed.stderr.decode("utf-8", "replace"), source)},
            )
        return destination
    if pyav_available():
        import av
        from PIL import Image

        try:
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                if at_ns > 0 and stream.time_base is not None:
                    container.seek(int(at_ns / 1_000_000_000 / float(stream.time_base)), stream=stream)
                for frame in container.decode(stream):
                    rgb = frame.to_ndarray(format="rgb24")
                    image = Image.fromarray(rgb)
                    image.thumbnail((int(width), int(width) * 4))
                    image.save(destination, format="JPEG", quality=85)
                    return destination
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Thumbnail extraction failed",
                details=_pyav_error_payload(exc, source),
            ) from exc
        raise ApiError(ErrorCode.FILE_DECODE_ERROR, "Thumbnail extraction produced no frame")
    raise FFmpegUnavailable("no media backend available to extract a thumbnail")


def extract_frames(
    source: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 1.0,
    prefix: str = "frame_",
    image_format: str = "jpg",
    timeout: float | None = None,
) -> list[Path]:
    """Extract RGB frames at a fixed sampling rate for previews / detection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a dependency
        raise ApiError(ErrorCode.INTERNAL_ERROR, "Pillow is required to extract frame images") from exc
    paths: list[Path] = []
    for index, rgb in enumerate(iter_frames(source, fps=fps, timeout=timeout)):
        path = output_dir / f"{prefix}{index:06d}.{image_format}"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    return paths


def command_versions(timeout: float = 10.0) -> dict[str, str]:
    """Return discovered binary paths/versions (audit evidence)."""
    versions: dict[str, str] = {}
    for kind, resolver in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)):
        try:
            binary = resolver()
        except ApiError:
            versions[kind] = "unavailable"
            continue
        try:
            completed = _run([binary, "-version"], timeout=timeout)
            first = completed.stdout.decode("utf-8", "replace").splitlines()
            versions[kind] = first[0] if first else binary
        except Exception:  # pragma: no cover - environment dependent
            versions[kind] = binary
    try:
        import av

        versions["pyav"] = f"PyAV {av.__version__}"
    except Exception:  # pragma: no cover - optional dependency
        versions["pyav"] = "unavailable"
    return versions


__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "DecodeCheck",
    "DecodedFrame",
    "FFmpegUnavailable",
    "FRAME_TOLERANCE_NS",
    "FrameTimeline",
    "GridSelection",
    "LetterboxTranscodeResult",
    "VideoProbe",
    "command_versions",
    "encode_canvas_frames_h264",
    "extract_frames",
    "ffmpeg_available",
    "ffmpeg_path",
    "ffprobe_available",
    "ffprobe_path",
    "frame_timeline",
    "full_decode_check",
    "iter_frames",
    "iter_raw_frames",
    "media_backend_summary",
    "probe",
    "pyav_available",
    "require_decodable",
    "reset_binary_cache",
    "select_grid_frames",
    "thumbnail",
    "transcode_to_h264_10fps_640x480_letterbox",
]

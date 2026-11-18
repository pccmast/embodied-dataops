"""Ego MP4 import adapter (SPEC 5.5).

An Ego video is exactly one ``ego_video_episode``: images, timestamps and
annotations only.  This adapter **never** synthesises robot ``action`` or
``state`` -- the canonical metadata is written with ``state_dim=0``,
``action_dim=0`` and ``joint_names=[]`` so a video can never masquerade as a
robot trajectory (SPEC 3.2/5.2).

Validation is deliberately strict: the container must be probeable *and* every
frame must decode (``ffmpeg -f null -``); a truncated MP4 therefore fails with
``FILE_DECODE_ERROR`` even when its header still looks valid (F07).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_dataops import SAMPLE_INTERVAL_NS
from embodied_dataops.canonical.timebase import frame_gap_windows
from embodied_dataops.domain.enums import EpisodeType, Severity, SourceType
from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.imports.base import EpisodeSource, ImportContext, ParsedSource, SourceParser
from embodied_dataops.imports.common import diagnostic, has_structural, merge_diagnostics
from embodied_dataops.processing.ffmpeg import (
    FRAME_TOLERANCE_NS,
    FrameTimeline,
    GridSelection,
    VideoProbe,
    frame_timeline,
    full_decode_check,
    iter_raw_frames,
    probe,
    select_grid_frames,
)

MAX_EGO_DURATION_NS = 600 * 1_000_000_000
MAX_SOURCE_GAP_NS = 200_000_000


@dataclass(frozen=True)
class EgoEpisodeData:
    """One Ego episode ready for canonical normalization."""

    path: Path
    probe_result: VideoProbe
    timeline: FrameTimeline
    selection: GridSelection
    decoded_frame_count: int
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    normalization_valid: bool = True

    @property
    def sample_count(self) -> int:
        return self.selection.sample_count

    @property
    def duration_ns(self) -> int:
        return self.sample_count * SAMPLE_INTERVAL_NS

    @property
    def source_duration_ns(self) -> int:
        return self.timeline.duration_ns

    def alignment(self) -> dict[str, Any]:
        payload = self.selection.alignment_dict()
        payload.update(
            {
                "alignment_unit": "nearest_source_frame",
                "decoded_frame_count": self.decoded_frame_count,
                "timeline_frame_count": self.timeline.frame_count,
                "timestamps_derived_from_cfr": self.timeline.derived_from_cfr,
                "timeline_backend": self.timeline.backend,
                "source_duration_ns": self.source_duration_ns,
                "duration_ns": self.duration_ns,
            }
        )
        return payload

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield source RGB frames in output-grid order (streaming)."""
        indexes = self.selection.source_indexes
        if not indexes:
            return
        occurrences: dict[int, int] = {}
        for index in indexes:
            occurrences[index] = occurrences.get(index, 0) + 1
        for decoded in iter_raw_frames(
            self.path, probe_result=self.probe_result, frame_indexes=set(occurrences)
        ):
            frame = decoded.rgb
            for _ in range(occurrences.get(decoded.index, 0)):
                yield frame


class EgoMp4Parser(SourceParser):
    """Single-file Ego MP4 adapter (one video -> one episode)."""

    source_type = SourceType.EGO_MP4

    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        source = Path(ctx.source_path)
        if not source.is_file():
            raise ApiError(ErrorCode.SOURCE_NOT_FOUND, "Ego video upload object is missing")
        probe_result = probe(source)
        if probe_result.duration_ns > MAX_EGO_DURATION_NS:
            raise ApiError(
                ErrorCode.SOURCE_DURATION_LIMIT,
                "Ego video exceeds the 600 s limit",
                details={
                    "duration_ns": probe_result.duration_ns,
                    "limit_ns": MAX_EGO_DURATION_NS,
                },
            )
        decode_check = full_decode_check(
            source,
            duration_ns=probe_result.duration_ns,
            expected_frames=probe_result.nb_frames,
        )
        if not decode_check.ok:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Ego video cannot be fully decoded",
                details={
                    "decoded_frame_count": decode_check.frame_count,
                    "expected_frame_count": decode_check.expected_frame_count,
                    "reason": decode_check.stderr_tail or "decode produced no frames",
                    "backend": decode_check.backend,
                },
            )
        timeline = frame_timeline(source, probe_result=probe_result)
        violations = timeline.monotonic_violations()
        selection = select_grid_frames(timeline, tolerance_ns=FRAME_TOLERANCE_NS)
        return {
            "format": "ego-mp4",
            "probe": probe_result.as_dict(),
            "decode": decode_check.as_dict(),
            "timeline_backend": timeline.backend,
            "timestamps_derived_from_cfr": timeline.derived_from_cfr,
            "frame_count": timeline.frame_count,
            "timestamp_violations": violations[:20],
            "grid_sample_count": selection.sample_count,
            "missing_grid_sample_count": selection.missing_count,
            "has_audio": probe_result.has_audio,
            "duration_ns": selection.sample_count * SAMPLE_INTERVAL_NS,
        }

    def parse(self, ctx: ImportContext) -> ParsedSource:
        source = Path(ctx.source_path)
        probe_result = probe(source)
        if probe_result.duration_ns > MAX_EGO_DURATION_NS:
            raise ApiError(
                ErrorCode.SOURCE_DURATION_LIMIT,
                "Ego video exceeds the 600 s limit",
                details={"duration_ns": probe_result.duration_ns, "limit_ns": MAX_EGO_DURATION_NS},
            )
        timeline = frame_timeline(source, probe_result=probe_result)
        selection = select_grid_frames(timeline, tolerance_ns=FRAME_TOLERANCE_NS)
        diagnostics = _diagnostics_for(timeline, selection, probe_result)
        normalization_valid = not has_structural(diagnostics)
        duration_ns = selection.sample_count * SAMPLE_INTERVAL_NS
        if selection.sample_count == 0:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Ego video produced no 10 Hz sample points",
                details={"frame_count": timeline.frame_count},
            )
        source_episode = EpisodeSource(
            source_episode_index=0,
            episode_type=EpisodeType.EGO_VIDEO_EPISODE,
            task_label="ego_video",
            duration_ns=duration_ns,
            payload={
                "camera_key": "front",
                "clock_domain": "ego_video_pts_relative",
                "source_fps": probe_result.fps,
                "frame_count": timeline.frame_count,
                "sample_count": selection.sample_count,
                "has_audio": probe_result.has_audio,
                "timeline_backend": timeline.backend,
                "timestamps_derived_from_cfr": timeline.derived_from_cfr,
            },
            diagnostics=diagnostics,
            normalization_valid=normalization_valid,
        )
        metadata = {
            "format": "ego-mp4",
            "probe": probe_result.as_dict(),
            "frame_count": timeline.frame_count,
            "sample_count": selection.sample_count,
            "missing_grid_sample_count": selection.missing_count,
            "timeline_backend": timeline.backend,
        }
        warnings: list[str] = []
        if probe_result.has_audio:
            warnings.append("source audio is preserved on the raw object and stripped from canonical video")
        return ParsedSource(episodes=[source_episode], metadata=metadata, warnings=warnings)

    def read_episode(self, ctx: ImportContext, episode_index: int = 0) -> EgoEpisodeData:
        """Re-read the video with a full decode check for normalization."""
        if episode_index != 0:
            raise ApiError(
                ErrorCode.EPISODE_ENUMERATION_FAILED,
                "an Ego MP4 file contains exactly one episode",
                details={"requested_episode_index": episode_index},
            )
        source = Path(ctx.source_path)
        probe_result = probe(source)
        if probe_result.duration_ns > MAX_EGO_DURATION_NS:
            raise ApiError(
                ErrorCode.SOURCE_DURATION_LIMIT,
                "Ego video exceeds the 600 s limit",
                details={"duration_ns": probe_result.duration_ns, "limit_ns": MAX_EGO_DURATION_NS},
            )
        decode_check = full_decode_check(
            source,
            duration_ns=probe_result.duration_ns,
            expected_frames=probe_result.nb_frames,
        )
        if not decode_check.ok:
            raise ApiError(
                ErrorCode.FILE_DECODE_ERROR,
                "Ego video cannot be fully decoded",
                details={
                    "decoded_frame_count": decode_check.frame_count,
                    "expected_frame_count": decode_check.expected_frame_count,
                    "reason": decode_check.stderr_tail or "decode produced no frames",
                    "backend": decode_check.backend,
                },
            )
        timeline = frame_timeline(source, probe_result=probe_result)
        selection = select_grid_frames(timeline, tolerance_ns=FRAME_TOLERANCE_NS)
        diagnostics = _diagnostics_for(timeline, selection, probe_result)
        normalization_valid = not has_structural(diagnostics) and selection.sample_count > 0
        return EgoEpisodeData(
            path=source,
            probe_result=probe_result,
            timeline=timeline,
            selection=selection,
            decoded_frame_count=decode_check.frame_count,
            diagnostics=merge_diagnostics(diagnostics),
            normalization_valid=normalization_valid,
        )


def _diagnostics_for(
    timeline: FrameTimeline, selection: GridSelection, probe_result: VideoProbe
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    violations = timeline.monotonic_violations()
    if violations:
        diagnostics.append(
            diagnostic(
                "TIMESTAMP_NON_MONOTONIC",
                Severity.ERROR,
                "source presentation timestamps are not strictly increasing",
                details={"violation_indexes": violations[:20], "violation_count": len(violations)},
            )
        )
    gaps = frame_gap_windows(list(timeline.pts_ns), max_gap_ns=MAX_SOURCE_GAP_NS)
    if gaps:
        diagnostics.append(
            diagnostic(
                "FRAME_GAP_EXCESS",
                Severity.WARN,
                "source camera frame gap exceeds 200 ms",
                details={"windows": gaps[:50], "window_count": len(gaps)},
                structural=False,
            )
        )
    if selection.missing_count:
        diagnostics.append(
            diagnostic(
                "SYNC_MISSING_FRAME",
                Severity.ERROR,
                "one or more 10 Hz grid points have no camera frame within 50 ms",
                details={
                    "missing_grid_t_ns": selection.missing_grid_t_ns[:200],
                    "missing_count": selection.missing_count,
                },
            )
        )
    if timeline.frame_count == 0:
        diagnostics.append(
            diagnostic(
                "FILE_DECODE_ERROR",
                Severity.ERROR,
                "video reports no decodable frames",
            )
        )
    # Source audio is preserved on the raw object and stripped from the
    # canonical video (SPEC 5.5); it is recorded as metadata, not as a quality
    # finding, so the QC rule vocabulary stays exactly ``qc-policy/1.0``.
    return diagnostics


__all__ = ["EgoEpisodeData", "EgoMp4Parser", "MAX_EGO_DURATION_NS"]

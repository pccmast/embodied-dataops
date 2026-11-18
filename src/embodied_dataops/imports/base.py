"""Import adapter interface (SPEC 5.1).

An adapter turns one validated source object into zero or more Episode sources.
Adapters are pure parsing: they never write canonical artifacts, never touch
Episode state, and never run quality rules.  They also never guess Episode
boundaries from file names -- LeRobot boundaries come from the official API
metadata, MCAP is restricted to a single Episode, and Ego MP4 is one Episode.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embodied_dataops.domain.enums import DataOrigin, EpisodeType, SourceType


@dataclass
class EpisodeSource:
    """A parsed Episode ready for normalization."""

    source_episode_index: int
    episode_type: EpisodeType
    task_label: str = ""
    duration_ns: int = 0
    #: adapter-specific payload consumed by the matching normalizer
    payload: dict[str, Any] = field(default_factory=dict)
    #: structural problems that make the episode unpublishable but diagnosable
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    normalization_valid: bool = True


@dataclass
class ImportContext:
    """Everything an adapter may read while parsing one source import."""

    import_id: str
    source_type: SourceType
    data_origin: DataOrigin
    source_path: Path
    workdir: Path
    mapping: dict[str, Any] = field(default_factory=dict)
    license: str = ""
    source_uri: str | None = None
    source_revision: str | None = None
    source_sha256: str = ""


@dataclass
class ParsedSource:
    episodes: list[EpisodeSource]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class SourceParser(abc.ABC):
    """Adapter for one supported `source_type`."""

    source_type: SourceType

    @abc.abstractmethod
    def validate(self, ctx: ImportContext) -> dict[str, Any]:
        """Cheap structural validation; raises ApiError on fatal problems."""

    @abc.abstractmethod
    def parse(self, ctx: ImportContext) -> ParsedSource:
        """Enumerate Episodes using the authoritative source metadata."""


def get_parser(source_type: str | SourceType) -> SourceParser:
    """Return the registered parser for a source type."""
    from embodied_dataops.imports.ego_mp4 import EgoMp4Parser
    from embodied_dataops.imports.lerobot_v3 import LerobotV3Parser
    from embodied_dataops.imports.mcap_profile_v1 import McapProfileV1Parser

    registry: dict[SourceType, type[SourceParser]] = {
        SourceType.LEROBOT_V3: LerobotV3Parser,
        SourceType.EGO_MP4: EgoMp4Parser,
        SourceType.MCAP_ROS2_PROFILE_V1: McapProfileV1Parser,
    }
    try:
        from embodied_dataops.imports.mcap_profile_v2 import McapProfileV2Parser

        registry[SourceType.MCAP_ROS2_PROFILE_V2] = McapProfileV2Parser
    except ImportError:
        # The V2 adapter is loaded lazily so a partially implemented tree keeps
        # the V1 imports working; selecting the V2 type then yields UNSUPPORTED_SOURCE_TYPE.
        pass
    if isinstance(source_type, str):
        try:
            source_type = SourceType(source_type)
        except ValueError as exc:
            from embodied_dataops.errors import ApiError, ErrorCode

            raise ApiError(
                ErrorCode.UNSUPPORTED_SOURCE_TYPE, f"Unsupported source_type: {source_type}"
            ) from exc
    return registry[source_type]()

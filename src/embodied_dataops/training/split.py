"""确定性来源分组与 train/val/test 划分（EDO-TRAIN-PREP-1.1 §6）。

两条不能让步的性质：

1. **同一来源不跨档**：同源文件、重复源摘要、同一次连续采集都属于同一个 group，
   整组进同一个 split。按 episode 随机切会把同一场景的近重复样本同时放进训练和验证，
   指标会虚高而没人看得出来。
2. **结果可重算**：``grouped_hash_v1`` 只依赖 ``(strategy, seed, group_id)`` 的摘要，
   与枚举顺序、机器、时间无关；``explicit_group_assignment_v1`` 的结果来自冻结的
   映射文件，且只允许 synthetic-fixture 正常路径使用（published 不得静默启用）。

本模块是纯函数：不读文件、不连数据库。调用方负责把 ``source_groups.json`` /
``group_assignments.json`` 读成 :class:`EpisodeRecord` / ``dict[str, str]``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import (
    SPLIT_NAMES,
    SPLIT_STRATEGY_EXPLICIT,
    SPLIT_STRATEGY_GROUPED_HASH,
    ExplicitAssignmentSplit,
    GroupedHashSplit,
    SplitConfig,
)

__all__ = [
    "EpisodeRecord",
    "SplitResult",
    "SourceGroup",
    "assign_explicit",
    "assign_grouped_hash",
    "build_source_groups",
    "build_split",
    "group_bucket",
]

#: 哈希分桶的算法标识。进摘要，改它就是新算法版本。
_BUCKET_ALGORITHM = "edo-group-bucket-v1"


@dataclass(frozen=True)
class EpisodeRecord:
    """分组的最小输入：episode 身份 + 来源证据。"""

    episode_id: str
    #: 源对象摘要（原始文件 SHA-256）。相同摘要 = 同源，必须同组。
    source_sha256: str
    #: 采集会话标识；有值时同会话的 episode 必须同组。
    session_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    episode_ids: tuple[str, ...]
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "episode_ids": list(self.episode_ids),
            "evidence": dict(sorted(self.evidence.items())),
        }


@dataclass(frozen=True)
class SplitResult:
    strategy: str
    seed: int | None
    ratios: dict[str, float] | None
    by_group: dict[str, str]
    by_episode: dict[str, str]
    counts: dict[str, int]

    @property
    def actual_ratios(self) -> dict[str, float]:
        episodes = sum(self.counts.values())
        if not episodes:
            return {name: 0.0 for name in SPLIT_NAMES}
        return {name: self.counts.get(name, 0) / episodes for name in SPLIT_NAMES}

    def as_document(self) -> dict[str, Any]:
        """``splits.json`` 的内容：全部按稳定 ID 排序，便于逐字节复现。"""
        return {
            "schema_version": "training-splits/1.0",
            "strategy": self.strategy,
            "seed": self.seed,
            "ratios": self.ratios,
            "counts": {name: self.counts.get(name, 0) for name in SPLIT_NAMES},
            "actual_ratios": {name: self.actual_ratios[name] for name in SPLIT_NAMES},
            "groups": dict(sorted(self.by_group.items())),
            "episodes": dict(sorted(self.by_episode.items())),
        }


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Deterministic so the result does not depend on input order.
            first, second = sorted((left_root, right_root))
            self._parent[second] = first


def build_source_groups(records: list[EpisodeRecord] | tuple[EpisodeRecord, ...]) -> tuple[SourceGroup, ...]:
    """把 episode 连成来源分组：同源摘要或同会话即连通。"""
    seen: set[str] = set()
    for record in records:
        if record.episode_id in seen:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "duplicate episode id in the source listing",
                details={"episode_id": record.episode_id},
            )
        seen.add(record.episode_id)
        if not record.source_sha256 or len(record.source_sha256) != 64:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "an episode record needs the source object digest",
                details={"episode_id": record.episode_id},
            )

    union = _UnionFind()
    for record in records:
        union.add(record.episode_id)
    by_digest: dict[str, str] = {}
    by_session: dict[str, str] = {}
    for record in records:
        previous = by_digest.setdefault(record.source_sha256, record.episode_id)
        union.union(previous, record.episode_id)
        if record.session_key:
            previous = by_session.setdefault(record.session_key, record.episode_id)
            union.union(previous, record.episode_id)

    buckets: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        buckets.setdefault(union.find(record.episode_id), []).append(record)

    groups: list[SourceGroup] = []
    for members in buckets.values():
        episode_ids = tuple(sorted(member.episode_id for member in members))
        groups.append(
            SourceGroup(
                group_id=_group_id(episode_ids),
                episode_ids=episode_ids,
                evidence=_group_evidence(members),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _group_id(episode_ids: tuple[str, ...]) -> str:
    """由成员推导的稳定 group id：组一旦合并就换 id，这正是我们希望被看见的。"""
    digest = hashlib.sha256("\n".join(episode_ids).encode("utf-8")).hexdigest()
    return f"g_{digest[:16]}"


def _group_evidence(members: list[EpisodeRecord]) -> dict[str, Any]:
    digests = sorted({member.source_sha256 for member in members})
    sessions = sorted({member.session_key for member in members if member.session_key})
    return {
        "episode_count": len(members),
        "source_digests": digests,
        "session_keys": sessions,
    }


def group_bucket(seed: int, group_id: str) -> float:
    """把一个 group 稳定映射到 ``[0, 1)``。

    公式固定为 ``int(sha256("<algo>|<seed>|<group_id>")[:8], 16) / 2**64``：
    取前 8 字节而不是全部 32 字节，是为了让「换语言重算」也能对上。
    """
    payload = f"{_BUCKET_ALGORITHM}|{int(seed)}|{group_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _cumulative(ratios: dict[str, float]) -> list[tuple[str, float, float]]:
    bounds: list[tuple[str, float, float]] = []
    lower = 0.0
    for name in SPLIT_NAMES:
        share = float(ratios.get(name, 0.0))
        upper = lower + share
        bounds.append((name, lower, upper))
        lower = upper
    return bounds


def assign_grouped_hash(
    groups: tuple[SourceGroup, ...],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    """按 group 哈希分配；结果与枚举顺序无关，且不会拆开任何 group。"""
    bounds = _cumulative(ratios)
    assignment: dict[str, str] = {}
    for group in groups:
        bucket = group_bucket(seed, group.group_id)
        chosen = SPLIT_NAMES[-1]
        for name, lower, upper in bounds:
            # 半开区间 [lower, upper)；bucket == 1.0 不可能出现，但保留兜底分支。
            if lower <= bucket < upper:
                chosen = name
                break
        assignment[group.group_id] = chosen
    return assignment


def assign_explicit(
    groups: tuple[SourceGroup, ...], assignments: dict[str, str]
) -> dict[str, str]:
    """按冻结映射分配：每个组必须恰好出现一次，不允许未知组。"""
    known = {group.group_id for group in groups}
    unknown = set(assignments) - known
    if unknown:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "group_assignments.json references groups that do not exist",
            details={"unknown_groups": sorted(unknown)},
        )
    missing = known - set(assignments)
    if missing:
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            "group_assignments.json does not cover every source group",
            details={"missing_groups": sorted(missing)},
        )
    for group_id, split in assignments.items():
        if split not in SPLIT_NAMES:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "group_assignments.json maps a group to an unknown split",
                details={"group_id": group_id, "split": split},
            )
    return dict(assignments)


def build_split(
    groups: tuple[SourceGroup, ...],
    config: SplitConfig,
    *,
    assignments: dict[str, str] | None = None,
) -> SplitResult:
    """执行划分并检查泄漏/空档；任何异常都返回明确错误码而不是降级。"""
    if not groups:
        raise ApiError(
            ErrorCode.INSUFFICIENT_SOURCE_GROUPS,
            "the source has no episode groups to split",
            details={"group_count": 0},
        )

    required = list(config.require_nonempty)
    if isinstance(config, GroupedHashSplit):
        if assignments is not None:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "grouped_hash_v1 must not read group_assignments.json",
                details={"strategy": SPLIT_STRATEGY_GROUPED_HASH},
            )
        if len(groups) < len(required):
            raise ApiError(
                ErrorCode.INSUFFICIENT_SOURCE_GROUPS,
                "fewer independent source groups than required non-empty splits",
                details={
                    "group_count": len(groups),
                    "required_splits": required,
                    "note": "three or ten groups cannot guarantee a non-empty hash bucket; "
                    "use the dedicated fixture recipe for the normal path",
                },
            )
        by_group = assign_grouped_hash(groups, seed=config.seed, ratios=config.ratios)
        seed: int | None = config.seed
        ratios: dict[str, float] | None = dict(config.ratios)
    elif isinstance(config, ExplicitAssignmentSplit):
        if assignments is None:
            raise ApiError(
                ErrorCode.INVALID_INPUT,
                "explicit_group_assignment_v1 requires group_assignments.json",
                details={"strategy": SPLIT_STRATEGY_EXPLICIT},
            )
        by_group = assign_explicit(groups, assignments)
        seed = None
        ratios = None
    else:  # pragma: no cover - the union is closed
        raise ApiError(
            ErrorCode.INVALID_INPUT,
            f"unsupported split strategy {getattr(config, 'strategy', '?')!r}",
        )

    by_episode: dict[str, str] = {}
    for group in groups:
        for episode_id in group.episode_ids:
            by_episode[episode_id] = by_group[group.group_id]
    by_episode = dict(sorted(by_episode.items()))

    counts = {name: 0 for name in SPLIT_NAMES}
    for split in by_group.values():
        counts[split] = counts.get(split, 0) + 1

    empty = [name for name in required if counts.get(name, 0) == 0]
    if empty:
        raise ApiError(
            ErrorCode.EMPTY_REQUIRED_SPLIT,
            "a required split received no source group",
            details={
                "empty_splits": empty,
                "counts": counts,
                "group_count": len(groups),
                "strategy": config.strategy,
            },
        )

    _check_leakage(groups, by_group)

    return SplitResult(
        strategy=config.strategy,
        seed=seed,
        ratios=ratios,
        by_group=dict(sorted(by_group.items())),
        by_episode=by_episode,
        counts=counts,
    )


def _check_leakage(groups: tuple[SourceGroup, ...], by_group: dict[str, str]) -> None:
    """同一源摘要出现在两个 split 就是泄漏。

    分组阶段已按摘要连通，所以正常情况下这里不可能触发；保留它是为了让「分组实现被改坏」
    变成一条明确错误，而不是一个悄无声息的指标虚高。
    """
    split_by_digest: dict[str, str] = {}
    group_by_digest: dict[str, str] = {}
    for group in groups:
        split = by_group[group.group_id]
        for digest in group.evidence.get("source_digests") or []:
            previous = split_by_digest.setdefault(str(digest), split)
            if previous != split:
                raise ApiError(
                    ErrorCode.GROUP_LEAKAGE,
                    "one source object appears in more than one split",
                    details={
                        "source_sha256": str(digest),
                        "splits": sorted({previous, split}),
                        "groups": sorted(
                            {group_by_digest[str(digest)], group.group_id} - {""}
                        ),
                    },
                )
            group_by_digest.setdefault(str(digest), group.group_id)

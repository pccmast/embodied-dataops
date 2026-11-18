"""来源分组与 train/val/test 划分（EDO-TRAIN-PREP-1.1 §6）。

两条性质必须被钉死：同一来源不跨档、结果可重算。第三条同样重要——**哈希空档时
必须失败，不能换 seed 凑结果**，所以这里既有一个正常路径的 golden，也有一个
"真实会空档"的负例。
"""

from __future__ import annotations

import pytest

from embodied_dataops.errors import ApiError, ErrorCode
from embodied_dataops.training.contract import (
    ExplicitAssignmentSplit,
    GroupedHashSplit,
    parse_recipe,
)
from embodied_dataops.training.split import (
    EpisodeRecord,
    SourceGroup,
    assign_grouped_hash,
    build_source_groups,
    build_split,
    group_bucket,
)

pytestmark = pytest.mark.no_db

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _records(count: int = 12) -> list[EpisodeRecord]:
    return [
        EpisodeRecord(episode_id=f"ep{index:02d}", source_sha256=f"{index:064x}")
        for index in range(count)
    ]


def _grouped(seed: int = 20261035) -> GroupedHashSplit:
    recipe = parse_recipe(
        {
            "target": {},
            "features": {},
            "temporal": {"chunk_size": 16, "n_action_steps": 16, "stride": 1},
            "split": {
                "strategy": "grouped_hash_v1",
                "seed": seed,
                "ratios": RATIOS,
                "group_mapping": "source_groups.json",
            },
        }
    )
    return recipe.split


def _explicit(required: tuple[str, ...] = ("train", "val", "test")) -> ExplicitAssignmentSplit:
    recipe = parse_recipe(
        {
            "target": {},
            "features": {},
            "temporal": {"chunk_size": 16, "n_action_steps": 16, "stride": 1},
            "split": {
                "strategy": "explicit_group_assignment_v1",
                "group_mapping": "source_groups.json",
                "assignments": "group_assignments.json",
                "require_nonempty": list(required),
            },
        }
    )
    return recipe.split


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
def test_the_same_source_digest_lands_in_one_group():
    groups = build_source_groups(
        [
            EpisodeRecord("ep-a", "d" * 64),
            EpisodeRecord("ep-b", "d" * 64),
            EpisodeRecord("ep-c", "e" * 64),
        ]
    )
    assert len(groups) == 2
    merged = next(group for group in groups if "ep-a" in group.episode_ids)
    assert merged.episode_ids == ("ep-a", "ep-b")
    assert merged.evidence["source_digests"] == ["d" * 64]


def test_the_same_session_key_lands_in_one_group():
    groups = build_source_groups(
        [
            EpisodeRecord("ep-a", "a" * 64, session_key="s1"),
            EpisodeRecord("ep-b", "b" * 64, session_key="s1"),
        ]
    )
    assert len(groups) == 1
    assert groups[0].episode_ids == ("ep-a", "ep-b")
    assert groups[0].evidence["session_keys"] == ["s1"]


def test_independent_episodes_stay_separate():
    groups = build_source_groups(_records(3))
    assert len(groups) == 3


def test_group_ids_do_not_depend_on_input_order():
    forward = build_source_groups(_records(5))
    backward = build_source_groups(list(reversed(_records(5))))
    assert [group.group_id for group in forward] == [group.group_id for group in backward]


def test_a_duplicate_episode_id_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_source_groups([EpisodeRecord("ep-a", "a" * 64), EpisodeRecord("ep-a", "b" * 64)])
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_a_record_without_a_source_digest_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_source_groups([EpisodeRecord("ep-a", "")])
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# hash assignment
# ---------------------------------------------------------------------------
def test_group_bucket_is_pinned():
    """换语言/换机器重算必须得到同一个数；改了公式就是新算法版本。"""
    assert round(group_bucket(20261031, "g_" + "0" * 16), 12) == 0.843010988869


def test_grouped_hash_assignment_is_pinned():
    groups = build_source_groups(_records(12))
    result = build_split(groups, _grouped(20261035))
    assert result.counts == {"train": 8, "val": 3, "test": 1}
    assert result.by_episode == {
        "ep00": "train",
        "ep01": "train",
        "ep02": "train",
        "ep03": "train",
        "ep04": "train",
        "ep05": "val",
        "ep06": "train",
        "ep07": "train",
        "ep08": "val",
        "ep09": "train",
        "ep10": "test",
        "ep11": "val",
    }


def test_grouped_hash_assignment_is_order_invariant():
    forward = build_split(build_source_groups(_records(12)), _grouped(20261035))
    backward = build_split(
        build_source_groups(list(reversed(_records(12)))), _grouped(20261035)
    )
    assert forward.by_episode == backward.by_episode


def test_a_different_seed_moves_at_least_one_group():
    """纯函数层面验证 seed 敏感；不经过 build_split，避免被"空档即失败"干扰。"""
    groups = build_source_groups(_records(12))
    first = assign_grouped_hash(groups, seed=20261035, ratios=RATIOS)
    second = assign_grouped_hash(groups, seed=20261036, ratios=RATIOS)
    assert first != second


def test_no_group_is_split_across_two_splits():
    groups = build_source_groups(_records(12))
    result = build_split(groups, _grouped(20261035))
    for group in groups:
        owner = result.by_group[group.group_id]
        assert {result.by_episode[episode_id] for episode_id in group.episode_ids} == {owner}


# ---------------------------------------------------------------------------
# refusals: this is the point of the spec, not an edge case
# ---------------------------------------------------------------------------
def test_an_empty_hash_bucket_is_refused_rather_than_reseeded():
    """seed=20261031 在这组 12 个 episode 上确实会让一个必需档为空。

    规格要求明确失败、不反复换 seed 凑结果；这条测试就是那个真实例子的证据。
    """
    groups = build_source_groups(_records(12))
    with pytest.raises(ApiError) as excinfo:
        build_split(groups, _grouped(20261031))
    assert excinfo.value.code is ErrorCode.EMPTY_REQUIRED_SPLIT
    assert excinfo.value.details["strategy"] == "grouped_hash_v1"


def test_fewer_groups_than_required_splits_is_refused():
    groups = build_source_groups(_records(2))
    with pytest.raises(ApiError) as excinfo:
        build_split(groups, _grouped())
    assert excinfo.value.code is ErrorCode.INSUFFICIENT_SOURCE_GROUPS


def test_no_groups_at_all_is_refused():
    with pytest.raises(ApiError) as excinfo:
        build_split((), _grouped())
    assert excinfo.value.code is ErrorCode.INSUFFICIENT_SOURCE_GROUPS


def test_a_source_digest_crossing_two_splits_is_leakage():
    """分组实现被改坏时，必须报 GROUP_LEAKAGE 而不是让指标虚高。"""
    shared = {"source_digests": ["d" * 64], "episode_count": 1, "session_keys": []}
    groups = (
        SourceGroup("g_one", ("ep-a",), shared),
        SourceGroup("g_two", ("ep-b",), dict(shared)),
    )
    with pytest.raises(ApiError) as excinfo:
        build_split(
            groups,
            _explicit(("train", "val")),
            assignments={"g_one": "train", "g_two": "val"},
        )
    assert excinfo.value.code is ErrorCode.GROUP_LEAKAGE


# ---------------------------------------------------------------------------
# explicit assignment: the fixture normal path
# ---------------------------------------------------------------------------
def test_explicit_assignment_covers_three_groups():
    groups = build_source_groups(_records(3))
    assignments = {groups[0].group_id: "train", groups[1].group_id: "val", groups[2].group_id: "test"}

    result = build_split(groups, _explicit(), assignments=assignments)

    assert result.counts == {"train": 1, "val": 1, "test": 1}
    assert result.seed is None and result.ratios is None
    assert set(result.by_episode.values()) == {"train", "val", "test"}


def test_explicit_assignment_requires_every_group():
    groups = build_source_groups(_records(3))
    with pytest.raises(ApiError) as excinfo:
        build_split(groups, _explicit(), assignments={groups[0].group_id: "train"})
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert "missing_groups" in str(excinfo.value.details)


def test_explicit_assignment_rejects_unknown_groups():
    groups = build_source_groups(_records(1))
    with pytest.raises(ApiError) as excinfo:
        build_split(
            groups,
            _explicit(),
            assignments={groups[0].group_id: "train", "g_unknown": "val"},
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_explicit_assignment_rejects_an_unknown_split_name():
    groups = build_source_groups(_records(1))
    with pytest.raises(ApiError) as excinfo:
        build_split(groups, _explicit(), assignments={groups[0].group_id: "holdout"})
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_explicit_assignment_still_refuses_an_empty_required_split():
    groups = build_source_groups(_records(2))
    assignments = {groups[0].group_id: "train", groups[1].group_id: "val"}
    with pytest.raises(ApiError) as excinfo:
        build_split(groups, _explicit(), assignments=assignments)
    assert excinfo.value.code is ErrorCode.EMPTY_REQUIRED_SPLIT
    assert excinfo.value.details["empty_splits"] == ["test"]


def test_grouped_hash_never_reads_assignments():
    groups = build_source_groups(_records(12))
    with pytest.raises(ApiError) as excinfo:
        build_split(
            groups, _grouped(20261035), assignments={groups[0].group_id: "test"}
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


# ---------------------------------------------------------------------------
# frozen document
# ---------------------------------------------------------------------------
def test_split_document_is_sorted_and_reproducible():
    groups = build_source_groups(_records(12))
    first = build_split(groups, _grouped(20261035)).as_document()
    second = build_split(groups, _grouped(20261035)).as_document()
    assert first == second
    assert list(first["groups"]) == sorted(first["groups"])
    assert list(first["episodes"]) == sorted(first["episodes"])
    assert first["counts"] == {"train": 8, "val": 3, "test": 1}
    # 实际比例不等于名义比例，必须如实报告。
    assert first["ratios"] == RATIOS
    assert round(first["actual_ratios"]["train"], 3) == 0.667

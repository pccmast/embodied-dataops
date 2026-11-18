"""PreparedACTDataset（§9）：范围受限、身份可证、padding 不容忍。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit, pytest.mark.no_db]

from embodied_dataops.errors import ApiError, ErrorCode  # noqa: E402
from embodied_dataops.training.contract import (  # noqa: E402
    VerifiedSource,
    parse_recipe,
    resolve_recipe,
)
from embodied_dataops.training.dataset import (  # noqa: E402
    PreparedACTDataset,
    build_reader_index,
    official_image_key,
)
from embodied_dataops.training.windows import Window  # noqa: E402

CHUNK = 2
STATE_DIM = 2
ACTION_DIM = 2


def _resolved(*, cameras: tuple[str, ...] = ("front",), image_shape=(3, 4, 4), fps: int = 10):
    recipe = parse_recipe(
        {
            "target": {},
            "features": {},
            "temporal": {"chunk_size": CHUNK, "n_action_steps": CHUNK, "stride": 1},
            "split": {
                "strategy": "explicit_group_assignment_v1",
                "group_mapping": "g.json",
                "assignments": "a.json",
                "require_nonempty": ["train"],
            },
            "normalization": {"modes": {"VISUAL": "IDENTITY"}},
        }
    )
    source = VerifiedSource(
        dataset_version_id="dv-1",
        manifest_sha256="a" * 64,
        profile_id="arm6_joint_position_v1",
        profile_revision=1,
        camera_keys=cameras,
        aligned_fps=fps,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        state_names=("j0", "j1"),
        action_names=("j0", "j1"),
        state_unit="rad",
        action_unit="rad",
        action_semantics="target_joint_position",
        image_shapes={camera: tuple(image_shape) for camera in cameras},
    )
    return resolve_recipe(recipe, source)


class _FakeHf:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def select_columns(self, names: list[str]) -> dict[str, list]:
        return {name: [row[name] for row in self._rows] for name in names}


class _FakeReader:
    """最小官方 Reader 替身：只提供 __getitem__ 与 hf_dataset 两列。

    ``columns`` 允许与 ``items`` 不一致：这正是"索引表说有、返回的帧却是另一帧"
    这种真实故障的最小复现方式。
    """

    def __init__(self, items: list[dict], columns: list[dict] | None = None) -> None:
        self._items = items
        rows = columns or [
            {
                "index": item["index"],
                "episode_index": item["episode_index"],
                "frame_index": item["frame_index"],
            }
            for item in items
        ]
        self.hf_dataset = _FakeHf(rows)

    def __getitem__(self, position: int) -> dict:
        return dict(self._items[position])


def _item(*, index: int, episode: int, frame: int, pad: tuple[bool, bool] = (False, False)) -> dict:
    return {
        "index": index,
        "episode_index": episode,
        "frame_index": frame,
        "observation.state": torch.zeros(STATE_DIM),
        "action": torch.zeros(CHUNK, ACTION_DIM),
        "action_is_pad": torch.tensor(pad),
        "observation.images.front": torch.zeros(3, 4, 4),
    }


def _window(
    *,
    split: str = "train",
    episode_id: str = "ep-0",
    official: int = 0,
    anchor: int = 0,
    chunk: int = CHUNK,
) -> Window:
    return Window(
        split=split,
        source_episode_id=episode_id,
        official_episode_index=official,
        anchor_frame_index=anchor,
        action_frame_indices=tuple(range(anchor, anchor + chunk)),
        source_row_digest="d" * 64,
    )


def _dataset(windows, items, *, resolved=None, split="train") -> PreparedACTDataset:
    reader = _FakeReader(items)
    return PreparedACTDataset(
        split=split,
        windows=windows,
        reader=reader,
        index=build_reader_index(reader),
        resolved=resolved or _resolved(),
    )


# ---------------------------------------------------------------------------
# 正例
# ---------------------------------------------------------------------------
def test_length_is_the_frozen_window_count_not_the_reader_length():
    dataset = _dataset([_window(anchor=0)], [_item(index=0, episode=0, frame=0)])
    assert len(dataset) == 1


def test_getitem_returns_the_act_batch_fields():
    dataset = _dataset([_window(anchor=0)], [_item(index=0, episode=0, frame=0)])
    batch = dataset[0]
    assert set(batch) >= {
        "observation.state",
        "observation.images.front",
        "action",
        "action_is_pad",
        "index",
        "episode_index",
        "frame_index",
    }
    assert tuple(batch["observation.state"].shape) == (STATE_DIM,)
    assert tuple(batch["action"].shape) == (CHUNK, ACTION_DIM)
    assert tuple(batch["observation.images.front"].shape) == (3, 4, 4)
    assert batch["action_is_pad"].dtype == torch.bool


def test_uint8_images_are_converted_to_float01():
    item = _item(index=0, episode=0, frame=0)
    item["observation.images.front"] = torch.full((3, 4, 4), 255, dtype=torch.uint8)
    dataset = _dataset([_window()], [item])
    image = dataset[0]["observation.images.front"]
    assert image.dtype == torch.float32
    assert float(image.max()) == pytest.approx(1.0)


def test_camera_names_are_namespaced_like_lerobot():
    assert official_image_key("front") == "observation.images.front"
    assert official_image_key("observation.images.front") == "observation.images.front"


def test_window_at_rejects_out_of_range_positions():
    dataset = _dataset([_window()], [_item(index=0, episode=0, frame=0)])
    with pytest.raises(IndexError):
        dataset.window_at(1)


# ---------------------------------------------------------------------------
# 负例
# ---------------------------------------------------------------------------
def test_a_window_from_another_split_is_refused():
    reader = _FakeReader([_item(index=0, episode=0, frame=0)])
    with pytest.raises(ApiError) as excinfo:
        PreparedACTDataset(
            split="train",
            windows=[_window(split="val")],
            reader=reader,
            index=build_reader_index(reader),
            resolved=_resolved(),
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["foreign_splits"] == ["val"]


def test_reader_identity_mismatch_is_refused():
    """索引表指向 (0,0)，返回的却是 frame 7：必须以身份不符失败。"""
    reader = _FakeReader(
        [_item(index=0, episode=0, frame=7)],
        columns=[{"index": 0, "episode_index": 0, "frame_index": 0}],
    )
    dataset = PreparedACTDataset(
        split="train",
        windows=[_window(anchor=0)],
        reader=reader,
        index=build_reader_index(reader),
        resolved=_resolved(),
    )
    with pytest.raises(ApiError) as excinfo:
        dataset[0]
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["observed"]["frame_index"] == 7
    assert excinfo.value.details["expected"]["frame_index"] == 0


def test_padded_labels_are_refused_under_tail_policy_drop():
    dataset = _dataset([_window()], [_item(index=0, episode=0, frame=0, pad=(False, True))])
    with pytest.raises(ApiError) as excinfo:
        dataset[0]
    assert excinfo.value.code is ErrorCode.EMPTY_REQUIRED_WINDOWS
    assert "padding" in excinfo.value.message


def test_a_window_that_does_not_cover_one_chunk_is_refused_at_construction():
    bad = Window(
        split="train",
        source_episode_id="ep-0",
        official_episode_index=0,
        anchor_frame_index=4,
        action_frame_indices=(4, 6),
        source_row_digest="d" * 64,
    )
    reader = _FakeReader([_item(index=0, episode=0, frame=0)])
    with pytest.raises(ApiError) as excinfo:
        PreparedACTDataset(
            split="train",
            windows=[bad],
            reader=reader,
            index=build_reader_index(reader),
            resolved=_resolved(),
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_a_window_without_a_reader_frame_is_refused():
    dataset = _dataset([_window(anchor=5)], [_item(index=0, episode=0, frame=0)])
    with pytest.raises(ApiError) as excinfo:
        dataset[0]
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["frame_index"] == 5


def test_an_action_chunk_with_the_wrong_shape_is_refused():
    item = _item(index=0, episode=0, frame=0)
    item["action"] = torch.zeros(CHUNK, ACTION_DIM + 1)
    dataset = _dataset([_window()], [item])
    with pytest.raises(ApiError) as excinfo:
        dataset[0]
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE


def test_an_image_that_contradicts_the_resolved_shape_is_refused():
    item = _item(index=0, episode=0, frame=0)
    item["observation.images.front"] = torch.zeros(3, 8, 8)
    dataset = _dataset([_window()], [item])
    with pytest.raises(ApiError) as excinfo:
        dataset[0]
    assert excinfo.value.code is ErrorCode.FEATURE_INCOMPATIBLE
    assert excinfo.value.details["expected_shape"] == [3, 4, 4]

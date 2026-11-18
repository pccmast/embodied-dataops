"""Reusable V2 M5 feedback fixtures (imported by tests/v2 integration modules).

`tests/v2/conftest.py` puts this directory on sys.path, so the modules use
``from feedback_fixtures import FeedbackFactory`` and expose a local ``feedback``
fixture.  This keeps the D2-M5 helpers additive alongside the D2-M1 helper.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal, generate_api_key
from embodied_dataops.db.models import (
    AnnotationSet,
    Asset,
    CollectionOrder,
    Dataset,
    DatasetVersion,
    DatasetVersionEpisode,
    Episode,
    PipelineRun,
    QualityReport,
    ScenarioTag,
    SourceImport,
)
from embodied_dataops.domain.enums import Role
from embodied_dataops.feedback.service import analyze_gaps, create_evaluation
from embodied_dataops.ids import new_id
from embodied_dataops.storage.base import LocalStorage

FIXTURE_SHA = "b" * 64


class FeedbackFactory:
    def __init__(self, db: Session, storage: LocalStorage) -> None:
        self.db = db
        self.storage = storage
        self._principals: dict[str, Principal] = {}

    def principal(self, role: str = "reviewer", name: str | None = None) -> Principal:
        key = name or role
        if key not in self._principals:
            _, record = generate_api_key(Role(role), name=f"v2-{key}")
            self.db.add(record)
            self.db.flush()
            self._principals[key] = Principal(
                key_id=record.id, role=Role(role), name=record.name
            )
        return self._principals[key]

    def episode(
        self,
        *,
        scenario_key: str | None = None,
        basis: str = "HUMAN_VERIFIED",
        quality_status: str = "PASS",
        approved_annotation: bool = True,
        collection_order_id: str | None = None,
        episode_type: str = "robot_episode",
        task_label: str = "pick up the transparent cup",
        business_revision: int = 1,
    ) -> dict[str, Any]:
        episode_id = new_id()
        source_sha = uuid.uuid4().hex + uuid.uuid4().hex
        asset = Asset(
            id=new_id(),
            sha256=source_sha[:64],
            byte_size=1024,
            media_type="application/gzip",
            object_key=f"raw/sha256/{source_sha[:64]}",
            origin="raw",
            logical_name="v2-fixture.tar.gz",
            meta_json={"data_origin": "synthetic"},
        )
        self.db.add(asset)
        self.db.flush()
        source = SourceImport(
            id=new_id(),
            source_type="LEROBOT_V3",
            upload_state="READY",
            asset_id=asset.id,
            data_origin="synthetic",
            license="CC-BY-4.0",
            filename="v2-fixture.tar.gz",
            size_bytes=1024,
            sha256=source_sha[:64],
            mapping_json={},
            collection_order_id=collection_order_id,
        )
        self.db.add(source)
        self.db.flush()
        episode = Episode(
            id=episode_id,
            source_import_id=source.id,
            source_episode_index=0,
            episode_type=episode_type,
            task_label=task_label,
            duration_ns=1_000_000_000,
            status="READY_FOR_REVIEW",
            canonical_version=1,
            canonical_sha256="c" * 64,
            quality_status=quality_status,
            sample_count=10,
            data_origin="synthetic",
            business_revision=business_revision,
        )
        self.db.add(episode)
        self.db.flush()
        run = PipelineRun(
            id=new_id(), episode_id=episode.id, pipeline_version="2.0.0", state="SUCCEEDED"
        )
        self.db.add(run)
        self.db.flush()
        report = QualityReport(
            id=new_id(),
            episode_id=episode.id,
            run_id=run.id,
            policy_version="qc-policy/2.0",
            status=quality_status,
            summary_json={"verdict": quality_status, "findings": []},
        )
        self.db.add(report)
        annotation = AnnotationSet(
            id=new_id(),
            episode_id=episode.id,
            run_id=run.id,
            model_id="yolo11n",
            model_sha256="d" * 64,
            status="APPROVED" if approved_annotation else "PENDING_REVIEW",
            summary_json={},
        )
        self.db.add(annotation)
        if scenario_key is not None:
            self.db.add(
                ScenarioTag(
                    id=new_id(),
                    episode_id=episode.id,
                    scenario_key=scenario_key,
                    basis=basis,
                )
            )
        self.db.flush()
        return {
            "episode_id": episode.id,
            "source_import_id": source.id,
            "quality_report_id": report.id,
            "annotation_set_id": annotation.id,
            "episode": episode,
            "source": source,
        }

    def version(
        self,
        members: list[dict[str, Any]],
        *,
        state: str = "PUBLISHED",
        slug: str | None = None,
        manifest_sha256: str = FIXTURE_SHA,
    ) -> dict[str, Any]:
        dataset = Dataset(
            id=new_id(),
            slug=slug or f"v2-dataset-{uuid.uuid4().hex[:12]}",
            description="V2 feedback fixture",
        )
        self.db.add(dataset)
        self.db.flush()
        version = DatasetVersion(
            id=new_id(),
            dataset_id=dataset.id,
            version=1,
            state=state,
            manifest_sha256=manifest_sha256,
            description="v2 fixture",
        )
        self.db.add(version)
        self.db.flush()
        for position, member in enumerate(members):
            self.db.add(
                DatasetVersionEpisode(
                    version_id=version.id,
                    episode_id=member["episode_id"],
                    canonical_digest="c" * 64,
                    quality_report_id=member["quality_report_id"],
                    annotation_set_id=member["annotation_set_id"],
                    position=position,
                )
            )
        self.db.flush()
        return {
            "dataset_id": dataset.id,
            "version_id": version.id,
            "version": version,
            "dataset": dataset,
        }

    def evaluation(
        self,
        version_id: str,
        results: list[dict[str, Any]],
        *,
        is_simulated: bool = True,
        actor: Principal | None = None,
    ):
        actor = actor or self.principal("operator")
        return create_evaluation(
            self.db,
            dataset_version_id=version_id,
            payload={
                "schema_version": "evaluation/1.0",
                "model_id": "demo-policy-v1",
                "results": results,
                "is_simulated": is_simulated,
            },
            actor=actor,
        )

    def orders(self, evaluation_id: str, *, actor: Principal | None = None) -> list[CollectionOrder]:
        actor = actor or self.principal("operator")
        analyze_gaps(
            self.db,
            evaluation_id=evaluation_id,
            success_rate_threshold=0.8,
            min_trials=20,
            actor=actor,
        )
        return list(
            self.db.scalars(
                select(CollectionOrder)
                .where(CollectionOrder.evaluation_report_id == evaluation_id)
                .order_by(CollectionOrder.scenario_key)
            )
        )

    def order(
        self, evaluation_id: str, scenario_key: str, *, actor: Principal | None = None
    ) -> CollectionOrder:
        for order in self.orders(evaluation_id, actor=actor):
            if order.scenario_key == scenario_key:
                return order
        raise AssertionError(f"no collection order for {scenario_key!r}")

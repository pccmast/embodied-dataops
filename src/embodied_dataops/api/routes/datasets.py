"""Dataset and dataset-version routes (SPEC 9.4, 10)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import Field
from sqlalchemy.orm import Session

from embodied_dataops.api.auth import Principal
from embodied_dataops.api.deps import get_session, request_id_of, run_idempotent
from embodied_dataops.api.schemas.common import StrictModel
from embodied_dataops.datasets import service as dataset_service

router = APIRouter(prefix="/api/v1", tags=["datasets"])


def _principal(request: Request) -> Principal:
    return request.state.principal


class CreateDatasetRequest(StrictModel):
    slug: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = Field(default="", max_length=2000)


class CreateVersionRequest(StrictModel):
    episode_ids: list[str] = Field(min_length=1, max_length=10_000)
    description: str = Field(default="", max_length=2000)


class MembershipRequest(StrictModel):
    add_episode_ids: list[str] = Field(default_factory=list, max_length=10_000)
    remove_episode_ids: list[str] = Field(default_factory=list, max_length=10_000)


@router.post("/datasets", status_code=201)
async def create_dataset(
    request: Request, payload: CreateDatasetRequest, db: Session = Depends(get_session)
) -> JSONResponse:
    principal = _principal(request)
    principal.require("dataset:write")
    dataset = dataset_service.create_dataset(
        db,
        slug=payload.slug,
        description=payload.description,
        actor=principal,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        status_code=201,
        content={
            "data": {"dataset_id": dataset.id, "slug": dataset.slug, "description": dataset.description},
            "request_id": request_id_of(request),
        },
    )


@router.post("/datasets/{dataset_id}/versions", status_code=202)
async def create_version(
    request: Request,
    dataset_id: str,
    payload: CreateVersionRequest,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("dataset:write")
    rid = request_id_of(request)

    def handler():
        version = dataset_service.create_version(
            db,
            dataset_id=dataset_id,
            episode_ids=payload.episode_ids,
            description=payload.description,
            actor=principal,
            request_id=rid,
        )
        db.flush()
        return 202, {
            "version_id": version.id,
            "state": version.state,
            "version": int(version.version),
        }

    if idempotency_key:
        status, body, _ = run_idempotent(
            db,
            principal=principal,
            route="/api/v1/datasets/{dataset_id}/versions",
            idempotency_key=idempotency_key,
            request_payload={"dataset_id": dataset_id, **payload.model_dump(mode="json")},
            handler=handler,
        )
    else:
        status, body = handler()
    return JSONResponse(status_code=status, content={"data": body, "request_id": rid})


@router.get("/datasets/{dataset_id}/versions/{version}")
async def get_version(
    request: Request, dataset_id: str, version: int, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    return JSONResponse(
        content={
            "data": dataset_service.get_version(db, dataset_id, version),
            "request_id": request_id_of(request),
        }
    )


@router.get("/datasets/{dataset_id}/versions")
async def list_versions(
    request: Request, dataset_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    _principal(request).require("read")
    rows = dataset_service.list_versions(db, dataset_id)
    return JSONResponse(
        content={
            "data": [
                {
                    "version_id": row.id,
                    "version": int(row.version),
                    "state": row.state,
                    "manifest_sha256": row.manifest_sha256,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "published_at": row.published_at.isoformat() if row.published_at else None,
                }
                for row in rows
            ],
            "request_id": request_id_of(request),
        }
    )


@router.post("/datasets/{dataset_id}/versions/{version}/membership")
async def modify_membership(
    request: Request,
    dataset_id: str,
    version: int,
    payload: MembershipRequest,
    db: Session = Depends(get_session),
) -> JSONResponse:
    principal = _principal(request)
    principal.require("dataset:write")
    row = dataset_service.get_version(db, dataset_id, version)
    updated = dataset_service.modify_version_membership(
        db,
        version_id=row["id"],
        add_episode_ids=payload.add_episode_ids,
        remove_episode_ids=payload.remove_episode_ids,
        actor=principal,
        request_id=request_id_of(request),
    )
    return JSONResponse(
        content={
            "data": {"version_id": updated.id, "state": updated.state},
            "request_id": request_id_of(request),
        }
    )


@router.post("/dataset-versions/{version_id}/publish")
async def publish(
    request: Request, version_id: str, db: Session = Depends(get_session)
) -> JSONResponse:
    principal = _principal(request)
    principal.require("publish")
    version = dataset_service.publish_version(
        db, version_id=version_id, actor=principal, request_id=request_id_of(request)
    )
    return JSONResponse(
        content={
            "data": {
                "version_id": version.id,
                "state": version.state,
                "manifest_sha256": version.manifest_sha256,
                "published_at": version.published_at.isoformat() if version.published_at else None,
            },
            "request_id": request_id_of(request),
        }
    )


class CreateEgoBundleRequest(StrictModel):
    """V2.1 §14.3: an ego deliverable is a bundle, never a LeRobot trajectory."""

    episode_ids: list[str] = Field(min_length=1, max_length=10_000)
    description: str = Field(default="", max_length=2000)
    licence_allows_video: bool = False


@router.post("/datasets/{dataset_id}/ego-bundle-versions", status_code=201)
async def create_ego_bundle_version(
    request: Request,
    dataset_id: str,
    payload: CreateEgoBundleRequest,
    db: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    """Create a READY ``EGO_ANNOTATION_BUNDLE`` version from approved ego episodes."""
    from embodied_dataops.datasets.ego_bundle import BundleError, create_ego_bundle_version
    from embodied_dataops.errors import ApiError, ErrorCode
    from embodied_dataops.storage.base import get_storage

    principal = _principal(request)
    principal.require("dataset:write")

    def _apply() -> tuple[int, dict]:
        try:
            data = create_ego_bundle_version(
                db,
                dataset_id=dataset_id,
                episode_ids=payload.episode_ids,
                storage=get_storage(),
                description=payload.description,
                licence_allows_video=payload.licence_allows_video,
                actor=principal,
            )
        except BundleError as exc:
            raise ApiError(ErrorCode.INVALID_INPUT, str(exc)[:300]) from exc
        return 201, data

    status, data, _replayed = run_idempotent(
        db,
        principal=principal,
        route=f"POST /api/v1/datasets/{dataset_id}/ego-bundle-versions",
        idempotency_key=idempotency_key,
        request_payload=payload.model_dump(mode="json"),
        handler=_apply,
    )
    return JSONResponse(
        status_code=status, content={"data": data, "request_id": request_id_of(request)}
    )

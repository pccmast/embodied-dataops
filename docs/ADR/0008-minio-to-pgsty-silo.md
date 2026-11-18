# ADR-0008: 对象存储镜像改用 PGSTY Silo（MinIO 官方镜像已停止发布）

Status: accepted (2026-10-29)
Supersedes: the MinIO image choice in `docker-compose.yml` / `infra/versions.lock` only.
Does NOT change the application contract: still S3 API + presigned URLs + MinIO-style
root credentials.

## Context

MinIO 官方在 2025-10 之后停止在 Docker Hub 发布官方镜像，并在 2026-09 前后将
`minio/minio` 官方仓库从 Docker Hub 下架。本机实测：

- `docker pull minio/minio:RELEASE.2025-09-07T16-13-09Z` 经配置的 docker.io 镜像加速器
  持续返回 `免费节点当前繁忙` / `denied`；
- `quay.io/minio/minio` 返回 `401 UNAUTHORIZED`；
- 因此 `make up` 无法启动对象存储，V1 的 MinIO 集成证据与 V2 “真实 MinIO”用例被阻塞。

## Decision

使用 **PGSTY Silo** 作为对象存储服务镜像，它是 Pigsty 团队维护的 MinIO 社区分支，
S3 接口与 MinIO 管理 API 兼容：

- 镜像：`pgsty/silo:RELEASE.2026-10-26T00-00-00Z`
  （digest `sha256:7bb2aff6b89efe1cb1303c10fc21d579447434ed06649ac9ec05a5e2dbf30ad7`；
  上游 `https://github.com/pgsty/silo`，licence `AGPL-3.0-or-later`，
  image labels 记录 version/revision/source）
- 完全复用原 MinIO 配置：service 名保持 `minio`、数据目录 `/data`、端口 `9000/9001`、
  `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` 环境变量不变；`S3_*` 应用配置零改动。
- 健康检查从 `mc ready local` 改为镜像内置的 `silo healthcheck live`（`ready` 亦可）。
- 因为 `minio/mc` 镜像同样不可再获取，移除 `minio-init` 服务；**桶由应用自身创建**：
  `S3Storage.ensure_bucket()` 在 API `/health/ready`、worker 启动和 seed 中调用
  （head bucket → create bucket），不在 compose 中依赖 mc。

## Evidence (executed on 2026-10-29, this host)

1. `docker pull pgsty/silo:RELEASE.2026-10-26T00-00-00Z` 成功（230 MB，digest 同上）。
2. `docker compose up -d minio` → `dataops-minio` healthy；
   `GET /minio/health/live` = 200，`/minio/health/ready` = 200，console `:9001` = 200。
3. `silo healthcheck live` / `ready` 在容器内 exit 0。
4. 本仓库 `S3Storage` 针对 Silo 的实测（`tests/integration/test_s3_storage_silo.py` 与手工脚本）：
   `ensure_bucket`、`put_bytes`、`get_bytes`、`stat`（含 last_modified）、`sha256_of`、
   `copy`、`list_keys`、`delete`、**预签名 PUT/GET（HTTP 200、内容一致）** 全部通过。
5. 以 `STORAGE_BACKEND=s3` 跑完整 `dataops seed`（真实 HTTP 无关的服务层链路）
   验证 canonical/quality/annotation 对象写入 Silo 后可被读取。

## Consequences

- `make up` 的对象存储依赖重新可用；MinIO API/S3 语义证据可以给出。
- 依赖许可：Silo 为 AGPL-3.0-or-later，与 MinIO 相同，仍作为独立网络服务使用，不链接进
  本仓库代码；`docs/licenses.md` 已更新。
- 任何假定官方 MinIO 二进制/`mc` 的旧脚本，必须改用应用侧 `ensure_bucket` 或 S3 客户端；
  仓库内已无 `mc` 依赖。
- 版本必须钉住 release tag + digest；`latest` 禁止进入交付配置（与 V1 规格 §2 一致）。
- 若未来 Silo 项目停更/不兼容，回退路径是把 `S3Storage` 指向任意 S3 兼容服务；应用代码
  不依赖 MinIO 专有 API（仅 S3 + presigned URL）。

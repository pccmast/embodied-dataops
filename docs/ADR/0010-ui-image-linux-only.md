# ADR-0010: UI 容器镜像不在本机 Windows 上构建（Linux/CI 构建 + 本机直跑 Streamlit）

Status: accepted (2026-10-29)
Related: ADR-0009 (build without apt), ADR-0002 (dev host deviations)

## Context

`make up` 的核心服务（PostgreSQL/Redis/对象存储/migrate/api/dispatcher/workers）在本机
Windows + Docker Desktop 上已经可以构建（`dataops-api:dev` 1.85 GB、`dataops-worker-cpu:dev`
2.67 GB 均成功导出，容器内 Python 3.11.13 可 import 应用）。

但 `ui` 镜像（Streamlit）在 Docker Desktop for Windows 上稳定卡在 **`exporting layers`**
阶段：

- 所有构建步骤（1–7，含 `uv sync --extra ui` 47.6s）都成功；
- 之后 `#14 exporting to image / exporting layers` 长时间无任何进展（10+ 分钟，日志不再增长）；
- 已排除：`--no-cache` 重建、换镜像 tag、`--sbom=false --provenance=false`、Legacy builder、
  禁用 SBOM/Scout 索引并重启 Docker Desktop、小镜像导出（3.2s 成功）、api/worker 镜像导出
  （均成功）。→ 这是 Docker Desktop Windows 的 image-store 写入路径问题（大量小文件的
  Streamlit 依赖层 + Windows AV/gvisor 文件 I/O），不是本仓库 Dockerfile 逻辑错误。

继续在 Windows 上调试该问题收益极低，且 UI 属于 compose 的可选 profile。

## Decision

1. **UI 容器镜像改在 Linux 环境/CI 构建**（交付基线本来就是 Linux x86-64）。Windows 开发机
   不把 UI 镜像构建作为任何验收/演示的前置条件。
2. **本机 UI 以 host 进程运行**：`make ui` → `dataops ui` → `streamlit run src/embodied_ui/app.py`，
   使用同一个 `src/embodied_ui` 代码、本地 `.venv`、dev API key 与 `UI_API_BASE_URL`。
   UI 代码与容器镜像行为一致（只通过 HTTP API 访问 DataOps）。
3. **compose 保持 UI 服务定义**（`profiles: ["ui"]`，镜像/端口/命令不变），供 Linux/CI
   构建与部署；`make up` 只启动核心服务，不触发 UI 镜像构建。新增 `make build`（核心镜像）
   与 `make ui-image`（Linux/CI 显式构建 UI 镜像）。
4. **验收口径**：
   - AT-01（`make up && make seed`）：核心服务 + `/health/ready`，与 UI 镜像无关，可正常执行；
   - AT-23（Streamlit 四页）：本机以 host Streamlit + API 驱动 + `streamlit.testing`
     headless 页面测试取证；容器化 UI 截图的证据由 Linux/CI 构建后补（如环境没有浏览器，
     不伪造截图，标记 PARTIAL 并记录原因）；
   - V2 的 UI 相关 DO2（标注修订编辑、覆盖/补采展示等）：先以 host Streamlit + AppTest +
     API 测试验证，镜像构建在 Linux/CI 完成后作为部署证据。
5. **CI**：新增 Linux CI job 实际构建 `api / worker-cpu / ui / worker-ros2` 镜像并做容器内
   `python -V`/import 冒烟；本机 Windows 不再重复该步骤。

## Consequences

- Windows 本机不再被 UI 镜像导出问题阻塞；核心 `make up`、`make seed`、`make test`、
  `make acceptance` 都可继续。
- `docker compose --profile ui up` 在 Windows 上仍会触发 UI 构建（不推荐）；文档明确指引
  Windows 使用 `make ui`，Linux/CI 使用 `make ui-image`。
- Streamlit 在 host 运行时需要 `.venv` 含 `ui` extra（`uv sync --extra ui`）与一个有效
  API key（`var/bootstrap/api_keys.json` 或 `API_KEY` 环境变量）。
- 少了“UI 容器内运行”这一条部署证据；必须在 Linux CI/演示环境补齐，不能把 host
  Streamlit 宣称为容器部署验证。
- 与 ADR-0009 一致：交付基线仍是 Linux 容器；Windows 只是开发/测试宿主。

# ADR-0011: 容器镜像构建从本机 Windows 移除，改为 Linux/云 GPU 演示环境完成

Status: accepted (2026-10-29)
Related: ADR-0002（开发机偏差）、ADR-0008（Silo）、ADR-0009（无 apt 镜像）、ADR-0010（UI 镜像 Linux-only）

## Context

本机 Windows + Docker Desktop 已经过完整排查：

- **镜像拉取**：postgres/redis/python/ros/prometheus/grafana/pgsty-silo 都能拉取（MinIO 官方镜像
  已被上游下架，见 ADR-0008）；
- **镜像构建**：`dataops-api:dev`(1.85 GB) 与 `dataops-worker-cpu:dev`(2.67 GB) 已成功构建并在
  容器内验证 `Python 3.11.13` + 应用 import；但 UI 镜像稳定卡在 `exporting layers`（已排除
  apt、README、缓存、tag、SBOM、legacy builder 等因素，见 ADR-0010）；
- **资源**：Docker Desktop VM 只有 8 GB，构建 + 测试并发会让 db/redis 容器 OOM kill（exit 137）；
- **目标环境**：项目最终演示/测试将在 **Linux + 云 GPU** 上执行，Windows 只是开发宿主。

继续在 Windows 上调试镜像构建/导出问题，边际收益很低，而且会挤占真正未完成的代码工作
（V2 仍有 55 项 GAP，其中 42 项 MISSING）。

## Decision

1. **停止在 Windows 本机构建项目容器镜像**（不再运行 `docker build` / `docker compose build`
   作为本机交付步骤）。Dockerfile、docker-compose.yml、版本锁定、CI 定义作为**交付源码**
   原样保留，由 Linux/云 GPU 环境负责构建与验证。
2. 本机只保留**已拉取的基础服务容器**（Postgres/Redis/PGSTY Silo，可选 Prometheus/Grafana）
   作为测试依赖；不构建任何项目镜像。
3. 本机开发/验收统一使用 **host-mode**：
   - DB/对象存储：已拉取的容器；
   - API/dispatcher/worker/UI：host Python 进程（同一套代码、同一套 DB/Redis/S3 契约）；
   - 一条命令：`scripts/dev_stack.ps1`（Windows）启动依赖 + migrate + api + dispatcher + workers；
     `scripts/stop_dev_stack.ps1` 停止；UI 用 `make ui`（host Streamlit，已验证 HTTP 200）。
4. **Linux/云 GPU 演示清单**独立成文：`docs/linux-demo-plan.md`，列出在 Linux 上必须重新执行
   的命令（`make up`、`make seed`、`make acceptance`、`benchmark`、`@pytest.mark.ros2`、GPU
   YOLO、UI 镜像、截图）与要回填的证据。
5. **验收状态语义**：本机无法完成的容器构建/GPU/真实云环境项统一标记
   **`DEFERRED_LINUX`**（原因：交付目标环境是 Linux+GPU，不在 Windows 伪造通过），并保留
   已完成的 host-mode 证据；不得写成 DONE/PASS。
6. 本机继续推进所有不依赖镜像构建的工作：V1 回归/修复、host-mode acceptance、benchmark、
   UI AppTest、V2 M0–M6 的代码/迁移/契约/测试。

## Consequences

- 本机不再因 Docker Desktop 的导出/内存问题阻塞；时间集中到代码逻辑与测试。
- 交付物完整性不变（Dockerfile/compose/CI 仍是源码的一部分），但**本机没有容器运行证据**；
  Linux 演示环境必须按清单执行并保存证据。
- `make up` 的本地语义变为“Linux 执行”；Windows 使用 `scripts/dev_stack.ps1`，README/runbook
  必须写清楚，不能暗示 Windows 已验证 `make up`。
- GPU 相关（`VISION_DEVICE=cuda`、GPU 利用率、真实 YOLO 吞吐）在 Linux 上验证；CPU 真实
  YOLO 已在本机 AT-21 证据中。
- CI（GitHub Actions/Linux）保留为容器构建的自动化证据来源；但 CI 网络可达性不受本机控制，
  因此 Linux 演示仍需本地复跑一次。

## Alternatives rejected

- 继续在 Windows 调试 Docker Desktop 导出：已尝试缓存/无缓存/换 tag/SBOM/legacy builder/
  重启，均无效，且属于宿主实现细节，不是项目代码问题。
- 把容器项直接标 BLOCKED：不准确——它们在 Linux 目标环境是可完成的，应标
  `DEFERRED_LINUX` 并在演示环境补齐。

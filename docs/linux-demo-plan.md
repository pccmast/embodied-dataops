# Linux / 云 GPU 演示与验收清单

> 背景：本机（Windows 11 + Docker Desktop 不可用）只做代码开发与 host-mode 验证；容器镜像
> 构建、GPU 推理、真实 ROS 2 容器、UI 容器截图按 ADR-0011 统一在 Linux/云 GPU 环境执行。
> 本文件是该环境的一次性可复现清单，执行后把结果回填到 `docs/test-report.md` /
> `docs/benchmark.md` / `artifacts/`。
>
> **顺序很重要**：`make bootstrap` 会跑 Alembic 迁移，因此必须先让依赖服务起来。
> 下面每一步都写成可直接复制的形式，且只引用 `Makefile` 里真实存在的 target。

## 0. 环境准备

```bash
# 要求：Linux x86-64、Docker Engine 24+ / Compose v2、NVIDIA 驱动 + Container Toolkit（仅 GPU 档需要）
docker version --format '{{.Server.Version}}'   # 守护进程可达
docker compose version                          # 必须有 Compose v2 插件；静态安装的 docker 不含它
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L   # 仅 GPU 档：必须能列出设备

git clone <repo> && cd <repo>
cp .env.example .env
# 网络受限时按需设置镜像（默认官方源）。写进 .env 更稳妥，compose 会自动读取：
#   PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# 注意：uv.lock 里每个包记录了 registry 来源，`uv sync --frozen` 会优先走锁内地址，
# 上面的索引只对未锁定/新建的解析生效。

# 1) 先把依赖服务拉起来：迁移需要一个真实可连的 PostgreSQL
docker compose up -d db redis minio
docker compose ps                       # 等 db/redis/minio 变成 healthy

# 2) 再同步依赖 + 迁移 + 生成初始 API Key（只显示一次）
make bootstrap

# 3) 校验迁移头（单 head，不允许并行 head）
uv run alembic heads
```

GPU 是**可选**的，且只有 `worker-vision` 需要：

- 变量名是 `YOLO_DEVICE`（`config.py` 里检测器读取的就是它），不是 `VISION_DEVICE`；
- 除设备变量外，镜像本身必须带 CUDA 版 torch（PyPI 默认轮子是 CPU 版），
  通过 `build-gpu` 传入的 `TORCH_INDEX_URL` 构建；
- 宿主有 GPU 不等于容器可见，也不等于模型跑在 GPU 上。

```bash
# GPU 档（仅在有 NVIDIA 设备的机器上）
# docker-compose.gpu.yml 已把 TORCH_INDEX_URL 作为 build arg 传入并默认 cu124，
# 这里不用 export；确需换 CUDA 版本时才覆盖：
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 make build-gpu
make build-gpu
make up-gpu

# 构建后必须确认镜像里真是 CUDA 版 torch（"宿主有 GPU" 不算证据）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm --entrypoint python worker-vision -c \
  "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"

# 容器内先确认设备，再用真实权重跑一次 YOLO 并把 device 写进报告
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec worker-vision python -c \
  "import torch; print('cuda_available=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
make yolo-real                          # 产出 reports/v21/yolo-real.json（含 device）
```

> `nvidia-smi` 成功不算证据。验收要求 `reports/v21/yolo-real.json` 里
> `device` 明确是 `cuda`，并记录权重 SHA、输入 SHA 与耗时。

无 GPU 时保持 `YOLO_DEVICE=cpu`，`make build` / `make up` 即可，GPU 项在验收表里标
`GPU_BENCH_BLOCKED`，不得写成 PASS。

## 1. 容器构建与启动（Windows 本机不执行）

```bash
docker compose --profile "*" pull        # 先拉第三方镜像：postgres/redis/silo/prometheus/grafana
make build              # api / dispatcher / worker-cpu / worker-vision
make ui-image           # Streamlit 镜像（Linux-only，见 ADR-0010）
docker compose --profile ros build worker-ros2   # ROS 2 Jazzy 原生 MCAP 路径

make up                 # 核心服务（db/redis/silo/migrate/api/dispatcher/workers）
docker compose --profile ui up -d ui             # UI 容器（可选）
docker compose --profile monitoring up -d prometheus grafana
curl -fsS http://localhost:8000/health/ready | python3 -m json.tool   # 未装 jq 时用它
```

构建期网络前提：`--extra lerobot` 会引入 `evdev`（只有 sdist），容器内需要能访问
Debian 源以安装 `gcc / libc6-dev / linux-libc-dev`（Dockerfile 已处理）。若构建机器
无法访问 `deb.debian.org`，`uv sync` 会以 `Failed to build evdev==2.0.0` 报错。

需要保存的证据：`docker images --digests`、`docker compose ps`、`/health/ready` 响应、
API 容器 `python -V`、worker 容器 `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`、
UI 页面截图（四页）。

容器相关的已知前提（构建后应逐条确认，而不是假设）：

| 检查 | 命令 |
|---|---|
| 镜像实际名字 | 本地 compose：`embodied-dataops-api:latest` 等（`docker compose --profile "*" config --images` 可查）；CI：`docker build -f … -t dataops-api:ci .`（`.github/workflows/ci.yml`），两套命名不同，别混用 |
| Profile/Prompt 资源已进镜像 | `docker run --rm --entrypoint python embodied-dataops-api:latest -c "import pathlib;print(pathlib.Path('/app/contracts').is_dir())"` |
| LeRobot 未被后续 sync 清理 | `docker run --rm --entrypoint python embodied-dataops-worker-cpu:latest -c "import lerobot; print(lerobot.__version__)"` |
| FFmpeg 可用 | `docker run --rm --entrypoint python embodied-dataops-worker-cpu:latest -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"` |
| UI 指向 compose 内 API | `docker compose exec ui printenv UI_API_BASE_URL`（应为 `http://api:8000`） |

## 2. V1 验收（AT-01～AT-25）

```bash
make seed                        # 十个 fixture 生成 + 导入 + 处理
make test                        # unit + integration + contract
make e2e
make failure-test
make acceptance                  # junit + AT 矩阵（artifacts/test-results/）
```

- 只跑某个 fixture：`make seed SEED_ONLY="F04_ego_video"`（`--only` 是 CLI 参数，不能直接
  跟在 `make seed` 后面，make 会把它当成自己的选项）。
- 容器内 worker 处理：`make seed` 由 host CLI 入队；容器 worker 需先 `make up`，compose 默认
  已传 `INSTALL_LEROBOT=1`/`INSTALL_VISION=1`。
- `make failure-test` 中的真实 SIGKILL/依赖断电项在 Linux 上执行（`scripts/inject_failure.sh`）。
- 真实 YOLO：`DETECTOR=yolo11n make seed SEED_ONLY="F04_ego_video"`，GPU 时设置
  `YOLO_DEVICE=cuda`；保存权重 SHA、输入 SHA、推理参数与 annotation JSON（AT-21）。
- 真实公开数据：`HF_ENDPOINT=https://hf-mirror.com make fetch-public-example` 后按
  `docs/public-dataset-provenance.md` 导入；V1 仅支持 6D profile，非 6D 数据只做只读检查。

## 3. 性能与故障实验（AT-22 等）

```bash
make bench-v2                            # 100 synthetic episodes, 并发 1/4 × 3 次
make bench-v2-smoke                      # 5 episodes 快速自检
uv run pytest tests/performance -q -m performance   # API P95
```

`scripts/bench_v2.py` 的参数是 `--episodes / --repetitions / --concurrencies`（复数、
逗号分隔），没有 `--cpu-concurrency`；直接写错会被 argparse 拒绝。

要求：记录宿主 CPU/内存/GPU、worker 并发、容器镜像 SHA、模型版本、实际数据来源；重复 3 次
报告均值/方差；不要声称未实测的吞吐。

故障注入（Linux）：`scripts/inject_failure.sh all`，以及真实 `kill -9` worker、停 Redis、
短时阻断 MinIO/Silo，验证 120s 内终态与“唯一有效 Asset”。

## 4. ROS 2 原生 MCAP

```bash
docker compose --profile ros up -d worker-ros2
make seed SEED_ONLY="F05_mcap_good F06_mcap_gap"
uv run pytest tests/integration/test_mcap_import.py -q        # 含 @pytest.mark.ros2
```

需要保存：rosbag2_py 读取的 topic/message count、native worker 任务日志、F05 PASS / F06
`SYNC_MISSING_STATE` FAIL 的质量报告。

## 5. V2 增量

按 `docs/design.md` 的约束与 `docs/ADR/0007` 的 Gate 执行：

```bash
make fixture-v2 && make test-v1-compat && make test-v2 && make test-contract
make demo && make bench-v2
```

必须保存：DO2-01～35 的 junit/matrix、OpenAPI v2 diff、迁移日志、真实数据/GPU 证据、
`docs/test-report-v2.md`、Git SHA。

## 6. V2.1（AI 预标注支路）

```bash
make test-v21                            # 离线单元测试（不需要 PostgreSQL）
make probe-sources                       # 数据源探针（exit 3 = BLOCKED）
make acceptance-v21                      # reports/acceptance-v21.json（exit 3 = BLOCKED）
```

`make acceptance-v21` 里的状态声明不等于真实探测：跑完请对照
`reports/v21/source-probe.json`、`reports/v21/yolo-real.json` 与 JUnit 证据人工复核。

真实 VLM 需要凭证与外发授权，两条都必须满足：

```bash
export VLM_PROVIDER=dashscope
export VLM_API_KEY=<key>            # 只放在环境/secret manager，不进 DB、日志、截图
# 数据外发授权是数据源属性，不是环境变量：
#   UPDATE source_imports SET egress_allowed = true WHERE id = '<import_id>';
```

未取得凭证时 VLM 项保持 `BLOCKED`，replay 结果不能填 live 列。

## 7. 回填与状态

- 把 Linux 上实际命令、输出、截图、镜像 digest 写入 `artifacts/linux/<date>/`；
- 更新 `docs/implementation-status.md`：以实测为准改状态，不用“存在即通过”；
- 若某项仍失败，保留失败输出并如实标 FAILED/BLOCKED，不得改写成 PASS；
- 本清单中的命令若因环境差异调整，需在 `docs/ADR/` 记录原因。

## 8. Windows 本机（仅开发）对照

```powershell
scripts/dev_stack.ps1        # 依赖容器 + migrate + api + dispatcher + workers(host)
make ui                      # host Streamlit
scripts/stop_dev_stack.ps1
```

Windows 不执行 `make build` / `make ui-image` / GPU 项；这些统一在上面的 Linux 清单完成。

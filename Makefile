# Embodied DataOps - single entrypoint for local development and acceptance.
# Every recipe is a single command so the file also works with GNU make on
# Windows (ezwinports.make) and in the Linux dev container.

SHELL := /bin/sh
COMPOSE ?= docker compose
DOCKER ?= docker
PY ?= uv run python
PYTEST ?= uv run pytest
#: worker-vision 用 GPU overlay 构建出来的镜像名；yolo-real-gpu 直接复用它。
VISION_IMAGE ?= embodied-dataops-worker-vision:latest
YOLO_DEVICE_GPU ?= cuda:0

.DEFAULT_GOAL := help

.PHONY: probe-sources yolo-real yolo-real-gpu test-v21 acceptance-v21 help bootstrap build build-gpu up-gpu dev-stack dev-stop ui ui-image up down fixture-v2 profiles test-v2 test-v1-compat test-concurrency bench-v2 bench-v2-smoke failure-test-v2 migrate seed seed-synthetic test test-ui test-unit test-integration test-e2e test-failure test-contract test-performance e2e demo benchmark failure-test acceptance openapi lint fmt fetch-public-example clean keys status logs

help:
	@$(PY) -m embodied_dataops.cli.main help

bootstrap: ## create .env, venv, install dependencies, build images
	@$(PY) -m embodied_dataops.cli.main bootstrap

up: ## start postgres, redis, object store, migrate, api and workers (UI: make ui)
	$(COMPOSE) up -d --build

build: ## build the core container images (Linux; no UI image on Windows)
	$(COMPOSE) build api dispatcher worker-cpu worker-vision

build-gpu: ## build worker-vision with a CUDA torch wheel (GPU overlay; Linux + NVIDIA only)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml build worker-vision

up-gpu: ## start the stack with the GPU overlay for worker-vision (needs build-gpu first)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d worker-vision

# 不用 `compose run`：worker-vision 声明了 db/redis/migrate 依赖，run 会把它们一起拉起。
# 这里直接用 docker run，显式挂载只读权重/视频与可写 reports，并强制 CUDA。
yolo-real-gpu: ## real YOLO11n inference on CUDA inside the GPU vision image (no CPU fallback)
	$(DOCKER) run --rm --gpus all \
	  -v "$(CURDIR)/scripts:/app/scripts:ro" \
	  -v "$(CURDIR)/var/models:/app/var/models:ro" \
	  -v "$(CURDIR)/public-data:/app/public-data:ro" \
	  -v "$(CURDIR)/reports:/app/reports" \
	  -w /app --entrypoint python \
	  $(VISION_IMAGE) scripts/run_yolo_real.py --require-cuda --device $(YOLO_DEVICE_GPU)

dev-stack: ## Windows host-mode stack: dependency containers + host api/dispatcher/workers
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/dev_stack.ps1

dev-stop: ## stop the Windows host-mode stack
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_dev_stack.ps1

ui: ## run the Streamlit UI on the host (no container image required)
	@$(PY) -m embodied_dataops.cli.main ui

ui-image: ## Linux/CI only: build the Streamlit container image
	$(COMPOSE) build ui

test-ui: ## Streamlit API-client + headless page tests
	$(PYTEST) tests/unit/test_ui_client.py -q

down: ## stop the local stack (objects and database volumes are kept)
	$(COMPOSE) down

migrate: ## apply all Alembic migrations (idempotent)
	uv run alembic upgrade head

seed: ## generate the ten synthetic fixtures, migrate, import and process them (SEED_ONLY="F04_ego_video" to filter)
	@$(PY) -m embodied_dataops.cli.main seed $(if $(SEED_ONLY),--only $(SEED_ONLY),)

fixture-v2: ## generate the deterministic V2 fixture manifest (profiles/6D/7D/dual-camera/pointcloud plans)
	@$(PY) -m embodied_dataops.cli.main fixtures-v2

profiles: ## register the built-in V2 robot profiles in PostgreSQL (idempotent)
	@$(PY) -m embodied_dataops.cli.main profiles

seed-synthetic: ## generate fixtures only (offline, no services required)
	@$(PY) -m embodied_dataops.cli.main fixtures

test: ## unit + integration + contract tests
	$(PYTEST) -m "not performance and not ros2 and not vision"

test-unit: ## fast in-process tests
	$(PYTEST) -m unit

test-concurrency: ## release gate: two workers on real row locks (needs PostgreSQL, ~30s)
	$(PYTEST) -m concurrency -q

test-integration: ## tests that need PostgreSQL, Redis and MinIO
	$(PYTEST) -m integration

test-e2e: ## end-to-end AT tests against the stack
	$(PYTEST) -m e2e

e2e: test-e2e

test-failure: ## failure-injection tests (leases, outbox, cancellation)
	$(PYTEST) -m failure

failure-test: test-failure

test-v2: ## V2 unit/integration/contract tests (tests/v2)
	$(PYTEST) tests/v2 -q

test-v1-compat: ## V1 regression tests that must keep passing during V2 work
	$(PYTEST) tests/v2/compat tests/unit tests/integration -q -m "not ros2 and not vision and not performance"

test-contract: ## external contracts (official LeRobot Reader/Writer, OpenAPI)
	$(PYTEST) -m contract

test-performance: ## benchmarks (writes artifacts/benchmarks/report.json)
	$(PYTEST) -m performance

benchmark: test-performance

bench-v2: ## V2 measured benchmark: 100 synthetic episodes, CPU concurrency 1/4 x3
	$(PY) scripts/bench_v2.py --episodes 100 --repetitions 3 --concurrencies 1,4

bench-v2-smoke: ## V2 benchmark smoke (5 episodes, both concurrencies, 1 repetition)
	$(PY) scripts/bench_v2.py --quick --report artifacts/benchmarks/bench-v2-smoke.json

failure-test-v2: ## V2 real fault injection: kill-worker + redis-down (storage-down needs s3)
	$(PY) scripts/fault_inject_v2.py --scenario all --episodes 3 --timeout 120

demo: ## run the scripted end-to-end demo
	@$(PY) -m embodied_dataops.cli.main demo

acceptance: ## run AT-01..AT-25 and write artifacts/test-results/junit.xml
	@$(PY) -m embodied_dataops.cli.main acceptance

# --- V2.1 ---------------------------------------------------------------

probe-sources: ## check the declared V2.1 data sources offline (exit 3 = BLOCKED)
	@$(PY) scripts/probe_sources.py --out reports/v21/source-probe.json

yolo-real: ## real YOLO11n inference evidence over a real dataset video (V21-10)
	@$(PY) scripts/run_yolo_real.py

test-v21: ## V2.1 unit tests (no PostgreSQL required)
	@$(PYTEST) tests/v21 -q

acceptance-v21: ## build reports/acceptance-v21.json and run the offline suite (exit 3 = BLOCKED/not run)
	@$(PY) scripts/acceptance_v21.py --run-tests

openapi: ## regenerate docs/api-openapi.json from the FastAPI app
	@$(PY) -m embodied_dataops.cli.main openapi

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

fetch-public-example: ## explicitly download the public LeRobot v3 demo dataset
	@$(PY) -m embodied_dataops.cli.main fetch-public-example

keys: ## print API keys stored in the database (hashes only, never plaintext)
	@$(PY) -m embodied_dataops.cli.main keys

status: ## show per-AT implementation status
	@$(PY) -m embodied_dataops.cli.main status

logs:
	$(COMPOSE) logs --tail=80

clean: ## DESTRUCTIVE: requires CONFIRM=YES
	@$(PY) -m embodied_dataops.cli.main clean --confirm "$(CONFIRM)"

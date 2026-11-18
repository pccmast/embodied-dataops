# ADR-0007: V2 canonical 2.0 / Profile Registry / 多流时间索引 / 兼容与迁移策略

Status: accepted as the V2 implementation baseline (2026-10-29) — binding for D2-M0…M2
Supersedes nothing; V1 decisions in ADR-0001…0006 remain valid unless explicitly overridden here.
Implementation owner: DataOps Agent (this repository).

## Context

V1 固定了 6D/10Hz/单 front 相机并把 canonical-episode/1.0 作为唯一内部契约；V2 需要
Profile Registry（6D/7D）、原生多流时间索引、双相机、语义 QC、标注修订、条件写合同。
V2 必须**原地升级**且不重写任何 V1 已发布资产/版本，同时保留 V1 读兼容。

## Decision

### 1. 双契约并行，V1 只读冻结

- 新导入只写 `canonical-episode/2.0`；V1 canonical 对象不被重写。
- `episodes.canonical_schema_version` 区分 1.0/2.0；V1 记录升级后为 `1.0`，**禁止推断回填
  profile**。新 canonical 输出生成新的 `canonical_version` 快照并原子切换当前指针，历史
  asset/digest 永久保留。
- V1 `dataset-manifest/1.0`、V1 PUBLISHED DatasetVersion、V1 annotation set/quality report
  永久冻结；V2 走 `dataset-manifest/2.0` 与新版本号。

### 2. canonical v2 目录与三种视图

按 SPEC §2.3 固定：

```text
canonical/{episode_id}/v2/{canonical_version}/
  metadata.json
  native/cameras/{front,wrist}/timestamps.parquet
  native/state.parquet
  native/action.parquet
  native/pointcloud/summary.parquet
  aligned/10hz/samples.parquet
  preview/{front,wrist}_10hz_640x480.mp4
  mapping_report.json
```

- `NATIVE` 视图保存源真实采样时间，**不重采样、不丢帧**；`ALIGNED_TRAINING` 只保存训练
  网格与 validity；`PREVIEW` 是可再生展示品，不是训练真值。
- `canonical_views` 以 `(episode_id,canonical_version,view_kind,config_sha256)` 唯一；
  `config_sha256` 为视图生成参数（fps/相机/容差/profile）的 canonical JSON SHA-256。
- `metadata.json` 必选字段严格按 SPEC §2.3；`modalities[]` 必须记录
  `timestamp_provenance`、`source_t_ns_origin`、`native_asset_sha256`、`alignment_method`、
  `alignment_tolerance_ns`。`clock_verified=false` 时只可称“基于日志时间的近似同步”。

### 3. Profile Registry 与语义权威

- `robot_profiles` 以 `(profile_id, profile_revision)` 唯一，`config_json` 的 canonical JSON
  SHA-256 写入 `config_sha256`；只有 `authority ∈ {SYNTHETIC_FIXTURE, SOURCE_DOCUMENTED}`
  且带证据的 profile 可通过语义 QC。
- 选择 profile 必须由导入 mapping 明确给出 `profile_id + profile_revision`，或显式
  `diagnostic_only`；**禁止**依据 `shape=[6|7]`、文件名、模型文本自动选择。缺证据一律
  `UNVERIFIED_SEMANTICS`（ERROR，不可发布）。
- 合成 Profile 的限位/速度仅用于 fixture 质量验证，不构成任何真实机械臂安全限位声明。

### 4. 精度与不可伪造性

- native 索引保存原始值的 float32 投影，并记录 float64→float32 的最大绝对误差；误差
  `>1e-5` 时禁止进入训练导出（原始数据仍保留）。
- aligned 视图对缺失/失效样本写 `valid=false`、数组为 `null`，**禁止 0 填充**；只有全
  valid 的 Episode 可发布训练格式。LeRobot 已对齐行不得宣称原生频率差异。
- 时间统一：MCAP 以 `log_time` 为基准、保留 header stamp 偏差；不作硬同步声明。

### 5. 迁移策略（expand-only）

- 迁移序列 `v2_001…v2_006` 全部 expand；空库连续两次 `upgrade head` 必须成功；V1 库升级
  后旧记录可查、已发布 SHA 不变。
- 只允许在“无任何 v2 对象”时 downgrade；否则显式失败，绝不删除已发布数据。
- `episodes.business_revision` 等列一次迁移到位，后续迁移需检测列是否已存在（可重入）。

### 6. 标注修订语义

- `annotation_sets` 是逻辑集合；`annotation_revisions` 是不可变内容版本；`annotation_revision_edits`
  记录每次变更的 before/after SHA 与 actor。
- POST revisions 采用“受影响帧全量替换”语义（请求只列修改帧），以 `If-Match:
  annotation-rev:<current>` + Idempotency-Key 做 CAS；冲突 412、无变化 422、旧 revision
  永久不可改。
- 批准需 `approved_by != created_by` 的另一 reviewer；`annotation_sets.approved_revision_id`
  是唯一发布指针，批准后该 revision 不可变。v1 已发布版本继续引用旧 approved revision，
  V2 版本必须保存 `annotation_revision_id`。
- 为兼容 v1，legacy v1 annotation set 仍可用 `POST /annotation-sets/{id}/reviews`；对 v2
  标注集该接口返回 `409 USE_REVISION_REVIEW_API`，避免双状态。

### 7. 条件写事务与 fence（DO-A02 C2/C3）

单事务、固定顺序，任何一步失败全部回滚：

1. 鉴权 + JSON 校验；
2. 按 `(tenant_id,idempotency_key)` 查 `operation_receipts` 并比较请求摘要——**幂等检查先于
   If-Match**（已提交请求重放不因 revision 改变而失败）；
3. `SELECT ... FOR UPDATE` 锁业务资源；
4. 校验强 `If-Match`（`episode-rev|task-rev|evaluation-rev`，禁止 `W/`）→ 428/412；
5. 校验资格/状态（`RETRY_NOT_SAFE`、`INVALID_STATE` 等 409）；
6. INSERT `action_effect_fences` 唯一键 `(tenant,action_kind,target,source_business_revision)`
   ——冲突即 `409 OPERATION_ALREADY_APPLIED` 并返回原 operation_id；
7. 创建 Run/Analysis + Outbox + `operation_receipts`，业务资源 `business_revision+1`；
8. commit → 202（`effect_state=COMMITTED`）。

- receipt 保留 ≥90 天；fence 永久保留（不随幂等 TTL 删除）；不提供破坏性清理 API。
- 同 Key 重试早于首次提交完成时最多等待 3s，超时 503 `COMMIT_IN_PROGRESS`。
- `GET /operations/*` 返回真实下游状态，不含签名 URL/密钥/堆栈；仅 operator 可读。
- 这是刻意的 V1 安全不兼容：三条写接口缺 If-Match 一律 428，迁移指南必须列出受影响调用方。

### 8. 诊断 API 的有界只读原则

- timeline：窗口 ≤10s、最多 8 路流、`max_points≤200`，等间隔抽取（首尾保留），仅时间证据，
  超限 413 `TOO_MANY_MODALITIES`。
- failure-analysis：≤30 task、≤50 事件，仅结构化已脱敏事件，无堆栈/主机秘密，超限 413。
- quality/aggregate：窗口 ≤31 天、limit≤50，先对全匹配总体 SQL 聚合再排序截断；必须声明
  统计口径（截至 computed_at 最新完成 report）；rule_code 分组和不必等于 Episode 数。
- 这些接口不返回对象存储原始字节/坐标；Agent 只能看到规范化证据。

### 9. PointCloud2 有界解析

- 仅 MCAP_ROS2_PROFILE_V2 `/points`，支持 x/y/z FLOAT32、LE/BE、offset/padding/organized；
  先验证 `len(data)>=row_step*height`、字段 offset+4≤point_step、`row_step>=point_step*width`。
- 单消息 ≤1,000,000 点；超过 `POINTCLOUD_TOO_LARGE` 且不全量加载。统计仅用
  `ceil(N/10000)` 抽样（≤10,000 点）计算 bbox/nonfinite；`point_count` 来自原始维度。
- 复杂布局返回 `UNSUPPORTED_POINTCLOUD_LAYOUT`；不保留/不向 Agent 暴露原始坐标。

### 10. 覆盖与补采快照

- `scenario_tags.basis ∈ {SOURCE_DECLARED, HUMAN_VERIFIED, UNKNOWN}`；`UNKNOWN` 不计入
  有证据覆盖率。
- 覆盖查询只针对 `PUBLISHED` DatasetVersion，发布时生成只读
  `scenario_coverage_snapshots`，分母是 Episode 数而非帧数；后续补采进入新版本。
- 订单状态固定 `PROPOSED→APPROVED→IN_PROGRESS→FULFILLED|CANCELLED`；`is_simulated=true`
  的评测来源在 API/UI/订单全链路显式标记，不得与真实训练效果混淆。

## Consequences

- V1 读兼容、V1 已发布物冻结；V1 三条写接口 + 调用方需要显式升级（对应的
  superseded/migration 对照表属于内部规格包，未随本仓库发布）。
- 迁移与 fence/幂等为 correctness 关键路径，先测试后实现；任何未通过 Gate 的实现不得合入。
- 环境阻塞（MinIO/HF/ROS 容器）会限制可宣称的验收状态，只能按
  `BLOCKED`/`CORE_COMPLETE_REAL_DATA_BLOCKED` 如实交付。

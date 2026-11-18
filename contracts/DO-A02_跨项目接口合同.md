# DO-A02 — DataOps ↔ RoboData Agent V2 跨项目 API 与语义合同

> 合同版本 `do-agent-contract/2.0.0`；状态：**实施约束**；2026-10-29。
> 使用范围：两个项目的 V2 增量规格。两端都必须引用本文件原文且通过同一组集成合同测试。
> 路径中的 `{id}` 均为标准 UUID。所有业务接口延续 DataOps v1 `/api/v1` 前缀、Bearer API Key、`{data,request_id}` 包装、UTC 时间以及原有 RBAC。新增写接口必须按本文执行。

## C1. 合同版本与启动检查

- `GET /api/v1/capabilities`（viewer）返回 `{data:{contract_version:"do-agent-contract/2.0.0",canonical_versions:["1.0","2.0"],robot_profiles:["arm6_joint_position_v1","arm7_joint_position_v1"],qc_policy_versions:["qc-policy/1.0","qc-policy/2.0"],conditional_writes:true,operation_receipts:true,diagnostic_queries:true}}`。
- RoboData Agent V2 readiness 必须检查 `contract_version=do-agent-contract/2.0.0`，三个诊断接口存在、条件写启用；不满足返回 `503 DATAOPS_CONTRACT_MISMATCH`，不得自动降级到不带条件写的 v1 API。
- DO-A01 的 `GET /evaluations/{id}` 与 Episode `data_origin` 字段继续有效。新补充字段不会改变旧字段类型或业务意义。

## C2. 业务修订号与条件写

DataOps 表 `episodes`、`tasks`、`evaluation_reports` 新增 `business_revision BIGINT NOT NULL DEFAULT 1`；可影响业务前置条件的状态、质量报告指针、配置变更必须在同一个事务中使 revision +1。该 revision 和 `ETag` 是业务状态版本，**不是** `canonical_version`，也不是源文件或模型版本。

只读资源返回：

- `GET /episodes/{id}`：`data.business_revision`、`data.canonical_version`、`data.profile_id`、`data.profile_revision`；HTTP `ETag: "episode-rev:<n>"`。
- `GET /pipeline-runs/{id}`：`data.tasks[]` 每项增加 `business_revision`、`retryable`；`GET /tasks/{id}`（viewer）返回单 Task 及 `ETag: "task-rev:<n>"`，用于重试前读取。
- `GET /evaluations/{id}`：`data.business_revision`、`data.results_sha256`、`data.is_simulated`；HTTP `ETag: "evaluation-rev:<n>"`。

新增所有三种受控写操作必须有 `Idempotency-Key: <UUID>` 与 `If-Match: "<kind>-rev:<n>"`。`If-Match` 采用**强 ETag**，不允许 `W/` 弱标记。后端 PostgreSQL 事务内 `SELECT ... FOR UPDATE`，先检查 ETag、资格、业务 fence、同请求幂等，然后一次提交有效效果及 `operation_receipts`；不能用“先 GET 再 POST”代替事务内校验。

## C3. 三个修改型 API 的固定 V2 合同

| 行为 | method/path | Body | `If-Match` 目标 | 成功 |
|---|---|---|---|---|
| 重新处理 Episode | `POST /episodes/{episode_id}/pipeline-runs` | `{"pipeline_version":"2.0.0"}` | `"episode-rev:<n>"` | `202 {operation_id,new_run_id,state:"QUEUED",effect_state:"COMMITTED"}` |
| 重试已失败 Task | `POST /tasks/{task_id}/retry` | `{"expected_run_id":"<uuid>"}` | `"task-rev:<n>"` | `202 {operation_id,new_run_id,state:"QUEUED",effect_state:"COMMITTED"}` |
| 分析评测缺口 | `POST /evaluations/{evaluation_id}/analyze-gaps` | `{"success_rate_threshold":0.8,"min_trials":20}` | `"evaluation-rev:<n>"` | `202 {operation_id,analysis_id,state:"QUEUED",effect_state:"COMMITTED"}` |

- 任意这些 API 的**首个成功提交**，在 DB 中持久插入 `operation_receipts` 与 `action_effect_fences`。跨 `Idempotency-Key` 的业务 fence 唯一键：`(tenant_id,action_kind,target_id,source_business_revision)`；`action_kind` 分别 `REPROCESS_EPISODE|RETRY_FAILED_TASK|ANALYZE_EVALUATION_GAPS`。同业务 revision 只允许一次有效效果。首次成功提交三种行为时，同时把相应资源 `business_revision` 增加 1（Episode 创建新 Run、Task 创建重试 Run、Evaluation 创建 GapAnalysis），后续不依赖状态字符串间接推断 revision。不同 request key 的重入返回 `409 OPERATION_ALREADY_APPLIED` 并返回原 `operation_id`；不能创建第二个 Run/Analysis。
- 幂等请求摘要按 `tenant_id + action_kind + target_id + canonical JSON body + If-Match 原始强 ETag` 计算并持久化；同 Key 且摘要不一致一律 409。相同 `Idempotency-Key` + 相同主体，即使跨超时、进程重启或业务 revision 已经变更，也返回同一个原始成功响应；相同 Key + 不同主体返回 `409 IDEMPOTENCY_KEY_REUSED`。DB 查幂等必须**先于**新 If-Match 比较，避免已提交请求的重放因 revision 变化错误失败。
- `If-Match` 缺失返回 `428 PRECONDITION_REQUIRED`；值过期返回 `412 PRECONDITION_FAILED`，错误 details 含安全字段 `current_business_revision`；不符合业务状态返回 `409 INVALID_STATE`。禁止从 Agent 为兼容 v1 自动省略 `If-Match`。
- DataOps `tasks` 若为不可重试错误、非 `FAILED`、或 `expected_run_id` 与所属 Run 不符，返回 `409 RETRY_NOT_SAFE`；同一 `run_id/step` 新建的下游链必须遵循 v1 状态机与唯一约束。
- 请求权限仍要求 `operator`，只在 RoboData Agent 的 `action-executor` 容器持有 operator credential。Agent LLM worker 仍只有 viewer 凭证。

## C4. 可靠结果回查

`GET /api/v1/operations/{operation_id}`（operator）与 `GET /api/v1/operations/by-key/{idempotency_key}`（operator）返回相同结构；不得包含原始签名 URL、秘密和敏感堆栈：

```json
{
 "data": {
   "operation_id":"<uuid>","action_kind":"RETRY_FAILED_TASK",
   "target_id":"<uuid>","source_business_revision":3,
   "effect_state":"COMMITTED",
   "downstream_kind":"pipeline_run","downstream_id":"<uuid>",
   "downstream_state":"RUNNING","downstream_updated_at":"2026-10-29T00:00:00Z",
   "error_code":null
 },"request_id":"<uuid>"
}
```

- `effect_state` 固定 `COMMITTED`（在 HTTP 202 返回前数据库效果已成立）；若不存在返回 404。`downstream_state` 对 PipelineRun 使用 `QUEUED|RUNNING|SUCCEEDED|FAILED|CANCELLED`；对于 GapAnalysis 使用 `QUEUED|RUNNING|SUCCEEDED|FAILED`。下游状态由 DataOps 的已提交业务状态推导；不能仅依靠 HTTP POST 202 认为下游完成。
- DataOps `operation_receipts` 最少保留 90 天用于审计/故障回查；`action_effect_fences` 保留且不能因幂等请求缓存的 7 天 TTL 删除。项目内不提供破坏性清理 fence 的管理 API。
- RoboData executor 遇到 POST 响应丢失先 `GET /operations/by-key/{key}`，若 404 可用**同一 Key + 同一 If-Match + 同一 Body**有限重试；拿到 `operation_id` 后轮询 `GET /operations/{id}`，不得为同一次批准生成新 Key。

## C5. 三个只读诊断 API：固定结构与安全限制

**1. `GET /api/v1/episodes/{episode_id}/timeline?start_ns=<int>&end_ns=<int>&max_points=200`**（viewer）
- 范围 `0 <= start_ns < end_ns <= episode.duration_ns`，`end_ns-start_ns <= 10_000_000_000`；`1<=max_points<=200`，默认 100。
- 返回 `{data:{episode_id,profile_id,window:{start_ns,end_ns},streams:[{name,kind,clock_domain,sample_count,observed_gap_count,max_gap_ns,p95_alignment_error_ns,points:[{t_ns,source_t_ns,valid}],truncated}],evidence_digest}}`。`points` 是确定性等间隔抽取（首尾保留），不是原始帧或全部 action 数值；仅提供时间证据。限制最多 8 路流；超过返回 `413 TOO_MANY_MODALITIES` 而非静默漏掉。

**2. `GET /api/v1/pipeline-runs/{run_id}/failure-analysis`**（viewer）
- 返回 `{data:{run_id,failed_tasks:[{task_id,step,error_code,attempt,worker_pool,first_failed_at,last_failed_at}],dependencies:[{task_id,depends_on_task_id,state}],worker_incidents:[{incident_code,at,worker_pool}],summary_rule_codes:[...],truncated:false}}`；最多 30 task、50 事件。只可使用结构化已脱敏事件，严禁暴露堆栈/主机秘密；超过上限返回 `413 DIAGNOSTIC_RESULT_TOO_LARGE`。
- 这不是 LLM 生成的根因结论，未知问题为 `UNKNOWN`。

**3. `GET /api/v1/quality/aggregate?from=<RFC3339>&to=<RFC3339>&group_by=source_type|profile_id|rule_code|collection_id&limit=20`**（viewer）
- 时间为 UTC，`from < to` 且窗口 `<=31天`；`1<=limit<=50`。查询必须对整个匹配总体进行 SQL 聚合，之后才按 `count DESC, key ASC` 排序截取；返回 `{data:{matched_episode_count,group_by,groups:[{key,episode_count,fail_episode_count,warn_episode_count}],truncated,computed_at,filter_window}}`。
- `group_by=rule_code` 可出现一个 Episode 多条规则，因此各组和不必等于 matched_episode_count；其余互斥分组应相等。结果必须说明统计口径（quality report 选“截至 computed_at 最新完成版本”）。严禁将只读前 20 条 Episode 当成全库统计。

## C6. 补采关联与 PointCloud2（DataOps 内部新业务接口）

- `source_imports` 新增 `collection_order_id uuid nullable`；`POST /api/v1/imports/initiate` 可选 `collection_order_id`（UUID，必须指向本组织 `APPROVED|IN_PROGRESS` 的订单），并把该字段持久化为可追溯来源。未传此字段的上传不得事后仅靠用户传入 episode_id 冒充补采成果。
- `POST /api/v1/collection-orders/{id}/approve` reviewer、`POST /api/v1/collection-orders/{id}/link-episodes` operator、`GET /api/v1/collection-orders/{id}` viewer；link body `{episode_ids:[<uuid>]}` 最多 100，要求每个 Episode 的 `source_imports.collection_order_id == order_id`，有可信 `scenario_key` 标签、质量与标注审核合格，且该 Episode 尚未被计入其它补采订单；订单只能 `PROPOSED -> APPROVED -> IN_PROGRESS -> FULFILLED` 或 `PROPOSED|APPROVED|IN_PROGRESS -> CANCELLED`。`requested_count` 是期望数量，不表示已采样。
- `GET /api/v1/quality/coverage?dataset_version_id=<uuid>&scenario_key=<key>` viewer，返回 `{dataset_version_id,scenario_key,episode_count,tagging_basis,computed_at}`，`tagging_basis` 枚举 `HUMAN_VERIFIED|SOURCE_DECLARED|UNKNOWN`。无可信场景标签不允许伪造覆盖量。
- Agent V2 的 `propose_gap_analysis` 只触发 DataOps 创建 `PROPOSED` 补采单，不包含自动采集、OTA 或机器人执行权限。

## C7. 契约升级与兼容性

- DataOps V1 `GET` 返回旧字段应保留；新 `canonical-episode/2.0` 的全量读取使用新增 `/episodes/{id}/canonical`（viewer，返回 v2 元数据与仅安全的资产描述，不含原始字节）。对旧 Episode 的 v1 数据可继续读取，旧已发布 DatasetVersion 永久冻结。
- V1 `POST` 如果不含 `If-Match`：**V2 受控的三条接口统一拒绝 428**。这是刻意的安全不兼容，需更新 v1 UI/脚本/Agent 客户端；迁移指南必须清楚列出受影响调用方，不能偷偷保留无条件写入口。
- DataOps v2 必须在本地先通过 DO-A02 的全部接口合同测试，Agent v2 才可以切换到新接口。合同测试使用独立 DataOps viewer/operator Key；不可为测试关闭 RBAC。

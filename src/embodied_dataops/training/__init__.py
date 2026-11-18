"""训练准备层（EDO-TRAIN-PREP-1.1）。

首期是一个**独立 CLI + 文件产物**的闭环，不碰任务 DAG、不碰数据库 schema、
不改已发布的 DatasetVersion。它把一个已核验的导出包变成一份可复算的
TrainingBundle：划分、有效窗口、train-only 统计、冻结 processor。

模块边界（与规格 §10 对应）：

===========================  ===============================================
``contract``                 配方合同、来源绑定、规范语义摘要
``split``                    来源分组与 train/val/test 划分
``windows``                  有效窗口索引与拒绝统计
``stats``                    train-only 流式统计（M2）
``processors``               官方 ACT pre/post processor 与有效 stats（M2）
``dataset`` / ``validate``   受限 Reader 与真实 forward/backward（M2）
``artifact``                 原子发布不可变 bundle（M3）
===========================  ===============================================
"""

from __future__ import annotations

from embodied_dataops.errors import ApiError, ErrorCode

#: 规格 §12 的退出码合同。CLI 层用它把 ``ApiError`` 映射成进程退出码；
#: 未列出的错误一律按 2（配置/合同错误）处理，宁可保守也不要返回 0。
EXIT_OK = 0
EXIT_CONTRACT = 2
EXIT_BLOCKED = 3
EXIT_DATA = 4
EXIT_NUMERICS = 5

_EXIT_BY_ERROR: dict[ErrorCode, int] = {
    # 配置 / 来源合同错误
    ErrorCode.INVALID_INPUT: EXIT_CONTRACT,
    ErrorCode.SOURCE_UNVERIFIED: EXIT_CONTRACT,
    ErrorCode.FEATURE_INCOMPATIBLE: EXIT_CONTRACT,
    ErrorCode.UNSUPPORTED_STATS_SOURCE: EXIT_CONTRACT,
    # 依赖 / 设备 / 权重不可用
    ErrorCode.CUDA_UNAVAILABLE: EXIT_BLOCKED,
    ErrorCode.MODEL_UNAVAILABLE: EXIT_BLOCKED,
    ErrorCode.MODEL_CHECKSUM_MISMATCH: EXIT_BLOCKED,
    ErrorCode.OBJECT_STORE_UNAVAILABLE: EXIT_BLOCKED,
    # 缺解码依赖不是数据缺陷：它是环境缺口（TP-21 明确要求 BLOCKED 而不是 PASS/FAIL）。
    ErrorCode.DEPENDENCY_UNAVAILABLE: EXIT_BLOCKED,
    # 数据完整性 / 质量
    ErrorCode.SOURCE_HASH_MISMATCH: EXIT_DATA,
    ErrorCode.GROUP_LEAKAGE: EXIT_DATA,
    ErrorCode.INSUFFICIENT_SOURCE_GROUPS: EXIT_DATA,
    ErrorCode.EMPTY_REQUIRED_SPLIT: EXIT_DATA,
    ErrorCode.EMPTY_REQUIRED_WINDOWS: EXIT_DATA,
    ErrorCode.EMPTY_TRAIN_WINDOWS: EXIT_DATA,
    ErrorCode.NONFINITE_INPUT: EXIT_DATA,
    ErrorCode.FILE_DECODE_ERROR: EXIT_DATA,
    # 模型 / 处理器 / 计算
    ErrorCode.PROCESSOR_STATS_MISMATCH: EXIT_NUMERICS,
    ErrorCode.MODEL_NUMERICS_FAILED: EXIT_NUMERICS,
}


def exit_code_for(error: ApiError | ErrorCode) -> int:
    """把一个训练层错误映射成规格 §12 的退出码。"""
    code = error.code if isinstance(error, ApiError) else error
    return _EXIT_BY_ERROR.get(code, EXIT_CONTRACT)


__all__ = [
    "EXIT_BLOCKED",
    "EXIT_CONTRACT",
    "EXIT_DATA",
    "EXIT_NUMERICS",
    "EXIT_OK",
    "exit_code_for",
]

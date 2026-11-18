"""``evaluation/1.0`` payload validation (SPEC 10.2).

The evaluation report is an *untrusted import*: every field is validated here
before a report row or asset is created.  No LLM and no heuristic is involved;
the same payload always yields the same collection orders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from embodied_dataops import EVALUATION_SCHEMA_VERSION
from embodied_dataops.canonical.contract import canonical_json_bytes
from embodied_dataops.errors import ApiError, ErrorCode

SCENARIO_KEY_RE = re.compile(r"^[a-z0-9_]{3,64}$")

#: Hard bounds keep a malformed/absurd report from creating nonsense orders.
MAX_RESULTS = 1_000
MAX_TRIALS = 10_000_000


@dataclass(frozen=True)
class ScenarioResult:
    scenario_key: str
    trials: int
    successes: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_key": self.scenario_key,
            "trials": self.trials,
            "successes": self.successes,
        }


@dataclass(frozen=True)
class EvaluationPayload:
    schema_version: str
    model_id: str
    is_simulated: bool
    results: tuple[ScenarioResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "results": [result.as_dict() for result in self.results],
            "is_simulated": self.is_simulated,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_dict())


def _invalid(message: str, details: dict[str, Any] | None = None) -> ApiError:
    return ApiError(
        ErrorCode.EVALUATION_SCHEMA_INVALID,
        message,
        details=details or {},
    )


def validate_scenario_key(value: Any) -> str:
    if not isinstance(value, str) or not SCENARIO_KEY_RE.match(value):
        raise ApiError(
            ErrorCode.SCENARIO_KEY_INVALID,
            "scenario_key must match ^[a-z0-9_]{3,64}$",
            details={"scenario_key": value if isinstance(value, str) else repr(value)},
        )
    return value


def _require_int(value: Any, field_name: str, *, minimum: int, maximum: int) -> int:
    # bool is a subclass of int; reject it explicitly so `true` is not "1 trial".
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{field_name} must be an integer", {"field": field_name, "value": repr(value)})
    if value < minimum or value > maximum:
        raise _invalid(
            f"{field_name} must be between {minimum} and {maximum}",
            {"field": field_name, "value": value},
        )
    return value


def validate_evaluation_payload(payload: Any) -> EvaluationPayload:
    """Validate an ``evaluation/1.0`` payload or raise the SPEC error codes."""
    if not isinstance(payload, dict):
        raise _invalid("evaluation payload must be a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != EVALUATION_SCHEMA_VERSION:
        raise _invalid(
            f"schema_version must be {EVALUATION_SCHEMA_VERSION}",
            {"schema_version": repr(schema_version)},
        )

    model_id = payload.get("model_id", "")
    if not isinstance(model_id, str):
        raise _invalid("model_id must be a string", {"field": "model_id"})
    if len(model_id) > 200:
        raise _invalid("model_id is too long", {"field": "model_id"})

    is_simulated = payload.get("is_simulated")
    if not isinstance(is_simulated, bool):
        raise _invalid(
            "is_simulated must be a boolean (simulated results must be labelled)",
            {"field": "is_simulated", "value": repr(is_simulated)},
        )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise _invalid("results must be a non-empty list", {"field": "results"})
    if len(raw_results) > MAX_RESULTS:
        raise _invalid(f"results must contain at most {MAX_RESULTS} entries", {"field": "results"})

    results: list[ScenarioResult] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            raise _invalid(f"results[{index}] must be an object", {"index": index})
        scenario_key = validate_scenario_key(raw.get("scenario_key"))
        if scenario_key in seen:
            raise _invalid(
                "results contain a duplicate scenario_key",
                {"scenario_key": scenario_key, "index": index},
            )
        seen.add(scenario_key)
        trials = _require_int(raw.get("trials"), f"results[{index}].trials", minimum=1, maximum=MAX_TRIALS)
        successes = _require_int(
            raw.get("successes"), f"results[{index}].successes", minimum=0, maximum=trials
        )
        results.append(ScenarioResult(scenario_key=scenario_key, trials=trials, successes=successes))

    return EvaluationPayload(
        schema_version=schema_version,
        model_id=model_id,
        is_simulated=is_simulated,
        results=tuple(results),
    )

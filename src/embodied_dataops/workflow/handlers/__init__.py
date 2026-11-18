"""Step handler registry (SPEC 6.1).

The engine only knows about :class:`~embodied_dataops.workflow.handlers.base.TaskHandler`
callables.  Each step module lives in its own file and is imported **lazily** by
:func:`get_handler`, so the task engine can run (and its tests can pass) even
while an unrelated handler module is still under development or its optional
third-party dependency is not installed.

Resolution order inside a module:

1. a module-level ``HANDLER`` object or callable,
2. a module-level ``handle`` callable,
3. a :class:`TaskHandler` subclass whose ``step`` matches the requested step.

Importing a step module that does not exist (or that fails to import) raises
:class:`HandlerUnavailable` with the module path and the original cause instead of
silently skipping work.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from embodied_dataops.domain.enums import Step
from embodied_dataops.workflow.handlers.base import HandlerContext, HandlerResult, TaskHandler

HandlerCallable = Callable[[HandlerContext], HandlerResult]

#: Step -> module that implements it.  Modules are imported on first use only.
MODULE_BY_STEP: dict[Step, str] = {
    Step.VALIDATE_SOURCE: "embodied_dataops.workflow.handlers.validate_source",
    Step.IMPORT_EPISODES: "embodied_dataops.workflow.handlers.import_episodes",
    Step.NORMALIZE_EPISODE: "embodied_dataops.workflow.handlers.normalize_episode",
    Step.QUALITY_CHECK: "embodied_dataops.workflow.handlers.quality_check",
    Step.DETECT_OBJECTS: "embodied_dataops.workflow.handlers.detect_objects",
    Step.BUILD_PREVIEW: "embodied_dataops.workflow.handlers.build_preview",
    Step.READY_FOR_REVIEW: "embodied_dataops.workflow.handlers.ready_for_review",
    Step.EXTRACT_KEYFRAMES: "embodied_dataops.workflow.handlers.extract_keyframes",
    Step.VLM_PREANNOTATE_CLIP: "embodied_dataops.workflow.handlers.vlm_preannotate_clip",
    Step.MERGE_CANDIDATE_ANNOTATIONS: (
        "embodied_dataops.workflow.handlers.merge_candidate_annotations"
    ),
    Step.VALIDATE_AI_OUTPUT: "embodied_dataops.workflow.handlers.validate_ai_output",
}


class HandlerUnavailable(RuntimeError):
    """A step has no usable handler (missing module, import error or bad contract)."""


def _coerce_step(step: Step | str) -> Step:
    if isinstance(step, Step):
        return step
    try:
        return Step(step)
    except ValueError as exc:
        raise HandlerUnavailable(f"Unknown task step {step!r}") from exc


def _module_handler(module, step: Step) -> HandlerCallable:
    candidate = getattr(module, "HANDLER", None)
    if candidate is None:
        candidate = getattr(module, "handle", None)
    if candidate is None:
        for value in vars(module).values():
            if not isinstance(value, type) or value is TaskHandler:
                continue
            if not issubclass(value, TaskHandler):
                continue
            if getattr(value, "step", None) != step:
                continue
            try:
                candidate = value()
            except TypeError as exc:
                raise HandlerUnavailable(
                    f"{module.__name__}.{value.__name__} must be instantiable without arguments"
                ) from exc
            break
    if candidate is None:
        raise HandlerUnavailable(f"{module.__name__} does not expose a handler for step {step.value}")

    if isinstance(candidate, TaskHandler):
        return candidate.handle
    if callable(candidate):
        return candidate
    raise HandlerUnavailable(
        f"{module.__name__} handler for {step.value} is neither a TaskHandler nor callable: {candidate!r}"
    )


def _load_handler(step: Step) -> HandlerCallable:
    module_path = MODULE_BY_STEP.get(step)
    if module_path is None:  # pragma: no cover - every Step is mapped
        raise HandlerUnavailable(f"No handler module registered for step {step.value}")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise HandlerUnavailable(
            f"Cannot import handler module {module_path!r} for step {step.value}: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc
    return _module_handler(module, step)


class _HandlerRegistry(dict):
    """``dict[Step, callable]`` that resolves each step on first access."""

    _UNRESOLVED = object()

    def __init__(self) -> None:
        super().__init__((step, self._UNRESOLVED) for step in Step)

    def __getitem__(self, step: Step | str) -> HandlerCallable:
        key = _coerce_step(step)
        try:
            value = dict.__getitem__(self, key)
        except KeyError as exc:
            raise HandlerUnavailable(f"No handler registered for step {step!r}") from exc
        if value is self._UNRESOLVED:
            value = _load_handler(key)
            dict.__setitem__(self, key, value)
        return value

    def get(self, step: Step | str, default: HandlerCallable | None = None) -> HandlerCallable | None:
        try:
            return self[step]
        except HandlerUnavailable:
            return default

    def items(self) -> list[tuple[Step, HandlerCallable]]:  # type: ignore[override]
        return [(step, self[step]) for step in dict.keys(self)]

    def values(self) -> list[HandlerCallable]:  # type: ignore[override]
        return [self[step] for step in dict.keys(self)]

    def resolve_all(self) -> dict[Step, HandlerCallable]:
        """Resolve every step (fails loudly if any handler module is unavailable)."""
        return {step: self[step] for step in Step}

    def install(self, step: Step | str, handler: HandlerCallable):
        """Install a handler without importing the step module; returns the previous value.

        Used by tests to keep the engine runnable while an unrelated step module is
        still under development.  Pair with :meth:`restore`.
        """
        key = _coerce_step(step)
        previous = dict.__getitem__(self, key)
        dict.__setitem__(self, key, handler)
        return previous

    def restore(self, step: Step | str, previous) -> None:
        """Restore the value returned by :meth:`install`."""
        dict.__setitem__(self, _coerce_step(step), previous)


#: Lazily resolved registry exposed to the engine.
HANDLERS: _HandlerRegistry = _HandlerRegistry()


def get_handler(step: Step | str) -> HandlerCallable:
    """Return the callable implementing ``step`` or raise :class:`HandlerUnavailable`."""
    return HANDLERS[step]


__all__ = [
    "HANDLERS",
    "MODULE_BY_STEP",
    "HandlerCallable",
    "HandlerUnavailable",
    "get_handler",
]

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass
class PipelineEvent:
    """
    Every significant occurrence in the pipeline is recorded as an event.
    The event bus accumulates these in memory for the duration of one
    document's processing, then the report builder reads them all at once.
    """
    timestamp: str
    stage: str
    event_type: str       # "stage_start" | "stage_end" | "equation_ok" |
                          # "equation_warning" | "equation_error" |
                          # "cache_hit" | "cache_miss" | "fallback_triggered"
    equation_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """
    In-memory event collector for one pipeline run.
    Not thread-safe — single-document, sequential processing only.
    """
    def __init__(self):
        self._events: list[PipelineEvent] = []
        self._subscribers: list[Callable[[PipelineEvent], None]] = []

    def emit(
        self,
        stage: str,
        event_type: str,
        equation_id: str | None = None,
        **payload,
    ) -> None:
        event = PipelineEvent(
            timestamp=datetime.utcnow().isoformat(),
            stage=stage,
            event_type=event_type,
            equation_id=equation_id,
            payload=payload,
        )
        self._events.append(event)
        for subscriber in self._subscribers:
            subscriber(event)

    def subscribe(self, fn: Callable[[PipelineEvent], None]) -> None:
        self._subscribers.append(fn)

    @property
    def all_events(self) -> list[PipelineEvent]:
        return list(self._events)

    def events_for_equation(self, equation_id: str) -> list[PipelineEvent]:
        return [e for e in self._events if e.equation_id == equation_id]

    def events_by_type(self, event_type: str) -> list[PipelineEvent]:
        return [e for e in self._events if e.event_type == event_type]

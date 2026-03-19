"""Abstract analyzer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from context_profiler.models import APIRequest


@dataclass
class AnalyzerResult:
    """Base result from any analyzer."""

    analyzer_name: str
    summary: dict[str, Any] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseAnalyzer(ABC):
    """All analyzers must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def analyze(self, request: APIRequest) -> AnalyzerResult:
        ...

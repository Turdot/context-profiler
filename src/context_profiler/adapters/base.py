"""Abstract adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from context_profiler.models import APIRequest


class BaseAdapter(ABC):
    """All input adapters must implement this interface."""

    @abstractmethod
    def parse(self, data: dict[str, Any]) -> APIRequest:
        """Parse raw JSON data into a canonical APIRequest."""
        ...

    @abstractmethod
    def can_handle(self, data: dict[str, Any]) -> bool:
        """Return True if this adapter can parse the given data."""
        ...

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class StrategyEntry:
    """Represents a high-level strategy and its relevance rule."""

    text: str
    condition: Optional[Callable[[Dict[str, float]], bool]] = None
    tags: List[str] = field(default_factory=list)

    def is_relevant(self, state: Dict[str, float]) -> bool:
        """Check whether the strategy applies to the given state."""
        if self.condition is None:
            return False
        try:
            return self.condition(state)
        except Exception:
            return False


class StrategyMemory:
    """Store and retrieve high-level startup strategies."""

    def __init__(self) -> None:
        self._strategies: List[StrategyEntry] = []

    def add_strategy(
        self,
        strategy_text: str,
        condition: Optional[Callable[[Dict[str, float]], bool]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Add a new strategy to memory."""
        if tags is None:
            tags = []
        entry = StrategyEntry(text=strategy_text, condition=condition, tags=tags)
        self._strategies.append(entry)

    def get_relevant_strategies(
        self,
        state: Dict[str, float],
        top_n: Optional[int] = None,
    ) -> List[str]:
        """Retrieve strategies that are relevant to the current state."""
        relevant = [entry for entry in self._strategies if entry.is_relevant(state)]

        if top_n is not None:
            relevant = relevant[:top_n]

        return [entry.text for entry in relevant]

    def list_strategies(self) -> List[str]:
        """Return all stored strategy texts."""
        return [entry.text for entry in self._strategies]

    def clear(self) -> None:
        """Clear all stored strategies."""
        self._strategies.clear()

    def find_by_tag(self, tag: str) -> List[str]:
        """Return strategies matching a given tag."""
        return [entry.text for entry in self._strategies if tag in entry.tags]


def default_low_cash_strategy() -> StrategyEntry:
    """Create a default low-cash strategy entry."""
    return StrategyEntry(
        text="If cash is low, avoid marketing-heavy actions.",
        condition=lambda state: state.get("cash", 0.0) < 30000.0,
        tags=["cash", "risk", "marketing"],
    )

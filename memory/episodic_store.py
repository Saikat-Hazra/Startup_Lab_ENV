"""Simple list-based episodic memory."""

from typing import Any, Dict, List


class EpisodicMemory:
    """Stores state-action-reward experiences."""

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.experiences: List[Dict[str, Any]] = []

    def add_experience(self, state: Dict[str, Any], action: str, reward: float) -> None:
        """Store one experience tuple."""
        self.experiences.append(
            {
                "state": dict(state),
                "action": action,
                "reward": float(reward),
            }
        )
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return most recent n experiences."""
        return self.experiences[-max(0, n) :]

    def search_similar(self, state: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        """
        Return k most similar experiences by simple numeric distance.
        """
        if not self.experiences:
            return []

        target = self._state_vector(state)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for exp in self.experiences:
            vec = self._state_vector(exp["state"])
            dist = sum((a - b) ** 2 for a, b in zip(target, vec))
            scored.append((dist, exp))

        scored.sort(key=lambda x: x[0])
        return [item[1] for item in scored[: max(1, k)]]

    def _state_vector(self, state: Dict[str, Any]) -> List[float]:
        startups = state.get("startups", [])
        s0 = startups[0] if len(startups) > 0 else {}
        s1 = startups[1] if len(startups) > 1 else {}
        return [
            float(state.get("market_demand", 0.0)),
            float(s0.get("cash", 0.0)),
            float(s0.get("product_quality", 0.0)),
            float(s1.get("cash", 0.0)),
            float(s1.get("product_quality", 0.0)),
        ]

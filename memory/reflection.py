"""Reflection module for turning experiences into insights."""

from collections import Counter, defaultdict
from typing import Dict, List


class Reflection:
    """Analyze episodic memory and produce list-of-string insights."""

    def analyze(self, experiences: List[Dict]) -> List[str]:
        if len(experiences) < 3:
            return ["Not enough experiences for strong insights yet."]

        insights: List[str] = []

        # Repeated failures by action
        failed = [e for e in experiences if e["reward"] < 0]
        failures_by_action = Counter(e["action"] for e in failed)
        for action, count in failures_by_action.items():
            if count >= 2:
                insights.append(f"Repeated failure detected for action '{action}'.")

        # Cash-threshold style pattern discovery
        grouped = defaultdict(list)
        for e in experiences:
            state = e.get("state", {})
            startups = state.get("startups", [])
            cash_values = [s.get("cash", 0.0) for s in startups if isinstance(s, dict)]
            min_cash = min(cash_values) if cash_values else 0.0
            grouped[e["action"]].append((min_cash, e["reward"]))

        for action, rows in grouped.items():
            low_cash_failures = [cash for cash, reward in rows if cash < 40_000 and reward < 0]
            if len(low_cash_failures) >= 2:
                threshold = int(sum(low_cash_failures) / len(low_cash_failures))
                insights.append(
                    f"Action '{action}' often fails when cash < {threshold}."
                )

        if not insights:
            insights.append("No repeated failure patterns detected.")
        return insights

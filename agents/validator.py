"""Decision validation to reduce avoidable mistakes."""

from typing import Dict, List


class DecisionValidator:
    """Apply guardrails to candidate actions."""

    def __init__(self):
        self.fallback_action = "analyze_market"

    def validate(self, state: Dict, action: str, history: List[Dict]) -> str:
        startups = state.get("startups", [])
        min_cash = min((s.get("cash", 0.0) for s in startups), default=0.0)

        # low cash -> no marketing
        if min_cash < 20_000 and action == "run_marketing":
            return self.fallback_action

        # repeated failures -> avoid same action
        recent = history[-5:]
        failed_same_action = [
            h for h in recent if h.get("action") == action and h.get("reward", 0) < 0
        ]
        if len(failed_same_action) >= 2:
            return self.fallback_action

        return action

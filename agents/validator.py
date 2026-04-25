"""Decision validation to reduce avoidable mistakes."""

from typing import Dict, List


class DecisionValidator:
    """Apply guardrails to candidate actions."""

    def __init__(self):
        self.fallback_action = "analyze_market"
        self.expensive_actions = {"run_marketing": 5000, "build_product": 7000}
        self.cheap_actions = {"analyze_market": 500, "reduce_price": 1000, "improve_quality": 4000}

    def validate(self, state: Dict, action: str, history: List[Dict]) -> str:
        startups = state.get("startups", [])
        min_cash = min((s.get("cash", 0.0) for s in startups), default=0.0)
        max_cash = max((s.get("cash", 0.0) for s in startups), default=100000.0)

        # Rule 1: Low cash → avoid expensive actions
        if min_cash < 20_000:
            if action in self.expensive_actions:
                return self.fallback_action
        
        # Rule 2: Very low cash → only analyze
        if min_cash < 5_000:
            return self.fallback_action

        # Rule 3: Avoid repeating recently failed actions (last 5 steps)
        recent = history[-5:]
        failed_same_action = [
            h for h in recent 
            if h.get("action") == action and h.get("reward", 0) < -0.5
        ]
        if len(failed_same_action) >= 2:
            return "improve_quality"  # Switch to quality building instead

        # Rule 4: If quality is already high, focus on marketing
        avg_quality = sum(s.get("product_quality", 0) for s in startups) / max(len(startups), 1)
        if avg_quality > 80 and action == "improve_quality":
            return "run_marketing"

        # Rule 5: If prices are competitive (low), don't reduce further
        recent_reduce = [h for h in recent[-3:] if h.get("action") == "reduce_price"]
        if len(recent_reduce) >= 2 and action == "reduce_price":
            return "analyze_market"

        # Rule 6: Balance action diversity - avoid same action 3 times in a row
        if len(recent) >= 3:
            last_3_actions = [h.get("action") for h in recent[-3:]]
            if all(a == action for a in last_3_actions):
                return "analyze_market"

        return action


"""Reflection module for turning experiences into insights."""

from collections import Counter, defaultdict
from typing import Dict, List


class Reflection:
    """Analyze episodic memory and produce list-of-string insights."""

    def analyze(self, experiences: List[Dict]) -> List[str]:
        if len(experiences) < 3:
            return ["Not enough experiences for strong insights yet. Keep exploring..."]

        insights: List[str] = []

        # 1. Repeated failures by action
        failed = [e for e in experiences if e["reward"] < 0]
        if failed:
            failures_by_action = Counter(e["action"] for e in failed)
            for action, count in failures_by_action.most_common(2):
                if count >= 2:
                    insights.append(f"⚠️ Action '{action}' failed {count} times. Avoid repeating it.")

        # 2. Success patterns
        successful = [e for e in experiences if e["reward"] > 1.0]
        if successful:
            success_actions = Counter(e["action"] for e in successful)
            best_action = success_actions.most_common(1)
            if best_action:
                action, count = best_action[0]
                insights.append(f"✓ Action '{action}' succeeded {count} times. This strategy is working!")

        # 3. Cash-threshold patterns
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
                    f"💰 Action '{action}' often fails when cash < ${threshold}. Increase cash reserves first."
                )

        # 4. Market sentiment
        recent_rewards = [e["reward"] for e in experiences[-10:]]
        avg_recent_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        if avg_recent_reward > 0.5:
            insights.append("📈 Momentum is positive! Recent decisions are paying off.")
        elif avg_recent_reward < -0.3:
            insights.append("📉 Experiencing losses. Shift to conservative, low-cost actions like analyze_market.")

        # 5. Diversity check
        actions_tried = set(e["action"] for e in experiences[-20:])
        if len(actions_tried) < 3:
            missing = [a for a in ["build_product", "improve_quality", "run_marketing", "reduce_price"] 
                      if a not in actions_tried]
            if missing:
                insights.append(f"🔍 Try exploring other actions: {', '.join(missing)}")

        if not insights:
            insights.append("No strong patterns detected yet. Continue experimenting with different actions.")
        
        return insights


"""LLM-powered controller agent for startup decisions."""

from typing import Any, Dict, List, Optional, Tuple


ALLOWED_ACTIONS = [
    "build_product",
    "improve_quality",
    "run_marketing",
    "reduce_price",
    "analyze_market",
]


def build_prompt(state: Dict[str, Any], insights: List[str]) -> str:
    return f"""
You are managing a startup in a competitive market.

Goal:
Maximize long-term profit while maintaining cash reserves.

Current State:
- Market Demand: {state.get('market_demand', 'unknown')}
- Startup 1 - Cash: ${state.get('startups', [{}])[0].get('cash', 0):.0f}, Quality: {state.get('startups', [{}])[0].get('product_quality', 0):.1f}
- Startup 2 - Cash: ${state.get('startups', [{}, {}])[1].get('cash', 0):.0f}, Quality: {state.get('startups', [{}, {}])[1].get('product_quality', 0):.1f}

Learnings from past failures:
{chr(10).join(f'- {insight}' for insight in insights) if insights else '- No lessons yet, explore carefully'}

Available Actions:
- build_product: Cost $7000, increases quality by 6
- improve_quality: Cost $4000, increases quality by 3
- run_marketing: Cost $5000, increases market demand by 8
- reduce_price: Cost $1000, increases price competitiveness
- analyze_market: Cost $500, gather information

Rules:
1. If your startup is low on cash (<$20K), avoid expensive actions
2. Avoid repeating actions that just failed
3. Balance short-term revenue with long-term quality
4. Think about competitive positioning

First, provide your reasoning in 1-2 sentences explaining your decision.
Then, on a new line, return ONLY the action name.
"""


class ControllerAgent:
    """Decision agent using Gemini with safe fallback behavior."""

    def __init__(self, model: Optional[Any] = None):
        if model is None:
            # Lazy import so tests can use a fake model without Gemini dependency.
            from models.model_interface import ModelInterface

            model = ModelInterface()
        self.model = model
        self.allowed_actions = ALLOWED_ACTIONS
        self.action_success_rate = {action: 0.0 for action in ALLOWED_ACTIONS}
        self.action_attempts = {action: 0 for action in ALLOWED_ACTIONS}

    def select_action(self, state: Dict[str, Any], insights: List[str]) -> Tuple[str, str]:
        """Build prompt, generate candidate action, and validate output."""
        prompt = build_prompt(state, insights)
        raw_response = self.model.generate(prompt).strip()
        return self._parse_response(raw_response, state, insights)

    def refine_action(self, state: Dict[str, Any], insights: List[str]) -> Tuple[str, str]:
        """Retry action generation when previous action was invalid/poor."""
        prompt = (
            f"{build_prompt(state, insights)}\n"
            "Previous action failed or was rejected. Suggest a different, more conservative action.\n"
            "Prioritize: analyze_market (low risk), improve_quality (medium risk), then others.\n"
            "First, provide your reasoning in 1-2 sentences.\n"
            "Then, on a new line, return ONLY the action name."
        )
        raw_response = self.model.generate(prompt).strip()
        return self._parse_response(raw_response, state, insights)

    def _parse_response(self, response: str, state: Dict[str, Any], insights: List[str]) -> Tuple[str, str]:
        lines = response.split('\n')
        reasoning = lines[0] if lines else "No reasoning provided"
        action_text = lines[-1] if len(lines) > 1 else response
        action = self._normalize_action(action_text.lower(), state, insights)
        return action, reasoning

    def _normalize_action(self, action_text: str, state: Dict[str, Any], insights: List[str]) -> str:
        """Normalize and validate action text."""
        for action in self.allowed_actions:
            if action in action_text:
                return action

        # Smarter fallback based on context
        startups = state.get("startups", [])
        min_cash = min((s.get("cash", 0.0) for s in startups), default=100000.0)
        
        # If low cash, be conservative
        if min_cash < 30_000:
            return "analyze_market"
        
        # If many marketing failures, try quality improvement
        recent_insights = " ".join(insights).lower()
        if "run_marketing" in recent_insights and "fail" in recent_insights:
            return "improve_quality"
        
        # Default safe action
        return "analyze_market"


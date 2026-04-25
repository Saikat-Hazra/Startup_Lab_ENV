"""LLM-powered controller agent for startup decisions."""

from typing import Any, Dict, List


ALLOWED_ACTIONS = [
    "build_product",
    "improve_quality",
    "run_marketing",
    "reduce_price",
    "analyze_market",
]


def build_prompt(state: Dict[str, Any], insights: List[str]) -> str:
    return f"""
You are managing a startup.

Goal:
Maximize long-term profit.

State:
{state}

Learnings:
{insights}

Actions:
- build_product
- improve_quality
- run_marketing
- reduce_price
- analyze_market

Rules:
- Avoid repeating failures
- Think long-term

Return ONLY one action.
"""


class ControllerAgent:
    """Decision agent using Gemini with safe fallback behavior."""

    def __init__(self, model: Any | None = None):
        if model is None:
            # Lazy import so tests can use a fake model without Gemini dependency.
            from models.model_interface import ModelInterface

            model = ModelInterface()
        self.model = model
        self.allowed_actions = ALLOWED_ACTIONS

    def select_action(self, state: Dict[str, Any], insights: List[str]) -> str:
        """Build prompt, generate candidate action, and validate output."""
        prompt = build_prompt(state, insights)
        raw_action = self.model.generate(prompt).strip().lower()
        return self._normalize_action(raw_action, state, insights)

    def refine_action(self, state: Dict[str, Any], insights: List[str]) -> str:
        """Retry action generation when previous action was invalid/poor."""
        prompt = (
            f"{build_prompt(state, insights)}\n"
            "Previous action failed. Suggest better action.\n"
            "Return ONLY one action."
        )
        raw_action = self.model.generate(prompt).strip().lower()
        return self._normalize_action(raw_action, state, insights)

    def _normalize_action(self, action_text: str, state: Dict[str, Any], insights: List[str]) -> str:
        for action in self.allowed_actions:
            if action in action_text:
                return action

        # fallback when output is invalid
        recent_failures = " ".join(insights).lower()
        if "run_marketing" in recent_failures:
            return "improve_quality"
        return "analyze_market"

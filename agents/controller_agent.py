"""LLM-powered controller agent for startup decisions."""

from typing import Any, Dict, List, Optional


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

First, provide your reasoning in 1-2 sentences.
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

    def select_action(self, state: Dict[str, Any], insights: List[str]) -> tuple[str, str]:
        """Build prompt, generate candidate action, and validate output."""
        prompt = build_prompt(state, insights)
        raw_response = self.model.generate(prompt).strip()
        return self._parse_response(raw_response, state, insights)

    def refine_action(self, state: Dict[str, Any], insights: List[str]) -> tuple[str, str]:
        """Retry action generation when previous action was invalid/poor."""
        prompt = (
            f"{build_prompt(state, insights)}\n"
            "Previous action failed. Suggest better action.\n"
            "First, provide your reasoning in 1-2 sentences.\n"
            "Then, on a new line, return ONLY the action name."
        )
        raw_response = self.model.generate(prompt).strip()
        return self._parse_response(raw_response, state, insights)

    def _parse_response(self, response: str, state: Dict[str, Any], insights: List[str]) -> tuple[str, str]:
        lines = response.split('\n')
        reasoning = lines[0] if lines else "No reasoning provided"
        action_text = lines[-1] if len(lines) > 1 else response
        action = self._normalize_action(action_text.lower(), state, insights)
        return action, reasoning

    def _normalize_action(self, action_text: str, state: Dict[str, Any], insights: List[str]) -> str:
        for action in self.allowed_actions:
            if action in action_text:
                return action

        # fallback when output is invalid
        recent_failures = " ".join(insights).lower()
        if "run_marketing" in recent_failures:
            return "improve_quality"
        return "analyze_market"

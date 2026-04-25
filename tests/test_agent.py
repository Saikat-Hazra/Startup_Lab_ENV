import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.controller_agent import ControllerAgent
from agents.validator import DecisionValidator


class FakeModel:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt: str) -> str:
        return self.output


def sample_state():
    return {
        "step": 1,
        "market_demand": 100.0,
        "startups": [
            {"cash": 15000.0, "product_quality": 45.0},
            {"cash": 25000.0, "product_quality": 55.0},
        ],
    }


def test_agent_returns_valid_action():
    agent = ControllerAgent(model=FakeModel("run_marketing"))
    action, reasoning = agent.select_action(sample_state(), [])
    assert action in agent.allowed_actions


def test_agent_fallback_for_invalid_output():
    agent = ControllerAgent(model=FakeModel("do_something_weird"))
    action, reasoning = agent.select_action(sample_state(), [])
    assert action in agent.allowed_actions


def test_validator_blocks_marketing_on_low_cash():
    validator = DecisionValidator()
    validated = validator.validate(sample_state(), "run_marketing", history=[])
    assert validated != "run_marketing"


def test_validator_blocks_repeated_failed_action():
    validator = DecisionValidator()
    history = [
        {"action": "build_product", "reward": -1.0},
        {"action": "build_product", "reward": -0.5},
    ]
    validated = validator.validate(sample_state(), "build_product", history=history)
    assert validated == "analyze_market"

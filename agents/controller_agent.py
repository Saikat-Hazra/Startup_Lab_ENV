"""
ControllerAgent for startup simulation.

Implements a multi-role agent architecture:
- Researcher: Analyzes past actions and outcomes
- Planner: Decides high-level strategy
- Executor: Selects final action

Modular design allows replacing role logic with LLM later.
"""

from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np


class Researcher:
    """
    Analyzes past actions and outcomes.
    
    Responsibilities:
    - Identify patterns in successful/failed actions
    - Calculate success rates
    - Detect problematic situations
    """
    
    def __init__(self):
        """Initialize researcher with tracking state."""
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.outcome_log: List[Dict[str, Any]] = []
    
    def analyze(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze historical data from episode.
        
        Args:
            history: List of (state, action, reward) events
        
        Returns:
            Analysis report with insights
        """
        if not history:
            return {
                "total_steps": 0,
                "avg_reward": 0.0,
                "best_action": None,
                "worst_action": None,
                "action_success_rates": {},
                "trend": "neutral",
                "cash_status": "unknown",
            }
        
        actions = [h.get("action", -1) for h in history]
        rewards = [h.get("reward", 0.0) for h in history]
        
        # Basic statistics
        total_steps = len(history)
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        # Action success analysis
        action_rewards = {}
        for action in set(actions):
            indices = [i for i, a in enumerate(actions) if a == action]
            action_rewards[action] = {
                "count": len(indices),
                "avg_reward": np.mean([rewards[i] for i in indices]),
                "success_rate": sum(
                    1 for i in indices if rewards[i] > 0
                ) / len(indices) if indices else 0,
            }
        
        # Identify best and worst actions
        best_action = max(action_rewards, key=lambda a: action_rewards[a]["avg_reward"])
        worst_action = min(action_rewards, key=lambda a: action_rewards[a]["avg_reward"])
        
        # Trend detection (last 5 vs first 5)
        trend = "neutral"
        if len(rewards) >= 5:
            early_avg = np.mean(rewards[:5])
            late_avg = np.mean(rewards[-5:])
            if late_avg > early_avg * 1.1:
                trend = "improving"
            elif late_avg < early_avg * 0.9:
                trend = "declining"
        
        # Cash status from last reward context
        cash_status = "healthy"
        if rewards and rewards[-1] < -5:
            cash_status = "declining"
        
        return {
            "total_steps": total_steps,
            "avg_reward": float(avg_reward),
            "best_action": int(best_action),
            "worst_action": int(worst_action),
            "action_success_rates": action_rewards,
            "trend": trend,
            "cash_status": cash_status,
        }


class Planner:
    """
    Decides high-level strategy based on state and analysis.
    
    Responsibilities:
    - Set strategic goals
    - Determine action priorities
    - Balance short-term and long-term objectives
    """
    
    ACTION_NAMES = {
        0: "build_product",
        1: "improve_quality",
        2: "run_marketing",
        3: "reduce_price",
        4: "analyze_market",
    }
    
    def __init__(self):
        """Initialize planner."""
        self.strategy_log: List[Dict[str, Any]] = []
    
    def plan(
        self,
        state: Dict[str, float],
        analysis: Dict[str, Any],
        insights: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Develop strategy based on current state and analysis.
        
        Args:
            state: Current environment state
            analysis: Analysis report from Researcher
            insights: Learned insights from memory reflection
        
        Returns:
            Strategy recommendation with action priorities
        """
        strategy = {
            "primary_goal": self._determine_goal(state),
            "priorities": self._calculate_priorities(state, analysis),
            "risk_level": self._assess_risk(state),
            "recommended_actions": [],
        }
        
        # Adjust priorities based on insights
        if insights:
            strategy["priorities"] = self._apply_insights(
                strategy["priorities"], insights
            )
        
        # Log strategy
        self.strategy_log.append(strategy)
        
        return strategy
    
    def _determine_goal(self, state: Dict[str, float]) -> str:
        """
        Determine primary goal based on state.
        
        Args:
            state: Current state
        
        Returns:
            Goal string
        """
        cash = state.get("cash", 0)
        quality = state.get("product_quality", 0)
        demand = state.get("market_demand", 0)
        
        # Crisis mode
        if cash < 30000:
            return "survive_cash_crisis"
        
        # Growth mode
        if quality < 50 or demand < 50:
            return "increase_capacity"
        
        # Optimization mode
        return "maximize_profit"
    
    def _calculate_priorities(
        self,
        state: Dict[str, float],
        analysis: Dict[str, Any],
    ) -> Dict[int, float]:
        """
        Calculate action priorities (0-1 scale).
        
        Args:
            state: Current state
            analysis: Analysis report
        
        Returns:
            Dictionary mapping action to priority score
        """
        cash = state.get("cash", 100000)
        quality = state.get("product_quality", 50)
        demand = state.get("market_demand", 50)
        competition = state.get("competition", 50)
        
        priorities = {
            0: 0.0,  # build_product
            1: 0.0,  # improve_quality
            2: 0.0,  # run_marketing
            3: 0.0,  # reduce_price
            4: 0.0,  # analyze_market
        }
        
        # Priority logic based on state
        
        # build_product: High priority if cash is good and quality low
        if cash > 50000 and quality < 60:
            priorities[0] = 0.8
        elif cash > 80000:
            priorities[0] = 0.5
        
        # improve_quality: Important for long-term value
        if quality < 70:
            priorities[1] = 0.7
        elif quality < 90:
            priorities[1] = 0.4
        
        # run_marketing: Critical if demand is low
        if demand < 40:
            priorities[2] = 0.9
        elif demand < 60:
            priorities[2] = 0.6
        elif cash > 100000:
            priorities[2] = 0.3
        
        # reduce_price: Use when market is tough (high competition)
        if competition > 60 and cash > 50000:
            priorities[3] = 0.6
        elif cash < 40000:
            priorities[3] = 0.4  # Quick cash from increased volume
        
        # analyze_market: Low cost, useful for learning
        if cash < 30000:
            priorities[4] = 0.8  # Cheap way to get info
        else:
            priorities[4] = 0.2  # Lower priority when flush with cash
        
        return priorities
    
    def _assess_risk(self, state: Dict[str, float]) -> str:
        """
        Assess overall risk level.
        
        Args:
            state: Current state
        
        Returns:
            Risk level: "high", "medium", "low"
        """
        cash = state.get("cash", 100000)
        
        if cash < 20000:
            return "high"
        elif cash < 50000:
            return "medium"
        else:
            return "low"
    
    def _apply_insights(
        self,
        priorities: Dict[int, float],
        insights: List[str],
    ) -> Dict[int, float]:
        """
        Adjust action priorities based on learned insights.
        
        Args:
            priorities: Current action priorities
            insights: List of insight messages
        
        Returns:
            Modified priorities dict
        """
        adjusted = priorities.copy()
        
        action_names = {
            0: "build_product",
            1: "improve_quality",
            2: "run_marketing",
            3: "reduce_price",
            4: "analyze_market",
        }
        
        # Parse insights to identify problematic actions
        for insight in insights:
            insight_lower = insight.lower()
            
            # Check for "avoid" or "never use" patterns
            if "avoid" in insight_lower or "never use" in insight_lower:
                for action_idx, action_name in action_names.items():
                    if action_name in insight_lower:
                        # Significantly reduce priority
                        adjusted[action_idx] *= 0.3
            
            # Check for recommended actions
            elif "effective" in insight_lower or "use" in insight_lower:
                for action_idx, action_name in action_names.items():
                    if action_name in insight_lower:
                        # Boost priority
                        adjusted[action_idx] *= 1.3
        
        return adjusted


class Executor:
    """
    Selects final action based on strategy.
    
    Responsibilities:
    - Convert priorities to concrete action
    - Handle exploration vs exploitation
    - Ensure action feasibility
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Initialize executor.
        
        Args:
            epsilon: Exploration rate (0-1)
        """
        self.epsilon = epsilon
        self.action_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    def act(
        self,
        priorities: Dict[int, float],
        state: Dict[str, float],
        analysis: Dict[str, Any],
    ) -> int:
        """
        Select action based on priorities.
        
        Balances exploitation (best action) with exploration (random action).
        
        Args:
            priorities: Action priorities from planner
            state: Current state
            analysis: Analysis from researcher
        
        Returns:
            Selected action (0-4)
        """
        # Exploration: random action with epsilon probability
        if np.random.random() < self.epsilon:
            action = np.random.randint(5)
        else:
            # Exploitation: weighted selection based on priorities
            action = self._select_best_action(priorities, state)
        
        # Track action
        self.action_count[action] += 1
        
        return action
    
    def _select_best_action(
        self,
        priorities: Dict[int, float],
        state: Dict[str, float],
    ) -> int:
        """
        Select action with highest priority.
        
        Args:
            priorities: Action priorities
            state: Current state
        
        Returns:
            Best action index
        """
        # Weighted selection with some randomness
        # If multiple actions have similar priority, pick randomly among them
        max_priority = max(priorities.values())
        
        if max_priority == 0:
            # No clear preference, default to analyze_market (cheap and safe)
            return 4
        
        # Get actions within 20% of max priority
        threshold = max_priority * 0.8
        candidates = [a for a, p in priorities.items() if p >= threshold]
        
        # Among candidates, prefer exploring less-used actions
        action_frequencies = [self.action_count.get(a, 0) for a in candidates]
        min_freq = min(action_frequencies)
        best_candidates = [
            candidates[i] for i, freq in enumerate(action_frequencies) if freq == min_freq
        ]
        
        # Return highest priority among best candidates
        return max(best_candidates, key=lambda a: priorities[a])


class ControllerAgent:
    """
    Multi-role agent controller for startup simulation.
    
    Coordinates Researcher, Planner, and Executor to select actions.
    Integrates memory insights to improve decision-making.
    """
    
    def __init__(self, agent_id: str = "controller_0", epsilon: float = 0.1):
        """
        Initialize controller agent.
        
        Args:
            agent_id: Unique agent identifier
            epsilon: Exploration rate for executor
        """
        self.agent_id = agent_id
        self.researcher = Researcher()
        self.planner = Planner()
        self.executor = Executor(epsilon=epsilon)
        
        self.decision_log: List[Dict[str, Any]] = []
        self.insights: List[str] = []  # Store learned insights
        self.failed_actions: set = set()  # Actions to avoid
    
    def select_action(
        self,
        state: Dict[str, float],
        history: List[Dict[str, Any]],
    ) -> int:
        """
        Select action based on state and history.
        
        Coordinates three roles:
        1. Researcher analyzes history
        2. Planner develops strategy (with insights)
        3. Executor selects action
        
        Args:
            state: Current environment state
            history: Episode history
        
        Returns:
            Selected action (0-4)
        """
        # Role 1: Research and analyze
        analysis = self.researcher.analyze(history)
        
        # Role 2: Plan strategy (pass insights)
        strategy = self.planner.plan(state, analysis, insights=self.insights)
        
        # Role 3: Execute action
        action = self.executor.act(strategy["priorities"], state, analysis)
        
        # Log decision
        decision = {
            "action": action,
            "analysis": analysis,
            "strategy": strategy,
            "state_snapshot": state.copy(),
        }
        self.decision_log.append(decision)
        
        return action
    
    def update_strategy(self, insights: List[str]) -> None:
        """
        Update strategy based on learned insights.
        
        Store insights internally to influence future decisions.
        
        Args:
            insights: List of insight strings from reflection module
        """
        self.insights = insights
        
        # Extract actions to avoid from insights
        action_names = {
            0: "build_product",
            1: "improve_quality",
            2: "run_marketing",
            3: "reduce_price",
            4: "analyze_market",
        }
        
        self.failed_actions = set()
        for insight in insights:
            insight_lower = insight.lower()
            if "avoid" in insight_lower or "never" in insight_lower:
                for action_idx, action_name in action_names.items():
                    if action_name in insight_lower:
                        self.failed_actions.add(action_idx)
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent decisions.
        
        Returns:
            Summary of decision history
        """
        if not self.decision_log:
            return {"decisions_made": 0}
        
        recent_decisions = self.decision_log[-5:]
        action_counts = Counter([d["action"] for d in recent_decisions])
        
        return {
            "decisions_made": len(self.decision_log),
            "recent_actions": dict(action_counts),
            "last_strategy": self.decision_log[-1]["strategy"] if self.decision_log else None,
            "last_analysis": self.decision_log[-1]["analysis"] if self.decision_log else None,
        }
    
    def reset(self) -> None:
        """Reset agent for new episode."""
        self.researcher = Researcher()
        self.planner = Planner()
        self.decision_log = []
        # Note: Keep insights across episodes for learning


def demo_controller_agent():
    """Demo the controller agent."""
    print("=" * 70)
    print("CONTROLLER AGENT DEMO")
    print("=" * 70)
    
    agent = ControllerAgent(epsilon=0.15)
    
    # Simulate history
    history = [
        {"action": 2, "reward": 1.5},
        {"action": 1, "reward": 0.8},
        {"action": 0, "reward": 2.3},
        {"action": 4, "reward": -0.5},
    ]
    
    # Simulate states
    states = [
        {
            "market_demand": 45.0,
            "cash": 120000,
            "product_quality": 55.0,
            "competition": 50.0,
            "units_sold": 150.0,
            "price": 50.0,
        },
        {
            "market_demand": 35.0,
            "cash": 25000,  # Crisis
            "product_quality": 40.0,
            "competition": 60.0,
            "units_sold": 80.0,
            "price": 45.0,
        },
    ]
    
    action_names = {
        0: "build_product",
        1: "improve_quality",
        2: "run_marketing",
        3: "reduce_price",
        4: "analyze_market",
    }
    
    for i, state in enumerate(states):
        print(f"\nScenario {i + 1}:")
        print(f"  Cash: ${state['cash']:,.0f}")
        print(f"  Quality: {state['product_quality']:.1f}")
        print(f"  Demand: {state['market_demand']:.1f}")
        
        action = agent.select_action(state, history)
        
        decision = agent.decision_log[-1]
        print(f"\n  Selected Action: {action_names[action]}")
        print(f"  Goal: {decision['strategy']['primary_goal']}")
        print(f"  Risk: {decision['strategy']['risk_level']}")
        print(f"  Trend: {decision['analysis']['trend']}")
    
    summary = agent.get_decision_summary()
    print(f"\nAgent Summary:")
    print(f"  Total Decisions: {summary['decisions_made']}")
    print(f"  Recent Actions: {summary['recent_actions']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_controller_agent()

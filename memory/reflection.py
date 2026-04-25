"""
Reflection module for learning from past experiences.

Analyzes episodic memory to identify patterns, failures, and generate insights.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter


@dataclass
class Insight:
    """Single insight or lesson learned."""
    
    category: str  # "action_failure", "state_danger", "action_pattern", "recovery"
    message: str
    confidence: float  # 0-1, how confident is this insight
    evidence_count: int  # Number of examples supporting this
    severity: str  # "high", "medium", "low"


class Reflection:
    """
    Reflection engine for learning from experiences.
    
    Analyzes episodic memory to generate insights about:
    - Repeated failures
    - Dangerous state-action combinations
    - Patterns and anti-patterns
    - Recovery strategies
    """
    
    ACTION_NAMES = {
        0: "build_product",
        1: "improve_quality",
        2: "run_marketing",
        3: "reduce_price",
        4: "analyze_market",
    }
    
    def __init__(self, min_evidence: int = 2):
        """
        Initialize reflection engine.
        
        Args:
            min_evidence: Minimum number of examples needed for insight
        """
        self.min_evidence = min_evidence
        self.insights: List[Insight] = []
    
    def generate_insights(self, memory) -> List[str]:
        """
        Generate insights from episodic memory.
        
        Args:
            memory: EpisodicMemory instance
        
        Returns:
            List of insight strings
        """
        self.insights = []
        
        if len(memory.experiences) < self.min_evidence:
            return ["Not enough experience to generate insights."]
        
        # Generate different types of insights
        self._find_action_failures(memory)
        self._find_state_dangers(memory)
        self._find_recovery_patterns(memory)
        self._find_action_patterns(memory)
        
        # Format insights as readable strings
        return self._format_insights()
    
    def _find_action_failures(self, memory) -> None:
        """
        Find actions that consistently lead to low rewards.
        
        Args:
            memory: EpisodicMemory instance
        """
        action_rewards = defaultdict(list)
        
        # Group rewards by action
        for exp in memory.experiences:
            action_rewards[exp.action].append(exp.reward)
        
        # Analyze each action
        for action, rewards in action_rewards.items():
            if len(rewards) < self.min_evidence:
                continue
            
            avg_reward = np.mean(rewards)
            success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
            
            # Flag consistently poor actions
            if avg_reward < -1.0 and success_rate < 0.3:
                severity = "high" if avg_reward < -2.0 else "medium"
                insight = Insight(
                    category="action_failure",
                    message=f"Avoid '{self.ACTION_NAMES[action]}' - "
                    f"it leads to low rewards ({avg_reward:.2f} avg)",
                    confidence=min(0.9, success_rate + 0.3),
                    evidence_count=len(rewards),
                    severity=severity,
                )
                self.insights.append(insight)
    
    def _find_state_dangers(self, memory) -> None:
        """
        Find dangerous state-action combinations.
        
        Args:
            memory: EpisodicMemory instance
        """
        # Track bad outcomes for specific situations
        low_cash_bad_actions = Counter()
        low_quality_bad_actions = Counter()
        high_competition_bad_actions = Counter()
        
        for exp in memory.experiences:
            if exp.reward < -1.0:  # Bad outcome
                state = exp.state
                action = exp.action
                
                if state.get("cash", 100000) < 30000:
                    low_cash_bad_actions[action] += 1
                
                if state.get("product_quality", 50) < 40:
                    low_quality_bad_actions[action] += 1
                
                if state.get("competition", 50) > 60:
                    high_competition_bad_actions[action] += 1
        
        # Generate insights from bad combinations
        
        # Low cash + bad action pattern
        if low_cash_bad_actions:
            worst_action = low_cash_bad_actions.most_common(1)[0][0]
            count = low_cash_bad_actions[worst_action]
            
            if count >= self.min_evidence:
                insight = Insight(
                    category="state_danger",
                    message=f"Never use '{self.ACTION_NAMES[worst_action]}' when cash is low - "
                    f"leads to crisis ({count} failures)",
                    confidence=min(0.85, count / 10),
                    evidence_count=count,
                    severity="high",
                )
                self.insights.append(insight)
        
        # Low quality + bad action pattern
        if low_quality_bad_actions:
            worst_action = low_quality_bad_actions.most_common(1)[0][0]
            count = low_quality_bad_actions[worst_action]
            
            if count >= self.min_evidence:
                insight = Insight(
                    category="state_danger",
                    message=f"When product quality is low, avoid '{self.ACTION_NAMES[worst_action]}' - "
                    f"ineffective ({count} poor outcomes)",
                    confidence=min(0.8, count / 10),
                    evidence_count=count,
                    severity="medium",
                )
                self.insights.append(insight)
        
        # High competition + bad action pattern
        if high_competition_bad_actions:
            worst_action = high_competition_bad_actions.most_common(1)[0][0]
            count = high_competition_bad_actions[worst_action]
            
            if count >= self.min_evidence:
                insight = Insight(
                    category="state_danger",
                    message=f"In high competition, '{self.ACTION_NAMES[worst_action]}' doesn't work - "
                    f"try marketing instead ({count} failures)",
                    confidence=min(0.8, count / 10),
                    evidence_count=count,
                    severity="medium",
                )
                self.insights.append(insight)
    
    def _find_recovery_patterns(self, memory) -> None:
        """
        Find actions that help recover from crises.
        
        Args:
            memory: EpisodicMemory instance
        """
        recovery_actions = Counter()
        
        # Look for experiences that improved situation after bad state
        for i, exp in enumerate(memory.experiences[:-1]):
            current_state = exp.state
            next_state = memory.experiences[i + 1].state
            
            # Crisis recovery: low cash -> more cash
            current_cash = current_state.get("cash", 100000)
            next_cash = next_state.get("cash", 100000)
            
            if current_cash < 30000 and next_cash > current_cash and exp.reward > 0:
                recovery_actions[exp.action] += 1
        
        # Generate insight on recovery actions
        if recovery_actions:
            best_action, count = recovery_actions.most_common(1)[0]
            
            if count >= self.min_evidence:
                insight = Insight(
                    category="recovery",
                    message=f"Use '{self.ACTION_NAMES[best_action]}' to recover from cash crisis - "
                    f"works {count} times",
                    confidence=min(0.9, count / 5),
                    evidence_count=count,
                    severity="high",  # Recovery is important
                )
                self.insights.append(insight)
    
    def _find_action_patterns(self, memory) -> None:
        """
        Find effective action sequences and patterns.
        
        Args:
            memory: EpisodicMemory instance
        """
        # Track consecutive successful actions
        high_reward_actions = Counter()
        
        for exp in memory.experiences:
            if exp.reward > 1.5:  # Good outcome
                high_reward_actions[exp.action] += 1
        
        if high_reward_actions:
            best_action, count = high_reward_actions.most_common(1)[0]
            
            if count >= self.min_evidence:
                insight = Insight(
                    category="action_pattern",
                    message=f"'{self.ACTION_NAMES[best_action]}' is effective - "
                    f"successful {count} times",
                    confidence=min(0.85, count / 10),
                    evidence_count=count,
                    severity="low",  # Informational
                )
                self.insights.append(insight)
    
    def _format_insights(self) -> List[str]:
        """
        Format insights as readable strings.
        
        Returns:
            List of formatted insight strings
        """
        if not self.insights:
            return ["No clear patterns found yet. Need more experience."]
        
        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_insights = sorted(
            self.insights,
            key=lambda x: (severity_order.get(x.severity, 3), -x.confidence),
        )
        
        formatted = []
        for insight in sorted_insights:
            formatted.append(f"[{insight.severity.upper()}] {insight.message}")
        
        return formatted
    
    def get_detailed_insights(self) -> List[Dict[str, Any]]:
        """
        Get detailed insight information.
        
        Returns:
            List of insight dictionaries with full metadata
        """
        return [
            {
                "category": insight.category,
                "message": insight.message,
                "confidence": float(insight.confidence),
                "evidence": insight.evidence_count,
                "severity": insight.severity,
            }
            for insight in self.insights
        ]
    
    def summarize_lessons(self) -> str:
        """
        Generate summary of key lessons.
        
        Returns:
            Summary string
        """
        if not self.insights:
            return "No lessons learned yet."
        
        high_severity = [i for i in self.insights if i.severity == "high"]
        
        summary_lines = ["=== KEY LESSONS ==="]
        
        if high_severity:
            summary_lines.append("\n⚠️ CRITICAL:")
            for insight in high_severity[:2]:
                summary_lines.append(f"  • {insight.message}")
        
        medium_severity = [i for i in self.insights if i.severity == "medium"]
        if medium_severity:
            summary_lines.append("\n⚠️ IMPORTANT:")
            for insight in medium_severity[:2]:
                summary_lines.append(f"  • {insight.message}")
        
        summary_lines.append(f"\n📊 Total insights: {len(self.insights)}")
        
        return "\n".join(summary_lines)


def demo_reflection():
    """Demo the reflection module."""
    import sys
    from pathlib import Path
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from memory.episodic_store import EpisodicMemory
    
    print("=" * 70)
    print("REFLECTION MODULE DEMO")
    print("=" * 70)
    
    # Create memory and add experiences
    memory = EpisodicMemory()
    
    # Scenario 1: Low cash + reduce_price fails repeatedly
    low_cash_state = {
        "market_demand": 30.0,
        "cash": 15000.0,
        "product_quality": 40.0,
        "competition": 60.0,
        "units_sold": 50.0,
        "price": 50.0,
    }
    
    # Add bad experiences: reduce_price (action 3) with low cash
    for i in range(4):
        memory.add_experience(
            state=low_cash_state,
            action=3,  # reduce_price
            reward=-2.5,
            next_state=low_cash_state,
        )
    
    # Scenario 2: Good recovery with run_marketing (action 2)
    recovery_state = {
        "market_demand": 35.0,
        "cash": 25000.0,
        "product_quality": 45.0,
        "competition": 55.0,
        "units_sold": 80.0,
        "price": 45.0,
    }
    
    for i in range(3):
        memory.add_experience(
            state=recovery_state,
            action=2,  # run_marketing
            reward=2.0,
            next_state=recovery_state,
        )
    
    # Scenario 3: Effective build_product (action 0)
    good_state = {
        "market_demand": 50.0,
        "cash": 120000.0,
        "product_quality": 60.0,
        "competition": 40.0,
        "units_sold": 150.0,
        "price": 50.0,
    }
    
    for i in range(3):
        memory.add_experience(
            state=good_state,
            action=0,  # build_product
            reward=2.5,
            next_state=good_state,
        )
    
    print(f"\nAdded {len(memory.experiences)} experiences")
    
    # Generate insights
    reflection = Reflection(min_evidence=2)
    insights = reflection.generate_insights(memory)
    
    print("\n" + "=" * 70)
    print("GENERATED INSIGHTS")
    print("=" * 70)
    
    for insight in insights:
        print(f"  {insight}")
    
    # Show detailed insights
    print("\n" + "=" * 70)
    print("DETAILED INSIGHTS")
    print("=" * 70)
    
    detailed = reflection.get_detailed_insights()
    for d in detailed:
        print(f"\nCategory: {d['category']}")
        print(f"Message: {d['message']}")
        print(f"Confidence: {d['confidence']:.2f} | Evidence: {d['evidence']} | Severity: {d['severity']}")
    
    # Show summary
    print("\n" + reflection.summarize_lessons())
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_reflection()

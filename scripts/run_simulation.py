"""
Simulation runner for startup environment demo.

Runs the environment for N episodes with random actions,
printing out actions, rewards, and final states.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.startup_env import StartupEnv


def format_state(state: dict) -> str:
    """Format state dictionary for pretty printing."""
    lines = [
        f"  Market Demand:    {state['market_demand']:.1f}",
        f"  Cash:             ${state['cash']:>12,.0f}",
        f"  Product Quality:  {state['product_quality']:.1f}",
        f"  Competition:      {state['competition']:.1f}",
        f"  Units Sold:       {state['units_sold']:.0f}",
        f"  Price:            ${state['price']:.2f}",
    ]
    return "\n".join(lines)


def format_action(action: int) -> str:
    """Format action index as readable string."""
    actions = {
        0: "build_product",
        1: "improve_quality",
        2: "run_marketing",
        3: "reduce_price",
        4: "analyze_market",
    }
    return actions.get(action, f"unknown_action_{action}")


def run_simulation(num_episodes: int = 10, max_steps: int = None, verbose: bool = True):
    """
    Run startup simulation for specified episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Max steps per episode (None = use env default)
        verbose: Whether to print detailed output
    """
    # Create environment
    env_kwargs = {}
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    
    env = StartupEnv(**env_kwargs)
    
    print("=" * 70)
    print("STARTUP SIMULATION - DEMO RUN")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {env.max_steps}\n")
    
    episode_rewards = []
    episode_profits = []
    episode_lengths = []
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        # Reset environment
        state = env.reset()
        
        episode_reward = 0.0
        episode_profit = 0.0
        actions_taken = []
        
        print("\nInitial State:")
        print(format_state(state))
        
        print("\n" + "-" * 70)
        print("STEP DETAILS")
        print("-" * 70)
        
        # Run episode
        for step in range(env.max_steps):
            # Random action
            action = env.action_space.sample()
            actions_taken.append(action)
            
            # Execute step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_profit += info.get("profit", 0.0)
            
            # Print step information
            if verbose:
                print(
                    f"\nStep {step + 1}:\n"
                    f"  Action:     {format_action(action)}\n"
                    f"  Reward:     {reward:>8.2f}\n"
                    f"  Profit:     ${info.get('profit', 0):>10,.0f}\n"
                    f"  Total Cash: ${info.get('total_cash', 0):>10,.0f}"
                )
            
            state = next_state
            
            if done:
                if verbose:
                    print(f"\n[Episode ended at step {step + 1}]")
                break
        
        # Episode summary
        print("\n" + "-" * 70)
        print("FINAL STATE")
        print("-" * 70)
        print(format_state(state))
        
        print("\n" + "-" * 70)
        print("EPISODE SUMMARY")
        print("-" * 70)
        print(f"  Total Steps:        {step + 1}")
        print(f"  Actions:            {', '.join(format_action(a) for a in actions_taken[:5])}")
        if len(actions_taken) > 5:
            print(f"                      ... ({len(actions_taken) - 5} more)")
        print(f"  Total Reward:       {episode_reward:.2f}")
        print(f"  Total Profit:       ${episode_profit:,.0f}")
        print(f"  Final Cash:         ${state['cash']:,.0f}")
        print(f"  Final Quality:      {state['product_quality']:.1f}")
        print(f"  Final Demand:       {state['market_demand']:.1f}")
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_profits.append(episode_profit)
        episode_lengths.append(step + 1)
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    import numpy as np
    
    print(f"  Total Episodes:           {num_episodes}")
    print(f"  Average Episode Length:   {np.mean(episode_lengths):.1f}")
    print(f"  Average Reward/Episode:   {np.mean(episode_rewards):.2f}")
    print(f"  Max Reward:               {np.max(episode_rewards):.2f}")
    print(f"  Min Reward:               {np.min(episode_rewards):.2f}")
    print(f"  Average Profit/Episode:   ${np.mean(episode_profits):>12,.0f}")
    print(f"  Max Profit:               ${np.max(episode_profits):>12,.0f}")
    print(f"  Min Profit:               ${np.min(episode_profits):>12,.0f}")
    print(f"\nSimulation complete! ✓")
    print("=" * 70)
    
    return {
        "episode_rewards": episode_rewards,
        "episode_profits": episode_profits,
        "episode_lengths": episode_lengths,
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_profit": float(np.mean(episode_profits)),
    }


def run_quick_demo():
    """Run a quick 3-episode demo for testing."""
    print("\n🚀 Running quick demo (3 episodes)...\n")
    return run_simulation(num_episodes=3, verbose=True)


def run_quiet_simulation(num_episodes: int = 10):
    """Run simulation with minimal output."""
    print(f"Running {num_episodes} episodes (quiet mode)...\n")
    results = run_simulation(num_episodes=num_episodes, verbose=False)
    
    print("\nResults Summary:")
    print(f"  Average Reward:  {results['avg_reward']:.2f}")
    print(f"  Average Profit:  ${results['avg_profit']:,.0f}")
    print(f"  Episodes Run:    {num_episodes}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run startup simulation demo")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: 50)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Run in quiet mode with minimal output",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick 3-episode demo",
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_quick_demo()
    elif args.quiet:
        run_quiet_simulation(num_episodes=args.episodes)
    else:
        run_simulation(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            verbose=True,
        )

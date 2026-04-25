# Self-Improving Autonomous Startup Lab

A hackathon project exploring how autonomous agents can run a startup simulation, learn from experience, and improve strategy over time.

This project combines a multi-startup market environment, a role-based decision architecture, episodic memory with FAISS retrieval, and a reflection loop that turns failures into actionable strategy updates.

## Problem Statement

Early-stage startups operate under uncertainty: they must balance product quality, pricing, marketing, and survival with limited cash. Traditional static policies break quickly when market conditions shift.

The challenge we tackle:

- Build autonomous startup agents that can act, learn, and adapt in a dynamic competitive market.
- Move beyond one-step rewards by adding memory and reflection.
- Detect strategy failures under specific conditions and update behavior accordingly.

## Environment Design

The simulation is implemented in `env/startup_env.py` as a Gym-style multi-agent environment.

- **Entities:** Multiple startups competing in one shared market.
- **Per-startup state:** `cash`, `product_quality`, `units_sold`, `price`.
- **Shared market state:** `market_demand`, `competition`.
- **Action space (5 discrete actions):**
  - `build_product`
  - `improve_quality`
  - `run_marketing`
  - `reduce_price`
  - `analyze_market`
- **Dynamics:** Market demand decays over time, competition shifts, quality degrades without investment, and sales are allocated by attractiveness.

This setup creates realistic trade-offs between growth, profitability, and resilience.

## Multi-Agent System

The controller architecture in `agents/controller_agent.py` uses specialized roles:

- **Researcher:** Analyzes action/reward history and trend signals.
- **Planner:** Chooses goals and action priorities from the current state.
- **Executor:** Selects concrete actions with exploration/exploitation behavior.

Each startup has its own controller, and all startups act simultaneously at each step. This produces emergent behavior under competition and provides a clean interface for future LLM-driven policies.

## Memory Architecture

The memory layer (`memory/episodic_store.py` + `memory/reflection.py`) enables learning from experience rather than only immediate rewards.

- **Episodic store**
  - Stores `(state, action, reward, next_state)` experiences.
  - Converts states into normalized numeric embeddings.
  - Indexes embeddings in **FAISS** for fast similarity search.
  - Supports retrieval via `search_similar(state, k=3)`.

- **Reflection engine**
  - Groups repeated failures.
  - Detects condition-action anti-patterns (for example, action failures under low cash thresholds).
  - Produces both human-readable insights and structured outputs:

```json
{
  "condition": "cash < 24000",
  "bad_action": "reduce_price",
  "suggestion": "Try run_marketing instead."
}
```

These insights are fed back into controllers to adapt future decisions.

## Training Results

Training workflow in `training/train.py` includes:

1. **Before-training baseline:** 5 episodes with random policy.
2. **Training loop:** multi-episode learning with replay, memory updates, and reflection.
3. **After-training evaluation:** 5 episodes with trained policy.
4. **Action distribution logging** for both baseline and trained runs.

Generated artifacts (saved to `training_output/`):

- `training_results.json`
- model checkpoints and final models
- `reward_plot.png`
- `average_cash_plot.png`
- `unique_strategies_plot.png`

These outputs make it easy to compare policy quality, cash sustainability, and strategic diversity over episodes.

## Demo Scripts

- `scripts/demo.py`: strategy-aware demo with reflection updates and improvement summary.
- `scripts/demo_run.py`: 2-startup/15-step trace with per-step actions, rewards, insights, and `"Strategy change detected"` events.

## Tech Stack

- Python
- NumPy
- Gymnasium
- Matplotlib
- PyTorch / Transformers (for extended training workflows)
- FAISS (vector memory search)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training:

```bash
python3 training/train.py
```

Run demos:

```bash
python3 scripts/demo.py
python3 scripts/demo_run.py
```

## Future Work

- Replace heuristic planner with LLM-based strategic reasoning.
- Upgrade from tabular Q-learning to deep function approximation with richer state encoding.
- Add long-horizon planning and explicit budget/risk constraints.
- Expand reflection to causal sequence mining (not only condition-action patterns).
- Introduce curriculum scenarios and stress tests (market shocks, demand collapse, adversarial competition).
- Build a dashboard for experiment tracking and side-by-side policy comparison.

## Why This Matters

The core idea is simple but powerful: autonomous agents become significantly more robust when they can remember, retrieve, and reflect on experience.  
This project demonstrates a practical blueprint for self-improving decision systems in dynamic environments.

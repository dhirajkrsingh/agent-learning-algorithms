# Agent Learning Algorithms

How agents learn from experience — reinforcement learning, multi-agent learning, and evolutionary strategies for intelligent agent systems.

## Overview

Learning is what separates reactive agents from truly intelligent ones. This repository covers the core algorithms that allow agents to improve their behavior over time through interaction with their environment and other agents.

```
                    ┌──────────────┐
                    │  Environment │
                    └──────┬───────┘
                           │ state, reward
                           ▼
    ┌─────────────────────────────────────────┐
    │              Learning Agent              │
    │  ┌───────────┐  ┌──────────┐  ┌───────┐│
    │  │  Q-Table / │  │ Policy   │  │Explore││
    │  │  Neural Net│  │ Gradient │  │  vs   ││
    │  │  (Value)   │  │ (Policy) │  │Exploit││
    │  └───────────┘  └──────────┘  └───────┘│
    └─────────────────┬───────────────────────┘
                      │ action
                      ▼
                ┌──────────────┐
                │  Environment │
                └──────────────┘
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Q-Learning** | Model-free RL that learns action-value function from experience |
| **Multi-Agent RL** | Multiple agents learning simultaneously in shared environment |
| **Policy Gradient** | Direct optimization of parameterized policies |
| **Evolutionary Strategy** | Population-based optimization inspired by natural selection |
| **Exploration vs Exploitation** | Balancing learning new things vs using known good strategies |
| **Experience Replay** | Storing and reusing past experiences for efficient learning |

## Examples

| File | Description |
|------|-------------|
| `01_q_learning.py` | Tabular Q-learning agent navigating a grid world |
| `02_multi_agent_learning.py` | Two agents learning to cooperate in a resource gathering game |
| `03_evolutionary_agents.py` | Evolutionary strategy for optimizing agent behavior |

## Best Practices

1. **Start with tabular methods** before moving to function approximation
2. **Tune the exploration rate** — too much exploration wastes time, too little gets stuck
3. **Use experience replay** to break correlation between consecutive samples
4. **Monitor convergence** — plot reward curves to detect instability
5. **In multi-agent settings**, account for non-stationarity (other agents are changing too)
6. **Normalize rewards** to stabilize training across different environments

## References

- [ray-project/ray (RLlib)](https://github.com/ray-project/ray) — Industry-standard RL library
- [openai/gym](https://github.com/openai/gym) — Environment toolkit for RL research
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) — Multi-agent RL environments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — Reliable RL implementations
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — Single-file RL implementations for learning

## Author

Dhiraj Singh

## Usage Notice

This repository is shared publicly for learning and reference.
It is made available for everyone through [VAIU Research Lab](https://vaiu.ai/Research_Lab).
For reuse, redistribution, adaptation, or collaboration, contact Dhiraj Singh / [VAIU Research Lab](https://vaiu.ai/Research_Lab).

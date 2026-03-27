"""
Example 1: Q-Learning Agent
==============================
A tabular Q-learning agent that learns to navigate a grid world,
finding the shortest path from start to goal while avoiding obstacles.

Run: python examples/01_q_learning.py
"""

import random
from collections import defaultdict


class GridWorld:
    """Simple grid environment with obstacles and a goal."""

    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.obstacles = {(1, 1), (2, 2), (3, 1), (1, 3)}
        self.actions = ["up", "down", "left", "right"]
        self.action_effects = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action: str):
        dr, dc = self.action_effects[action]
        new_r = max(0, min(self.size - 1, self.state[0] + dr))
        new_c = max(0, min(self.size - 1, self.state[1] + dc))
        new_state = (new_r, new_c)

        if new_state in self.obstacles:
            reward = -10.0
            new_state = self.state  # Bounce back
        elif new_state == self.goal:
            reward = 100.0
        else:
            reward = -1.0  # Step cost encourages shortest path

        self.state = new_state
        done = (self.state == self.goal)
        return self.state, reward, done

    def render(self, q_table=None):
        """Print the grid with arrows showing learned policy."""
        arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        print(f"\n  {'─' * (self.size * 4 + 1)}")
        for r in range(self.size):
            row = "  │"
            for c in range(self.size):
                pos = (r, c)
                if pos == self.goal:
                    row += " G │"
                elif pos in self.obstacles:
                    row += " █ │"
                elif q_table and pos in q_table:
                    best = max(self.actions, key=lambda a: q_table[pos].get(a, 0))
                    row += f" {arrow[best]} │"
                else:
                    row += " · │"
            print(row)
            print(f"  {'─' * (self.size * 4 + 1)}")


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration."""

    def __init__(self, actions: list, lr: float = 0.1, gamma: float = 0.95, epsilon: float = 1.0):
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})

    def choose_action(self, state) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state].values())

        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes: int = 500):
    env = GridWorld(size=5)
    agent = QLearningAgent(env.actions)

    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while steps < 100:  # Max steps per episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = sum(rewards_history[-50:]) / 50
            print(f"  Episode {ep+1:4d} | Avg Reward (last 50): {avg:7.1f} | Epsilon: {agent.epsilon:.3f}")

    return env, agent, rewards_history


if __name__ == "__main__":
    print("=== Q-Learning Grid World ===\n")

    env, agent, rewards = train(episodes=500)

    print("\nLearned Policy (arrows show best action at each cell):")
    env.render(q_table=dict(agent.q_table))

    # Test the learned policy
    print("\nTest Run (greedy policy):")
    state = env.reset()
    path = [state]
    for _ in range(20):
        q_values = agent.q_table[state]
        action = max(env.actions, key=lambda a: q_values.get(a, 0))
        state, reward, done = env.step(action)
        path.append(state)
        if done:
            break

    print(f"  Path: {' -> '.join(str(p) for p in path)}")
    print(f"  Steps: {len(path) - 1}")
    print(f"  Reached goal: {path[-1] == env.goal}")

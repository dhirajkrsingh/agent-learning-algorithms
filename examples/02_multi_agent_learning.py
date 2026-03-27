"""
Example 2: Multi-Agent Learning
==================================
Two agents learn to cooperate in a resource-gathering game.
Each agent has limited visibility and must learn to share resources.

Run: python examples/02_multi_agent_learning.py
"""

import random
from collections import defaultdict


class ResourceWorld:
    """Grid world where two agents gather resources cooperatively."""

    def __init__(self, size: int = 6):
        self.size = size
        self.actions = ["up", "down", "left", "right", "gather"]
        self.action_effects = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
            "gather": (0, 0),
        }
        self.reset()

    def reset(self):
        self.agents = {
            "A": {"pos": (0, 0), "inventory": 0},
            "B": {"pos": (self.size - 1, self.size - 1), "inventory": 0},
        }
        self.resources = {}
        self._spawn_resources(8)
        self.step_count = 0
        return self._get_observations()

    def _spawn_resources(self, count: int):
        occupied = {a["pos"] for a in self.agents.values()}
        for _ in range(count):
            while True:
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if pos not in occupied and pos not in self.resources:
                    self.resources[pos] = random.randint(1, 3)
                    break

    def _get_observations(self) -> dict:
        """Each agent sees its position, nearby resources, and partner location."""
        obs = {}
        for name, agent in self.agents.items():
            r, c = agent["pos"]
            nearby = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if (nr, nc) in self.resources:
                            nearby.append(((nr, nc), self.resources[(nr, nc)]))
            partner = [a["pos"] for n, a in self.agents.items() if n != name][0]
            obs[name] = (agent["pos"], tuple(sorted(nearby)[:3]), partner)
        return obs

    def step(self, actions: dict):
        """Both agents act simultaneously."""
        self.step_count += 1
        rewards = {name: -0.1 for name in self.agents}  # Step cost

        for name, action in actions.items():
            agent = self.agents[name]
            if action == "gather":
                if agent["pos"] in self.resources:
                    value = self.resources.pop(agent["pos"])
                    agent["inventory"] += value
                    rewards[name] += value * 5
                    # Cooperation bonus: both get reward when either gathers
                    for other_name in self.agents:
                        if other_name != name:
                            rewards[other_name] += value * 2
            else:
                dr, dc = self.action_effects[action]
                new_r = max(0, min(self.size - 1, agent["pos"][0] + dr))
                new_c = max(0, min(self.size - 1, agent["pos"][1] + dc))
                agent["pos"] = (new_r, new_c)

        # Respawn resources occasionally
        if self.step_count % 10 == 0 and len(self.resources) < 5:
            self._spawn_resources(3)

        done = self.step_count >= 50
        obs = self._get_observations()
        return obs, rewards, done


class IndependentQLearner:
    """Q-learning agent for multi-agent setting (independent learner)."""

    def __init__(self, name: str, actions: list, lr=0.1, gamma=0.9, epsilon=1.0):
        self.name = name
        self.actions = actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.q_table = defaultdict(lambda: {a: 0.0 for a in actions})

    def _state_key(self, obs):
        """Discretize observation into a hashable state."""
        pos, nearby_resources, partner_pos = obs
        has_resource_here = any(r[0] == pos for r in nearby_resources)
        # Simplified: just use position + whether resource is here + relative partner direction
        pr, pc = partner_pos
        partner_dir = (
            "above" if pr < pos[0] else "below" if pr > pos[0] else "same_row",
            "left" if pc < pos[1] else "right" if pc > pos[1] else "same_col",
        )
        return (pos, has_resource_here, len(nearby_resources) > 0, partner_dir)

    def choose_action(self, obs) -> str:
        state = self._state_key(obs)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.q_table[state]
        max_q = max(q_values.values())
        best = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best)

    def learn(self, obs, action, reward, next_obs, done):
        state = self._state_key(obs)
        next_state = self._state_key(next_obs)
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.gamma * max(self.q_table[next_state].values())
        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes: int = 300):
    env = ResourceWorld(size=6)
    agents = {
        "A": IndependentQLearner("A", env.actions),
        "B": IndependentQLearner("B", env.actions),
    }

    history = []

    for ep in range(episodes):
        obs = env.reset()
        total_rewards = {"A": 0, "B": 0}
        total_gathered = {"A": 0, "B": 0}

        while True:
            actions = {name: agents[name].choose_action(obs[name]) for name in agents}
            next_obs, rewards, done = env.step(actions)

            for name in agents:
                agents[name].learn(obs[name], actions[name], rewards[name], next_obs[name], done)
                total_rewards[name] += rewards[name]

            for name in env.agents:
                total_gathered[name] = env.agents[name]["inventory"]

            obs = next_obs
            if done:
                break

        for a in agents.values():
            a.decay_epsilon()

        team_reward = sum(total_rewards.values())
        team_gathered = sum(total_gathered.values())
        history.append((team_reward, team_gathered))

        if (ep + 1) % 50 == 0:
            recent = history[-25:]
            avg_reward = sum(h[0] for h in recent) / len(recent)
            avg_gathered = sum(h[1] for h in recent) / len(recent)
            print(f"  Episode {ep+1:4d} | Avg Team Reward: {avg_reward:6.1f} | Avg Gathered: {avg_gathered:4.1f} | ε: {agents['A'].epsilon:.3f}")

    return agents, history


if __name__ == "__main__":
    print("=== Multi-Agent Cooperative Learning ===\n")

    agents, history = train(episodes=300)

    # Show improvement
    first_50 = history[:50]
    last_50 = history[-50:]
    print(f"\n  First 50 episodes avg reward:  {sum(h[0] for h in first_50)/50:.1f}")
    print(f"  Last 50 episodes avg reward:   {sum(h[0] for h in last_50)/50:.1f}")
    print(f"  First 50 episodes avg gather:  {sum(h[1] for h in first_50)/50:.1f}")
    print(f"  Last 50 episodes avg gather:   {sum(h[1] for h in last_50)/50:.1f}")
    print(f"\n  Q-table sizes: A={len(agents['A'].q_table)}, B={len(agents['B'].q_table)}")

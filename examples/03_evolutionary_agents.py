"""
Example 3: Evolutionary Agents
================================
An evolutionary strategy that evolves a population of agents,
selecting the fittest to reproduce and mutate.

Run: python examples/03_evolutionary_agents.py
"""

import random
import math
from copy import deepcopy


class NeuralAgent:
    """Simple agent with a weight-based decision function (no numpy needed)."""

    def __init__(self, input_size: int = 4, hidden_size: int = 6, output_size: int = 4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initialize weights randomly
        self.w1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.w2 = [[random.gauss(0, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [random.gauss(0, 0.1) for _ in range(output_size)]
        self.fitness = 0.0

    def forward(self, inputs: list) -> list:
        """Forward pass through the network."""
        # Hidden layer with tanh activation
        hidden = []
        for j in range(self.hidden_size):
            val = self.b1[j]
            for i in range(self.input_size):
                val += inputs[i] * self.w1[i][j]
            hidden.append(math.tanh(val))

        # Output layer with softmax
        output = []
        for j in range(self.output_size):
            val = self.b2[j]
            for i in range(self.hidden_size):
                val += hidden[i] * self.w2[i][j]
            output.append(val)

        # Softmax
        max_val = max(output)
        exp_vals = [math.exp(v - max_val) for v in output]
        total = sum(exp_vals)
        return [v / total for v in exp_vals]

    def decide(self, inputs: list) -> int:
        """Choose action based on highest probability."""
        probs = self.forward(inputs)
        return probs.index(max(probs))

    def mutate(self, rate: float = 0.1, strength: float = 0.3):
        """Apply random mutations to weights."""
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                if random.random() < rate:
                    self.w1[i][j] += random.gauss(0, strength)
        for j in range(self.hidden_size):
            if random.random() < rate:
                self.b1[j] += random.gauss(0, strength)
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                if random.random() < rate:
                    self.w2[i][j] += random.gauss(0, strength)
        for j in range(self.output_size):
            if random.random() < rate:
                self.b2[j] += random.gauss(0, strength)


class ForagingEnvironment:
    """Grid environment where agents forage for food."""

    def __init__(self, size: int = 10, num_food: int = 15):
        self.size = size
        self.num_food = num_food
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down
        self.reset()

    def reset(self):
        self.food = set()
        while len(self.food) < self.num_food:
            self.food.add((random.randint(0, self.size - 1), random.randint(0, self.size - 1)))
        self.agent_pos = (self.size // 2, self.size // 2)
        return self._get_obs()

    def _get_obs(self) -> list:
        """Observation: direction to nearest food (dx, dy normalized), distance, num nearby."""
        if not self.food:
            return [0.0, 0.0, 1.0, 0.0]

        r, c = self.agent_pos
        nearest = min(self.food, key=lambda f: abs(f[0] - r) + abs(f[1] - c))
        dist = abs(nearest[0] - r) + abs(nearest[1] - c)
        dx = (nearest[1] - c) / max(1, self.size)
        dy = (nearest[0] - r) / max(1, self.size)
        nearby = sum(1 for f in self.food if abs(f[0] - r) + abs(f[1] - c) <= 3)
        return [dx, dy, dist / self.size, nearby / self.num_food]

    def step(self, action_idx: int):
        dr, dc = self.actions[action_idx]
        r = max(0, min(self.size - 1, self.agent_pos[0] + dr))
        c = max(0, min(self.size - 1, self.agent_pos[1] + dc))
        self.agent_pos = (r, c)

        reward = -0.01  # Step cost
        if self.agent_pos in self.food:
            self.food.discard(self.agent_pos)
            reward = 10.0

        return self._get_obs(), reward


def evaluate_agent(agent: NeuralAgent, env: ForagingEnvironment, steps: int = 100) -> float:
    """Evaluate fitness of an agent over multiple steps."""
    obs = env.reset()
    total_reward = 0.0
    for _ in range(steps):
        action = agent.decide(obs)
        obs, reward = env.step(action)
        total_reward += reward
    return total_reward


class EvolutionaryTrainer:
    """Evolve a population of agents using tournament selection."""

    def __init__(self, pop_size: int = 30, elite_frac: float = 0.2,
                 mutation_rate: float = 0.15, mutation_strength: float = 0.3):
        self.pop_size = pop_size
        self.elite_count = max(2, int(pop_size * elite_frac))
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.population = [NeuralAgent() for _ in range(pop_size)]
        self.generation = 0
        self.best_fitness_history = []

    def evaluate_population(self, env: ForagingEnvironment, eval_episodes: int = 3):
        """Evaluate all agents."""
        for agent in self.population:
            fitness = sum(evaluate_agent(agent, env) for _ in range(eval_episodes)) / eval_episodes
            agent.fitness = fitness

    def select_and_reproduce(self):
        """Tournament selection + elitism + crossover + mutation."""
        # Sort by fitness
        self.population.sort(key=lambda a: a.fitness, reverse=True)
        best = self.population[0].fitness
        self.best_fitness_history.append(best)

        # Elite survive unchanged
        new_pop = [deepcopy(self.population[i]) for i in range(self.elite_count)]

        # Fill rest with tournament selection + mutation
        while len(new_pop) < self.pop_size:
            # Tournament: pick 3 random, choose best
            candidates = random.sample(self.population, min(3, len(self.population)))
            parent = max(candidates, key=lambda a: a.fitness)
            child = deepcopy(parent)
            child.mutate(self.mutation_rate, self.mutation_strength)
            child.fitness = 0.0
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        return best

    def train(self, generations: int = 50, env: ForagingEnvironment = None):
        if env is None:
            env = ForagingEnvironment()

        print(f"  Population: {self.pop_size}, Elite: {self.elite_count}")
        print(f"  Mutation rate: {self.mutation_rate}, Strength: {self.mutation_strength}\n")

        for gen in range(generations):
            self.evaluate_population(env)
            best_fitness = self.select_and_reproduce()

            if (gen + 1) % 10 == 0:
                avg_fitness = sum(a.fitness for a in self.population) / self.pop_size
                print(f"  Gen {gen+1:3d} | Best: {best_fitness:7.1f} | Avg: {avg_fitness:7.1f}")

        return self.population[0]


if __name__ == "__main__":
    print("=== Evolutionary Agent Training ===\n")

    env = ForagingEnvironment(size=10, num_food=15)
    trainer = EvolutionaryTrainer(pop_size=30, mutation_rate=0.15)

    best_agent = trainer.train(generations=50, env=env)

    # Demonstrate best agent
    print(f"\n--- Best Agent Demo ---")
    obs = env.reset()
    food_collected = 0
    for step in range(100):
        action = best_agent.decide(obs)
        obs, reward = env.step(action)
        if reward > 0:
            food_collected += 1
            print(f"  Step {step:3d}: Collected food at {env.agent_pos} (total: {food_collected})")

    print(f"\n  Total food collected: {food_collected}")
    print(f"  Best fitness over generations: {trainer.best_fitness_history[-1]:.1f}")
    improvement = trainer.best_fitness_history[-1] - trainer.best_fitness_history[0]
    print(f"  Improvement from gen 1: {'+' if improvement > 0 else ''}{improvement:.1f}")

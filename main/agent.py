from typing import Tuple
from environment import TaskEnv
import numpy as np
import random
from abc import ABC, abstractmethod


class TDAgent(ABC):

    def __init__(self, env: TaskEnv, exploration_rate: float,
                 learning_rate: float, discount_factor: float):
        self.epsilon = exploration_rate
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.q_table = np.zeros(
            (env.observation_space.n + 1, env.action_space.n), dtype=float)
        self.actions = env.action_space
        self.states = env.observation_space

    def select_action(self,
                      state: Tuple[int, int],
                      use_greedy_strategy: bool = False) -> int:
        if not use_greedy_strategy:
            if random.random() < self.epsilon:
                next_action = self.actions.sample()
                return next_action

        x = state
        max_val = np.max(self.q_table[x, :])
        find_max_val = np.where(self.q_table[x, :] == max_val)
        next_action = np.random.choice(find_max_val[0])

        # Use this instead --- NO DON'T DO THAT
        # next_action = np.argmax(self.q_table[x])

        return next_action

    @abstractmethod
    def learn(self, state: Tuple[int, int], action: int,
              next_state: Tuple[int, int], reward: float, done: bool) -> int:
        pass


class QAgent(TDAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, state: Tuple[int, int], action: int,
              next_state: Tuple[int, int], reward: float, done: bool) -> int:
        # Update the Q based on the result
        best_q = np.amax(self.q_table[next_state])
        cell_to_update = (state, action)
        if done:
            self.q_table[cell_to_update] += self.alpha * (
                reward - self.q_table[cell_to_update])
        else:
            self.q_table[cell_to_update] += self.alpha * (
                reward + (self.gamma * best_q) - self.q_table[cell_to_update])
        return None


class SarsaAgent(TDAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, state: Tuple[int, int], action: int,
              next_state: Tuple[int, int], reward: float, done: bool) -> int:
        # Update the Q based on the result
        next_action = self.select_action(next_state, use_greedy_strategy=False)
        next_q = self.q_table[(next_state, next_action)]
        cell_to_update = (state, action)
        if done:
            self.q_table[cell_to_update] += self.alpha * (
                reward - self.q_table[cell_to_update])
        else:
            self.q_table[cell_to_update] += self.alpha * (
                reward + (self.gamma * next_q) - self.q_table[cell_to_update])
        return next_action


class ExpectedSarsaAgent(TDAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn(self, state: Tuple[int, int], action: int,
              next_state: Tuple[int, int], reward: float, done: bool) -> int:
        # Update the Q based on the result
        next_q = np.mean(self.q_table[next_state])
        cell_to_update = (state, action)
        if done:
            self.q_table[cell_to_update] += self.alpha * (
                reward - self.q_table[cell_to_update])
        else:
            self.q_table[cell_to_update] += self.alpha * (
                reward + (self.gamma * next_q) - self.q_table[cell_to_update])
        return None


class RandomAgent(TDAgent):

    def __init__(self,
                 env: TaskEnv,
                 exploration_rate: float = None,
                 learning_rate: float = None,
                 discount_factor: float = None) -> int:
        self.epsilon = 1  # A random agent "explores" always, so epsilon will be 1
        self.alpha = 0  # A random agent never learns, so there's no need for a learning rate
        self.gamma = 0  # A random agent does not update it's q-table. Hence, it's zero.
        self.q_table = np.zeros(
            (env.observation_space.n + 1, env.action_space.n), dtype=float)
        self.actions = env.action_space

    def select_action(self,
                      state: Tuple[int, int],
                      use_greedy_strategy: bool = False) -> int:
        if not use_greedy_strategy:
            if random.random() < self.epsilon:
                next_action = self.actions.sample()
                return next_action

        x = state
        max_val = np.max(self.q_table[x, :])
        find_max_val = np.where(self.q_table[x, :] == max_val)
        next_action = np.random.choice(find_max_val[0])
        return next_action

    def learn(self, state, action, next_state, reward, done):
        return None


class MostFrequentPolicyAgent(TDAgent):

    def __init__(self,
                 env: TaskEnv,
                 exploration_rate: float = None,
                 learning_rate: float = None,
                 discount_factor: float = None) -> int:
        self.epsilon = 0  # A random agent "explores" always, so epsilon will be 1
        self.alpha = 0  # A random agent never learns, so there's no need for a learning rate
        self.gamma = 0  # A random agent does not update it's q-table. Hence, it's zero.
        self.q_table = np.zeros(env.observation_space.shape +
                                (env.action_space.n, ),
                                dtype=float)
        self.actions = env.action_space
        self.default_action = env.act2idx[env.motions[1]]

    def select_action(self,
                      state: Tuple[int, int],
                      use_greedy_strategy: bool = False) -> int:
        return self.default_action

    def learn(self, state, action, next_state, reward, done):
        return None


# https://gibberblot.github.io/rl-notes/single-agent/MDPs.html
class PolicyIterationAgent(TDAgent):

    def __init__(self, env: TaskEnv, exploration_rate: float,
                 learning_rate: float, discount_factor: float):
        super().__init__(env, exploration_rate, learning_rate, discount_factor)
        self.state_values = np.zeros(self.states.n+1)
        self.env = env
        self.policy_iteration()

    def policy_evaluation(self, theta=0.01):
        while True:
            delta = 0.0
            for state in range(self.states.n):
                old_value = self.state_values[state]
                action = self.select_action(state, True)
                new_value = self.get_q_value(state, action)
                self.state_values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:
                break

    def policy_iteration(self, max_iterations=100, theta=0.001):
        for i in range(1, max_iterations + 1):
            policy_changed = False
            self.policy_evaluation()
            for state in range(self.states.n):
                old_action = self.select_action(state, True)
                for action in range(self.actions.n):
                    new_value = self.get_q_value(state, action)
                    self.q_table[state, action] = self.q_table[state, action] + self.alpha * new_value
                new_action = self.select_action(state, True)
                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )
            if not policy_changed:
                return i
        return max_iterations

    def get_q_value(self, state, action):
        q_value = 0 
        for next_state in range(self.states.n+1):
            probability = self.env.transition_probability(
                        action, state, next_state)
            reward = self.env.reward_function(
                        action, state, next_state)
            q_value += probability * (reward + (self.gamma * self.q_table[state, action]))
        return q_value
                    
    def learn(self, state, action, next_state, reward, done):
        return None


if __name__ == "__main__":
    env = TaskEnv(frequencies_file="data/frequencies_final_3.csv")
    # agent = agents.RandomAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)
    # agent = agents.SarsaAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)
    agent = PolicyIterationAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9)
# %%
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import gym


from IPython import get_ipython
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from IPython import display
import seaborn as sns
from tqdm.notebook import tqdm
import environment as envs
from typing import Tuple, List
import itertools as it
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.gridspec as gridspec
import agent as agents

import importlib

importlib.reload(envs)
importlib.reload(agents)


def visualize_agent_brain(agent, env: envs.TaskEnv):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title("Highest state value at position (x,y)")
    path = env.maze.objects.free.positions

    max_qs = agent.q_table.max(axis=2)
    state_value_map = np.ones_like(max_qs) * ((max_qs.min() * 1.1))
    state_value_map[path[:, 0], path[:, 1]] = max_qs[path[:, 0], path[:, 1]]
    sns.heatmap(state_value_map, ax=ax1, cmap="viridis")

    ax2.set_title("Chosen action at position (x,y)")
    n = env.action_space.n + 1

    decisions_map = np.array([[x_, y_, agent.select_action([x_, y_], True) + 1] for x_, y_ in path])
    state_action_map = np.zeros_like(agent.q_table.max(axis=2))
    state_action_map[decisions_map[:, 0], decisions_map[:, 1]] = decisions_map[:, 2]
    cmap = sns.color_palette("viridis", n)
    sns.heatmap(state_action_map, cmap=cmap, ax=ax2)
    colorbar = ax2.collections[0].colorbar
    r = (colorbar.vmax) - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(["N/A", "north", "south", "west", "east"])
    fig.tight_layout()
    return plt.show()


env = envs.TaskEnv(time_out=6, frequencies_file="../data/frequencies_final_3.csv")
env.reset()
# %% [markdown]
#

def run_real_episode(agent: agents.TDAgent, env: envs.TaskEnv):
    next_action = None
    done = False
    current_state = env.current_position
    total_reward = 0
    while not done:
        next_action = agent.select_action(current_state, True)
        next_state, reward, done, _ = env.step(next_action)
        total_reward += reward
        current_state = next_state
    return total_reward


def run_training_episode(agent: agents.TDAgent, env: envs.TaskEnv):
    next_action = None
    done = False
    current_state = env.reset()
    total_reward = 0
    while not done:
        next_action = agent.select_action(current_state) if not next_action else next_action
        next_state, reward, done, _ = env.step(next_action)
        total_reward += reward 
        # print(reward)
        next_action = agent.learn(current_state, next_action, next_state, reward, done)
        current_state = next_state

    last_state = current_state
    return total_reward, last_state, agent


def animate_run(data: List[np.ndarray]):
    remaining_img = data
    ax = plt.gca()  # only call this once
    ax.axis([0, 10, 0, 10])

    for idx, ev in enumerate(remaining_img):
        ax.text(
            idx,
            8,
            ev,
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
        )
        display.display(plt.gcf())
        display.clear_output(wait=True)


def show_trained_agent(agent: agents.TDAgent, env: envs.TaskEnv):
    procedure = []
    next_action = None
    done = False
    current_state = env.reset()
    procedure.append(current_state)
    s_total_reward = 0
    while not done:
        next_action = agent.select_action(current_state, use_greedy_strategy=True)
        next_state, reward, done, _ = env.step(next_action)
        procedure.extend((next_action, next_state))

        s_total_reward += reward
        current_state = next_state
    # env.close()print(rewards)
    # animate_run(procedure)
    plt.show()


# env = TaskEnv(timeout_reward=-1, goal_reward=1, invalid_reward=-1, time_reward_multiplicator=.01)
env = envs.TaskEnv(frequencies_file="../data/frequencies_final_3.csv")
# agent = agents.RandomAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)
# agent = agents.SarsaAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)
# agent = agents.QAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9)
# agent = agents.ExpectedSarsaAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)

# %%
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
importlib.reload(envs)
importlib.reload(agents)
env.reset()
agent = agents.PolicyIterationAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9)
for i in tqdm(range(1000)): 
    total_reward, lat_state, agent = run_training_episode(agent, env)
# show_trained_agent(agent, env)
agent.state_values

#TODO: Reset tohe Q-Table
# %%

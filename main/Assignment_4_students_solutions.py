# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Assignment 4 – Finding the way out in a maze
#
# *Due: Friday April 9 at 17:00 CET*
#
# In the forth assignment, you are going to use RL algorithms to solve the practical problem of ‘robot in a maze’.
# The project is divided into two parts:
#
# -	In the first part, you need to get familier with the OpenAI gym framework.
# -	In the second part, based on the gym framework, you will implement your own RL agents and train them to find the shortest route to get out of a maze.
#
# %% [markdown]
# ## 1. Let's start with the OpenAI gym
#
# Gym (https://gym.openai.com/) is a wide-used toolkit for developing and comparing reinforcement learning algorithms.
#
# 1. Gym makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano.
#
# 2. The gym library is a collection of test problems — **environments** — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.
#
# First, we download & install the gym library. Then import the gym class.

# %%
import gym

# %% [markdown]
# We are now going to explain how the RL framework of gym works.
# - An **ENVIRONMENT**,
# - You also have an **AGENT**,
# - In MDP problems (like ours), the **ENVIRONMENT** will also provides an **OBSERVATION**, which represets the state of the **ENVIRONMENT** at the current moment.
# - The agent takes an **ACTION** based on its **OBSERVATION**,
# - When a single **ACTION** is chosen and fed to our **ENVIRONMENT**, the **ENVIRONMENT** measures how good the action was taken and produces a **REWARD**, which is usually a numeric value.
#
# Please read the 'Getting Started with gym' https://gym.openai.com/docs/ for better understand the framework.
# %% [markdown]
#  ## 2. Go back to our own task
#
#  Next, you will solve a practical MDP problem 'robot in a maze' based on the gym framework. You shall implement the RL agent and train it to find the shortest route to achieve the maze goal. In this MDP, the enviroment is a grid world (a maze) while the agent is a robot. At each time step, the robot starts at a random location and can move around in the grid world. The long-term objective is finding the way out (reaching the final location). Hence, you need to find a fixed goal position within the maze.
# %% [markdown]
# ### 2.0 Model the practical task into a MDP
#
# To solve a RL problem, we start with formalize the problem into a MDP model. Notice: No empricial data provided in this assignment, so the point of 'data description and exploration' will be given to this step.
#
# While exploring your MDP model, you may also think about questions such as:
# - What is the environment? How does it look like?
# - What is the simulated data?
# - What simulated data can your RL agent observe from the environment?
# - Which data is considered as the state? Which data is considered as the reward?
# %% [markdown]
# ### 2.1 Set up the environment
#
# There is no need to implement your own environment. You can use the environment we provide in the file **environment.py**. (Make sure to have a look at it)
#
# The core gym interface is **Env**, which is the unified environment interface. There is no interface for agents. The following are the Env methods you should know:
#
# - reset(self): Reset the environment's state. Returns observation.
# - step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
# - render(self, mode='rgb_array'): Render one frame of the environment. The default mode will do something human friendly, such as pop up a window. In this assignment, there is no need to create a pop up window.
#

# %% [markdown]
# We also provide a few helper functions to make it easier to debug your agents.
#  - `animate_run` will enable you to see the agent's behavior. It takes a list of images which can be produced by the `env.render` function of the environment
#  - `visualize_agent_brain` will provide you with a way to visualize the agents learned q_table. Use it after you have implemented and trained your agents. The first plot will show the highest q-value per state (position on the map) and the second will tell you which action the agent would choose at that state/position. It takes the environment and the agent as input.
#
# Below you will find a basic example of how the animation function works. Please notice that: whenever you **reset()** the environment, the agent will start at a random position (a different state).

# %%

# %%
# The helper functions

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
# %% [markdown]
# ### 2.2  Implement the agents
#
# In this part, you are expected to implement two RL agents.
#
# - Agent 1 uses the Q-learning algorithm to learn the optimal solution
# - Agent 2 uses the SARSA algorithm to learn the optimal solution. To decide the action to take at each time step, the this agent uses the epsilon greedy action selection.
#
# Here we also provided an example agent: Random Agent. It follows a random policy to move at each step (randomly select the action).
#

# %%
# TODO: implement two agents

# %% [markdown]
# ### 2.3 Run the simulation
#
# Finally, we write the codes for running a simulation. In each run, you shall setup the epsilon parameter.

# %%
# TODO: run the simulation


def run_real_episode(agent: agents.TDAgent, env: envs.TaskEnv):
    next_action = None
    done = False
    current_state = env.reset()
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
        total_reward += reward / env.timer
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
agent = agents.QAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9)
# agent = agents.ExpectedSarsaAgent(env=env, exploration_rate=0.1, learning_rate=.1, discount_factor=0.9)
for i in tqdm(range(3000)):
    total_reward, last_state, agent = run_training_episode(agent, env)
show_trained_agent(agent, env)
# %%
res = []
res2 = []
for j in range(5):
    env.reset()
    agent = agents.QAgent(env=env, exploration_rate=0.01, learning_rate=0.1, discount_factor=0.1)
    for i in tqdm(range(10000)):
        total_reward, last_state, agent = run_training_episode(agent, env)
        res.append({
            "episode":i,
            "run": j,
            "reward":total_reward,
        })
    env.reset()
    for i in range(100):
        total_reward = run_real_episode(agent, env)
        res2.append({
            "episode":i,
            "run": j,
            "reward":total_reward,
        })

# %%
res_df = pd.DataFrame(res)
for run, gr_df in res_df.groupby('run'):
    plt.plot(gr_df["episode"], gaussian_filter1d(gr_df["reward"], 10))

# %%
res_df2 = pd.DataFrame(res2)
for run, gr_df in res_df2.groupby('run'):
    plt.plot(gr_df["episode"], gr_df["reward"])


# %% [markdown]
#  ## 3 Play with parameters and analyse results
#
# Finally, you will describe, evaluate and interpret your results from two RL agents, as well as compare your agents with the given Random agent. Feel free to use the provided helper functions for evaluating your environments.
#
# - Both quantified evaluation and human evaluation are needed in the report. The quantified evaluation shall focus on the measurement of reward. In human evaluation, you can use the visual tool provided by the environment package to interpret your results. Your report shall include at least one plot presenting compariable measures of the different agents.
#
# - While evaluating the results of Agent 2 (with SARSA algorithm), please try at least 2 different values of **epsilon** (expect 0) and discuss the influence of different epsilon values on results. In the end, please identify a reasonable epsilon value that could balance the exploration and exploiation, then fix this value for comparing two agents. Present your trails and results in the report.
#
# - In the report, you also need to parcitularly describe and discuss the similarity and difference of results from two RL agents (hint: on-policy VS off-policy). For this, please make sure that the compared results are obtained from the same environment (a same grid world for two different agents). Also, while evaluating the results of two agents, please try at least 2 different values of **gamma**. In this way, you could discuss the influence of this discounted factor in your report.
#
# - Please run the simulation for multiple times and average them for all your results.
#

# %%
# TODO: evaluation
epsilons = [0.01, 0.1, 0.5, 0.9]
alphas = [0.01, 0.1, 0.5, 0.9]
gammas = [0.1, 0.5, 0.9]
repeats = list(range(10))
num_episodes = 1000

all_params = list(it.product(epsilons, alphas, gammas, repeats))
all_results = []
for e, a, g, r in tqdm(all_params):
    s_agent = agents.SarsaAgent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    q_agent = agents.QAgent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    e_agent = agents.ExpectedSarsaAgent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    r_agent = agents.RandomAgent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    f_agent = agents.MostFrequentPolicyAgent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    for i in range(1, num_episodes + 1):
        reward_s_agent, last_state_s_agent, s_agent = run_training_episode(s_agent, env)
        reward_q_agent, last_state_q_agent, q_agent = run_training_episode(q_agent, env)
        reward_e_agent, last_state_e_agent, e_agent = run_training_episode(e_agent, env)
        reward_r_agent, last_state_r_agent, r_agent = run_training_episode(r_agent, env)
        reward_f_agent, last_state_f_agent, f_agent = run_training_episode(f_agent, env)

        all_results.append(
            {
                "agent": "SARSA",
                "epsilon": e,
                "alpha": a,
                "gamma": g,
                "episode": i,
                "repeat": r,
                "total_reward": reward_s_agent,
            }
        )
        all_results.append(
            {
                "agent": "QLearn",
                "epsilon": e,
                "alpha": a,
                "gamma": g,
                "episode": i,
                "repeat": r,
                "total_reward": reward_q_agent,
            }
        )
        all_results.append(
            {
                "agent": "ESARSA",
                "epsilon": e,
                "alpha": a,
                "gamma": g,
                "episode": i,
                "repeat": r,
                "total_reward": reward_e_agent,
            }
        )
        all_results.append(
            {
                "agent": "Random",
                "epsilon": e,
                "alpha": a,
                "gamma": g,
                "episode": i,
                "repeat": r,
                "total_reward": reward_r_agent,
            }
        )
        all_results.append(
            {
                "agent": "Frequent",
                "epsilon": e,
                "alpha": a,
                "gamma": g,
                "episode": i,
                "repeat": r,
                "total_reward": reward_f_agent,
            }
        )
# %%
df_all_results = pd.DataFrame(all_results)
df_mean_results = df_all_results.groupby(["agent", "epsilon", "alpha", "gamma", "episode"]).mean().drop("repeat", axis=1).reset_index()
df_mean_results_ = df_all_results.groupby(["agent", "epsilon", "alpha", "gamma", "episode"]).agg(["mean", "std"]).drop("repeat", axis=1).reset_index()
df_mean_results_.sort_values(("total_reward", "mean"), ascending=False)#.groupby(["agent", "epsilon", "alpha", "gamma"]).mean().reset_index().drop("episode", axis=1)

# %%
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(25, 15))

smooth_factor = 10
tmp = df_mean_results  # [df_mean_results.agent.isin(["SARSA", "QLearn", "ESARSA", "Random",])]
all_labels = set()
for (e, a, g), df in df_mean_results.groupby(["epsilon", "alpha", "gamma"]):
    all_labels.add((e, a, g))

all_labels_to_color = {key: val for val, key in enumerate(all_labels)}
cmap = cm.get_cmap("icefire")
norm = colors.Normalize(vmin=0, vmax=len(all_labels_to_color))
# norm = colors.LogNorm(vmin=0.01, vmax=len(all_labels_to_color))

for (e, a, g), df in df_mean_results.groupby(["epsilon", "alpha", "gamma"]):
    sarsa_selector = df.agent == "SARSA"
    q_selector = df.agent == "QLearn"
    e_selector = df.agent == "ESARSA"
    random_selector = df.agent == "Random"
    frequent_selector = df.agent == "Frequent"

    ax1.set_title("SARSA")
    label = f"e={e:.2f}|a={a:.2f}|g={g:.2f}"
    cl = cmap(norm(all_labels_to_color[(e, a, g)]))
    ax1.plot(
        df[sarsa_selector].episode,
        gaussian_filter1d(df[sarsa_selector].total_reward, smooth_factor),
        c=cl,
        label=label,
    )
    ax2.set_title("Q-Learning")
    ax2.plot(
        df[q_selector].episode,
        gaussian_filter1d(df[q_selector].total_reward, smooth_factor),
        c=cl,
        label=label,
    )
    ax3.set_title("Expected SARSA")
    ax3.plot(
        df[e_selector].episode,
        gaussian_filter1d(df[e_selector].total_reward, smooth_factor),
        c=cl,
        label=label,
    )
    ax4.set_title("Random")
    ax4.plot(
        df[random_selector].episode,
        gaussian_filter1d(df[random_selector].total_reward, smooth_factor),
        c=cl,
        label=label,
    )
    ax5.set_title("Frequent Policy")
    ax5.plot(
        df[frequent_selector].episode,
        gaussian_filter1d(df[frequent_selector].total_reward, smooth_factor),
        c=cl,
        label=label,
    )

    [tmp.set_xlabel("Episode") for tmp in (ax1, ax2, ax3, ax4)]
    [tmp.set_ylabel("Total Rewards") for tmp in (ax1, ax2, ax3)]


def annotate_plots(ax):
    for idx, line in enumerate(ax.lines):
        x = np.random.randint(len(line.get_xdata()))
        y = line.get_ydata()[x]
        ax.annotate(
            line.get_label(),
            xy=(x, y),
            xytext=(6, 0),
            backgroundcolor="w",
            textcoords="offset points",
            size=10,
            va="center",
            color=line.get_color(),
            rotation=15,
        )


annotate_plots(ax1)
annotate_plots(ax2)
annotate_plots(ax3)
annotate_plots(ax4)

boundaries = np.array([tmp.get_ylim() for tmp in (ax1, ax2, ax3)])
min_bound, max_bound = boundaries[:, 0].min(), boundaries[:, 1].max()
test = [tmp.set_ylim((min_bound, max_bound)) for tmp in (ax1, ax2, ax3)]

# [tmp.legend() for tmp in (ax1, ax2)]
# fig.legend()
fig.tight_layout()

# %%

best_configs = (
    (
        df_mean_results.groupby(
            [
                "agent",
                "epsilon",
                "alpha",
                "gamma",
            ]
        )
        .mean()
        .reset_index()
        .sort_values("total_reward", ascending=False)
        .groupby(
            [
                "agent",
            ]
        )
        .apply(lambda df: df.head(1))
    )
    .drop(["agent"], axis=1)
    .reset_index()
    .set_index("agent")
)

best_configs

# %%
all_conf = best_configs.to_dict("index")
env.reset()
s_conf = all_conf["SARSA"]
s_agent = agents.SarsaAgent(
    env=env,
    exploration_rate=s_conf["epsilon"],
    learning_rate=s_conf["alpha"],
    discount_factor=s_conf["gamma"],
)
q_conf = all_conf["QLearn"]
q_agent = agents.QAgent(
    env=env,
    exploration_rate=q_conf["epsilon"],
    learning_rate=q_conf["alpha"],
    discount_factor=q_conf["gamma"],
)
e_conf = all_conf["ESARSA"]
e_agent = agents.ExpectedSarsaAgent(
    env=env,
    exploration_rate=e_conf["epsilon"],
    learning_rate=e_conf["alpha"],
    discount_factor=e_conf["gamma"],
)
r_conf = all_conf["Random"]
r_agent = agents.RandomAgent(
    env=env,
    exploration_rate=r_conf["epsilon"],
    learning_rate=r_conf["alpha"],
    discount_factor=r_conf["gamma"],
)
f_conf = all_conf["Frequent"]
f_agent = agents.MostFrequentPolicyAgent(
    env=env,
    exploration_rate=f_conf["epsilon"],
    learning_rate=f_conf["alpha"],
    discount_factor=f_conf["gamma"],
)

# s_agent = SarsaAgent(env=env, exploration_rate=0.1, learning_rate=0.9, discount_factor=0.9)
# q_agent = QAgent(env=env, exploration_rate=0.1, learning_rate=0.9, discount_factor=0.9)
# e_agent = ExpectedSarsaAgent(env=env, exploration_rate=0.1, learning_rate=0.9, discount_factor=0.9)

rewards = []

for i in tqdm(range(20), desc="Repetition"):
    for j in tqdm(range(1000), desc="Train"):
        _, _, s_agent = run_training_episode(s_agent, env)
        _, _, q_agent = run_training_episode(q_agent, env)
        _, _, e_agent = run_training_episode(e_agent, env)

    for k in tqdm(range(1000), desc="Run"):
        initial_starting_point = env.reset()
        env.timer = 0
        env.current_position = initial_starting_point
        s_total_reward = run_real_episode(s_agent, env)
        env.timer = 0
        env.current_position = initial_starting_point
        q_total_reward = run_real_episode(q_agent, env)
        env.timer = 0
        env.current_position = initial_starting_point
        e_total_reward = run_real_episode(e_agent, env)
        env.timer = 0
        env.current_position = initial_starting_point
        r_total_reward = run_real_episode(r_agent, env)
        env.timer = 0
        env.current_position = initial_starting_point
        f_total_reward = run_real_episode(f_agent, env)
        rewards.append(
            {
                "repetition": i,
                "run": k,
                "QLearn": q_total_reward,
                "SARSA": s_total_reward,
                "ESARSA": e_total_reward,
                "Random": r_total_reward,
                "Frequent": f_total_reward,
                "initial_states": initial_starting_point,
            }
        )

pd.DataFrame(rewards)

# %%
across_training_rewards = pd.DataFrame(rewards)
smooth_factor = 20
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5))
ax1.plot(across_training_rewards["SARSA"], label="Sarsa")
ax1.plot(across_training_rewards["QLearn"], label="QLearn")
ax1.plot(across_training_rewards["ESARSA"], label="Expected Sarsa")
ax1.plot(across_training_rewards["Random"], label="Random")
ax1.plot(across_training_rewards["Frequent"], label="Most Frequent")
ax1.legend()
ax2.plot(gaussian_filter1d(across_training_rewards["SARSA"], smooth_factor), label="Sarsa")
ax2.plot(gaussian_filter1d(across_training_rewards["QLearn"], smooth_factor), label="QLearn")
ax2.plot(
    gaussian_filter1d(across_training_rewards["ESARSA"], smooth_factor),
    label="Expected Sarsa",
)
ax2.plot(
    gaussian_filter1d(across_training_rewards["Random"], smooth_factor),
    label="Random",
)
ax2.plot(
    gaussian_filter1d(across_training_rewards["Frequent"], smooth_factor),
    label="Most Frequent",
)
ax2.legend()
plt.show()

# %%
rewards_df = pd.DataFrame(rewards)
rewards_df.agg(["mean", "median", "max", "min"])

# %%
rewards_df["Initial incident"] = rewards_df["initial_states"].replace(env.idx2inc)
rewards_by_state = rewards_df.groupby("Initial incident").mean().reset_index()
rewards_by_state

# %%
melted_rewards = rewards_df.melt(
    id_vars="Initial incident", value_vars=["QLearn", "SARSA", "ESARSA", "Random", "Frequent"], var_name="Agent", value_name="Total Rewards"
)
melted_rewards

# %%
sns.boxplot(melted_rewards, x="Agent", y="Total Rewards")
plt.show()
# %%

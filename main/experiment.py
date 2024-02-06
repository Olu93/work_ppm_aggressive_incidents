import pandas as pd
from environment import TaskEnv
from agent import ExpectedSarsaAgent, QAgent, RandomAgent, SarsaAgent
import multiprocessing as mp
from tqdm import tqdm
import itertools as it


def run_training_episode(agent, env):
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


def run_experiment(params):
    (Agent, e, a, g, r, num_e) = params
    env = TaskEnv()
    all_results = []
    agent = Agent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    for i in range(1, num_e + 1):
        reward_s_agent, last_state_s_agent, s_agent = run_training_episode(agent, env)
        all_results.append({
            "agent": type(agent).__name__,
            "epsilon": e,
            "alpha": a,
            "gamma": g,
            "episode": i,
            "repeat": r,
            "total_reward": reward_s_agent
        })
    return all_results


if __name__ == "__main__":
    epsilons = [0.01, 0.1, 0.5, 0.9]
    alphas = [0.01, 0.1, 0.5, 0.9]
    gammas = [0.1, 0.5, 0.9]
    agents = [
        SarsaAgent,
        QAgent,
        ExpectedSarsaAgent,
        RandomAgent,
    ]
    repeats = list(range(50))
    episodes = [1000]

    all_params = list(it.product(agents, epsilons, alphas, gammas, repeats, episodes))
    all_results = []
    with mp.Pool(10) as p:
        for results in p.imap_unordered(run_experiment, tqdm(all_params, total=len(all_params)), chunksize=2):
            all_results.extend(results)

    df_all_results = pd.DataFrame(all_results)
    print(df_all_results)
    df_all_results.to_csv("exp.csv")

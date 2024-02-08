import pandas as pd
from environment import TaskEnv
from agent import ExpectedSarsaAgent, MostFrequentPolicyAgent, QAgent, RandomAgent, SarsaAgent, TDAgent
import multiprocessing as mp
from tqdm import tqdm
import itertools as it


def run_real_episode(agent: TDAgent, env: TaskEnv):
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


def run_experiment(params):
    (Agent, e, a, g, r, num_e, min_inc) = params
    env = TaskEnv(frequencies_file=min_inc)
    all_results = []
    agent = Agent(env=env,
                  exploration_rate=e,
                  learning_rate=a,
                  discount_factor=g)
    for i in range(1, num_e + 1):
        reward_s_agent, last_state_s_agent, s_agent = run_real_episodes(
            agent, env)
        all_results.append({
            "agent": type(agent).__name__,
            "epsilon": e,
            "alpha": a,
            "gamma": g,
            "episode": i,
            "min_inc":min_inc,
            "repeat": r,
            "total_reward": reward_s_agent
        })
    return all_results


if __name__ == "__main__":
    epsilons = [0.01, 0.1, 0.5, 0.9]
    alphas = [0.01, 0.1, 0.5, 0.9]
    gammas = [0.1, 0.5, 0.9]
    agents = [
        SarsaAgent, QAgent, ExpectedSarsaAgent, RandomAgent,
        MostFrequentPolicyAgent
    ]
    repeats = list(range(50))
    min_amount_incidents = [
        "data/frequencies_final_1.csv",
        "data/frequencies_final_3.csv",
        "data/frequencies_final_5.csv",
        "data/frequencies_final_7.csv",
    ]
    episodes = [3000]

    all_params = list(
        it.product(agents, epsilons, alphas, gammas, repeats, episodes))
    all_results = []
    with mp.Pool(5) as p:
        for results in p.imap_unordered(run_experiment,
                                        tqdm(all_params,
                                             total=len(all_params)),
                                        chunksize=2):
            all_results.extend(results)

    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv("experiment_parameter_search.csv")

    df_mean_results = (df_all_results.groupby([
        "agent",
        "epsilon",
        "alpha",
        "gamma",
        "episode",
    ]).mean().drop("repeat", axis=1).reset_index())
    df_mean_results.sort_values("total_reward", ascending=False).groupby([
        "agent",
        "epsilon",
        "alpha",
        "gamma",
    ]).mean().reset_index().drop("episode", axis=1)

    best_configs = (df_mean_results.sort_values(
        "total_reward", ascending=False).reset_index().groupby([
            "agent"
        ]).apply(lambda df: df.head(1)).set_index("agent").groupby([
            "agent", "epsilon", "alpha", "gamma"
        ]).mean().drop("episode", axis=1).reset_index().set_index("agent"))

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

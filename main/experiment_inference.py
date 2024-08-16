import pandas as pd
from environment import TaskEnv
from agent import ExpectedSarsaAgent, MostFrequentPolicyAgent, PolicyIterationAgent, QAgent, RandomAgent, SarsaAgent, TDAgent
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


def run_real_episode(agent: TDAgent, env: TaskEnv):
    next_action = None
    done = False
    current_state = env.current_position
    total_reward = 0
    steps = []
    while not done:
        next_action = agent.select_action(current_state, True)
        next_state, reward, done, info = env.step(next_action)
        total_reward += reward
        current_state = next_state
        steps.append(info["step_sequence"])
    return total_reward, steps


def run_experiment(params):
    (Agent, e, a, g, r, num_e, min_inc) = params
    env = TaskEnv(frequencies_file=min_inc)
    all_results = []
    agent = Agent(env=env, exploration_rate=e, learning_rate=a, discount_factor=g)
    for i in range(1, num_e + 1):
        reward_s_agent, last_state_s_agent, s_agent = run_real_episode(agent, env)
        all_results.append(
            {"agent": type(agent).__name__, "epsilon": e, "alpha": a, "gamma": g, "episode": i, "min_inc": min_inc, "repeat": r, "total_reward": reward_s_agent}
        )
    return all_results


def map_reward_func(type_of_reward):
    severity = {"va": 0.0, "po": -1.0, "sib": -3.0, "pp": -4.0, "Tau": 1.0}

    action_rewards = {
        "contact beeindigd/weggegaan": -1.0,
        "client toegesproken/gesprek met client": 0,
        "geen": 0,
        "client afgeleid": -1.0,
        "naar andere kamer/ruimte gestuurd": -1.0,
        "met kracht tegen- of vastgehouden": -2.0,
        "afzondering (deur op slot)": -2.0,
    }

    if type_of_reward == "reward_all_actions_the_same":
        action_rewards = {
            "contact beeindigd/weggegaan": -1.0,
            "client toegesproken/gesprek met client": -1.0,
            "geen": -1.0,
            "client afgeleid": -1.0,
            "naar andere kamer/ruimte gestuurd": -1.0,
            "met kracht tegen- of vastgehouden": -1.0,
            "afzondering (deur op slot)": -1.0,
        }

    if type_of_reward == "reward_zero_tau":
        severity = {"va": -1.0, "po": -2.0, "sib": -4.0, "pp": -5.0, "Tau": 0.0}

    if type_of_reward == "reward_zero_tau_all_actions_the_same":
        severity = {"va": -1.0, "po": -2.0, "sib": -4.0, "pp": -5.0, "Tau": 0.0}
        action_rewards = {
            "contact beeindigd/weggegaan": -1.0,
            "client toegesproken/gesprek met client": -1.0,
            "geen": -1.0,
            "client afgeleid": -1.0,
            "naar andere kamer/ruimte gestuurd": -1.0,
            "met kracht tegen- of vastgehouden": -1.0,
            "afzondering (deur op slot)": -1.0,
        }
    return severity, action_rewards


if __name__ == "__main__":
    min_inc = "data/frequencies_final_1.csv"
    min_amount_incidents = [
        # "../data/frequencies_final_1.csv",
        "../data/frequencies_final_3.csv",
        # "data/frequencies_final_5.csv",
        # "data/frequencies_final_7.csv",
    ]
    reward_fn = [
        # "reward_bart",
        "reward_all_actions_the_same",
        # "reward_zero_tau",
        # "reward_zero_tau_all_actions_the_same",
    ]
    episodes = 100
    # episodes = 10
    episodesT = 100
    # episodesT = 100
    repeats = list(range(1000))
    # repeats = list(range(3))
    all_results = []
    
    pbar = tqdm(total=(len(min_amount_incidents) * len(reward_fn) * len(repeats)))
    
    for min_inc in min_amount_incidents:
        for rew_type in reward_fn:
            severity, action_reward = map_reward_func(rew_type)
            env = TaskEnv(time_out=6, frequencies_file=min_inc)
            env.severity = severity
            env.action_reward = action_reward

            s_agent = SarsaAgent(env=env, exploration_rate=0.1, learning_rate=0.2, discount_factor=0.2)
            q_agent = QAgent(env=env, exploration_rate=0.1, learning_rate=0.2, discount_factor=0.2)
            e_agent = ExpectedSarsaAgent(env=env, exploration_rate=0.1, learning_rate=0.2, discount_factor=0.2)
            r_agent = RandomAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
            f_agent = MostFrequentPolicyAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
            p_agent = PolicyIterationAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9)


            for i in repeats:
                for j in range(episodes):
                    _, _, s_agent = run_training_episode(s_agent, env)
                    _, _, q_agent = run_training_episode(q_agent, env)
                    _, _, e_agent = run_training_episode(e_agent, env)
                    r_agent = RandomAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
                    f_agent = MostFrequentPolicyAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)

                agents = [
                    p_agent, 
                    # s_agent, 
                    # q_agent, 
                    # e_agent, 
                    r_agent, 
                    f_agent,
                ]
                for k in range(episodesT):
                    initial_starting_point = env.reset()
                    for agent in agents:
                        env.reset()
                        env.timer = 0
                        env.current_position = initial_starting_point
                        total_reward, steps = run_real_episode(agent, env)
                        all_results.append(
                            {
                                "repetition": i,
                                "episode": k,
                                "agent": agent.__class__.__name__,
                                "min_inc": min_inc,
                                "rew_type": rew_type,
                                "total_reward": total_reward,
                                "steps": steps,
                                "time": len(steps),
                            }
                        )
                pbar.update(1)

    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv("../data/experiment_inference_policy_iteration.csv")

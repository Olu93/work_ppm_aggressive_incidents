import pandas as pd
from environment import TaskEnv
from agent import ExpectedSarsaAgent, MostFrequentPolicyAgent, QAgent, RandomAgent, SarsaAgent, TDAgent
import multiprocessing as mp
from tqdm import tqdm
import itertools as it

from environment_probablistic_reward import TaskEnv2StepProbablisticTimePenalty, TaskEnvProbablisticTimePenalty


def run_training_episode(agent, env):
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
    min_amount_incidents = [
        # "../data/frequencies_final_1.csv",
        "../data/frequencies_final_3.csv",
        # "../data/frequencies_final_5.csv",
        # "../data/frequencies_final_7.csv",
    ]
    reward_fn = [
        "reward_bart",
        "reward_all_actions_the_same",
        # "reward_zero_tau",
        # "reward_zero_tau_all_actions_the_same",
    ]
    episodes = 100
    # episodes = 10
    episodesT = 1000
    # episodesT = 100
    repeats = list(range(100))
    # repeats = list(range(3))
    all_results = []
    
    pbar = tqdm(total=(len(min_amount_incidents) * len(reward_fn) * len(repeats)))
    
    for min_inc in min_amount_incidents:
        for rew_type in reward_fn:
            severity, action_reward = map_reward_func(rew_type)
            env = TaskEnv2StepProbablisticTimePenalty(time_out=365, frequencies_file=min_inc, time_probabilities_file="../data/prob_time_given_incident_action.json", classification_pipeline="../data/logistic_regression_pipeline.pkl" )
            env.severity = severity
            env.action_reward = action_reward

            e_agent = ExpectedSarsaAgent(env=env, exploration_rate=0.5, learning_rate=0.9, discount_factor=0.5)
            q_agent = QAgent(env=env, exploration_rate=0.1, learning_rate=0.01, discount_factor=0.9)
            s_agent = SarsaAgent(env=env, exploration_rate=0.9, learning_rate=0.01, discount_factor=0.1)
            r_agent = RandomAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
            f_agent = MostFrequentPolicyAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)


            for i in repeats:
                for j in range(episodes):
                    _, _, s_agent = run_training_episode(s_agent, env)
                    _, _, q_agent = run_training_episode(q_agent, env)
                    _, _, e_agent = run_training_episode(e_agent, env)
                    r_agent = RandomAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
                    f_agent = MostFrequentPolicyAgent(env=env, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.1)
                agents = [s_agent, q_agent, e_agent, r_agent, f_agent]
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
    df_all_results.to_csv("../data/experiment_probablistic_time_reward_2_inference.csv")

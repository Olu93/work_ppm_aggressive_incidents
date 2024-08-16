import pandas as pd
from environment import TaskEnv
from agent import ExpectedSarsaAgent, MostFrequentPolicyAgent, QAgent, RandomAgent, SarsaAgent
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
        next_action = agent.select_action(
            current_state) if not next_action else next_action
        next_state, reward, done, _ = env.step(next_action)
        total_reward += reward / env.timer
        # print(reward)
        next_action = agent.learn(current_state, next_action, next_state,
                                  reward, done)
        current_state = next_state

    last_state = current_state
    return total_reward, last_state, agent


def run_experiment(params):
    (Agent, e, a, g, r, num_e, min_inc, rew_type) = params
    
    env = TaskEnv2StepProbablisticTimePenalty(time_out=365, frequencies_file=min_inc, time_probabilities_file="../data/prob_time_given_incident_action.json", classification_pipeline="../data/logistic_regression_pipeline.pkl" )
    severity, action_reward = map_reward_func(rew_type)
    env.severity=severity
    env.action_reward=action_reward
    
    
    all_results = []
    agent = Agent(env=env,
                  exploration_rate=e,
                  learning_rate=a,
                  discount_factor=g)
    for i in range(1, num_e + 1):
        reward_s_agent, last_state_s_agent, s_agent = run_training_episode(
            agent, env)
        all_results.append({
            "agent": type(agent).__name__,
            "epsilon": e,
            "alpha": a,
            "gamma": g,
            "episode": i,
            "min_inc": min_inc,
            "rew_type": rew_type,
            "repeat": r,
            "total_reward": reward_s_agent
        })
    return all_results


def map_reward_func(type_of_reward):
    severity = {'va': 0.0, 'po': -1.0, 'sib': -3.0, 'pp': -4.0, 'Tau': 1.0}

    action_rewards = {
        'contact beeindigd/weggegaan': -1.0,
        'client toegesproken/gesprek met client': 0,
        'geen': 0,
        'client afgeleid': -1.0,
        'naar andere kamer/ruimte gestuurd': -1.0,
        'met kracht tegen- of vastgehouden': -2.0,
        'afzondering (deur op slot)': -2.0,
    }

    if type_of_reward == "reward_all_actions_the_same":
        action_rewards = {
            'contact beeindigd/weggegaan': -1.0,
            'client toegesproken/gesprek met client': -1.0,
            'geen': -1.0,
            'client afgeleid': -1.0,
            'naar andere kamer/ruimte gestuurd': -1.0,
            'met kracht tegen- of vastgehouden': -1.0,
            'afzondering (deur op slot)': -1.0,
        }

    if type_of_reward == "reward_zero_tau":
        severity = {
            'va': -1.0,
            'po': -2.0,
            'sib': -4.0,
            'pp': -5.0,
            'Tau': 0.0
        }

    if type_of_reward == "reward_zero_tau_all_actions_the_same":
        severity = {
            'va': -1.0,
            'po': -2.0,
            'sib': -4.0,
            'pp': -5.0,
            'Tau': 0.0
        }
        action_rewards = {
            'contact beeindigd/weggegaan': -1.0,
            'client toegesproken/gesprek met client': -1.0,
            'geen': -1.0,
            'client afgeleid': -1.0,
            'naar andere kamer/ruimte gestuurd': -1.0,
            'met kracht tegen- of vastgehouden': -1.0,
            'afzondering (deur op slot)': -1.0,
        }
    return severity, action_rewards


if __name__ == "__main__":
    epsilons = [0.01, 0.1, 0.5, 0.9]
    alphas = [0.01, 0.1, 0.5, 0.9]
    gammas = [0.01, 0.1, 0.5, 0.9]
    agents = [
        SarsaAgent,
        QAgent,
        ExpectedSarsaAgent,
        # RandomAgent,
        # MostFrequentPolicyAgent,
    ]
    repeats = list(range(10))
    min_amount_incidents = [
        # "../data/frequencies_final_1.csv",
        "../data/frequencies_final_3.csv",
        # "../data/frequencies_final_5.csv",
        # "../data/frequencies_final_7.csv",
    ]
    reward_fn = [
        # "reward_bart",
        "reward_all_actions_the_same",
        # "reward_zero_tau",
        # "reward_zero_tau_all_actions_the_same",
    ]
    episodes = [1000]

    all_params = list(
        it.product(agents, epsilons, alphas, gammas, repeats, episodes,
                   min_amount_incidents, reward_fn))
    all_results = []

    r = run_experiment(all_params[0])
    with mp.Pool(4) as p:
        for results in p.imap_unordered(run_experiment,
                                        tqdm(all_params,
                                             total=len(all_params)),
                                        chunksize=3):
            all_results.extend(results)

    df_all_results = pd.DataFrame(all_results)
    # df_all_results.to_csv("experiment_parameter_search.csv")

    df_mean_results = (df_all_results.groupby([
        "agent",
        "epsilon",
        "alpha",
        "gamma",
        "episode",
        "min_inc",
        "rew_type",
    ]).mean().drop("repeat", axis=1).reset_index())
    df_mean_results.to_csv("../data/experiment_probablistic_time_reward_2_parameters.csv")


    # TODO:
    # - Receiving the actual most frequent path
    # - Average rewards
    # - Most frequent paths by policy
    # - Most frequent action/response by incident and algorithm

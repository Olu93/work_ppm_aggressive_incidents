from collections import defaultdict
import io
import json
import pandas as pd
import environment_probablistic_reward as envs
from agent import ExpectedSarsaAgent, MostFrequentPolicyAgent, PolicyIterationAgent, QAgent, RandomAgent, SarsaAgent, TDAgent
import multiprocessing as mp
from tqdm import tqdm
import itertools as it
import numpy as np
import csv
from tqdm.contrib.concurrent import process_map  # or thread_map
from functools import partial
import cloudpickle as pkl

def run_training_episode(agent, env):
    next_action = None
    done = False
    current_state = env.reset()
    total_reward = 0
    while not done:
        next_action = agent.select_action(current_state) if not next_action else next_action
        next_state, reward, done, _ = env.step(next_action)
        # print(reward)
        next_action = agent.learn(current_state, next_action, next_state, reward, done)
        current_state = next_state

    last_reward = reward
    last_state = current_state
    return last_reward, last_state, agent


def run_real_episode(agent: TDAgent, env: envs.TaskEnv2StepProbablisticTimePenalty, reset=False):
    next_action = None
    done = False
    current_state = env.current_position if not reset else env.reset()
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
    env = envs.TaskEnv2StepProbablisticTimePenalty(frequencies_file=min_inc)
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


def run_stage_3(Env, time_out, all_agents, min_inc, rew_type, best_agent_collection, repetition, episodes):
    severity, action_reward = map_reward_func(rew_type)
    env = Env(time_out=time_out, frequencies_file=min_inc, time_probabilities_file="../data/prob_time_given_incident_action.json", classification_pipeline="../data/logistic_regression_pipeline.pkl")
    env.severity = severity
    env.action_reward = action_reward    
    initial_starting_point = env.reset()
    partial_results = []
    for k in range(episodes):
        for Agent in all_agents:
            agent = best_agent_collection[min_inc][rew_type][Agent]['agent']
            total_reward, steps = run_real_episode(agent, env)
            partial_results.append({
                        "repetition": repetition,
                        "episode": k,
                        "agent": agent.__class__.__name__,
                        "min_inc": min_inc,
                        "rew_type": rew_type,
                        "total_reward": total_reward,
                        "steps": steps,
                        "num_steps": len(steps),
                        "time": env.timer,
                    }) 
            env.reset()
            env.current_position = initial_starting_point
    return partial_results

def wrapper_for_experiment_stage3(params):
    Env, time_out, all_agents, best_agent_collection,  min_inc, rew_type, repetition, episodes_stage_3 = params
    return run_stage_3(Env, time_out, all_agents, min_inc, rew_type, best_agent_collection, repetition, episodes_stage_3)


def wrapper_for_experiment_stage1(params):
    TIME_OUT, episodes_stage_1, episodes_stage_2, repeats_stage_2, Env, all_agent_params, min_inc, rew_type, agent_number, Agent = params
    severity, action_reward = map_reward_func(rew_type)    
    env = Env(time_out=TIME_OUT, frequencies_file=min_inc, time_probabilities_file="../data/prob_time_given_incident_action.json", classification_pipeline="../data/logistic_regression_pipeline.pkl")
    env.severity = severity
    env.action_reward = action_reward    
    agent = Agent(env=env, **all_agent_params[Agent])
    for e in range(episodes_stage_1):
        last_reward, _, trained_agent = run_training_episode(agent, env)
        
    agent_performance_results = []
    for r in repeats_stage_2:
        agent_performance_results_per_run = []
        for e in range(episodes_stage_2):
            env.reset()
            last_reward, trained_agent = run_real_episode(agent, env, True)
            agent_performance_results_per_run.append(last_reward)
        agent_performance_results.append(np.mean(agent_performance_results_per_run))
    return min_inc, rew_type, agent_number, Agent, {"agent": agent, "last_reward":np.mean(agent_performance_results)}
    # training_results[min_inc][rew_type][Agent][agent_number]["agent"] = agent
    # training_results[min_inc][rew_type][Agent][agent_number]["last_reward"] = np.mean(agent_performance_results_per_run)

if __name__ == "__main__":

    TIME_OUT = 365

    min_amount_incidents = [
        # "../data/frequencies_final_1.csv",
        "../data/frequencies_final_3.csv",
        # "data/frequencies_final_5.csv",
        # "data/frequencies_final_7.csv",
    ]
    reward_fn = [
        "reward_bart",
        # "reward_all_actions_the_same",
        # "reward_zero_tau",
        # "reward_zero_tau_all_actions_the_same",
    ]
    episodes_stage_1 = 1000
    episodes_stage_2 = 100
    episodes_stage_3 = 100
    repeats_stage_1 = list(range(5))
    repeats_stage_2 = list(range(100))
    repeats_stage_3 = list(range(1000))
    all_results = []

    Env = envs.TaskEnv2StepProbablisticTimePenalty

    all_agent_params = {
        SarsaAgent: {
            "exploration_rate":0.1,
            "learning_rate":0.9,
            "discount_factor":0.9,
        },
        QAgent: {
            "exploration_rate":0.5,
            "learning_rate":0.1,
            "discount_factor":0.1,
        },
        ExpectedSarsaAgent: {
            "exploration_rate":0.01,
            "learning_rate":0.9,
            "discount_factor":0.5,
        },
        RandomAgent: {
            "exploration_rate":0,
            "learning_rate":0,
            "discount_factor":0,
        },
        MostFrequentPolicyAgent: {
            "exploration_rate":0,
            "learning_rate":0,
            "discount_factor":0,
        },
        # PolicyIterationAgent: {
        #     "exploration_rate":0.0,
        #     "learning_rate":0.0,
        #     "discount_factor":0.9,
        # },
    }
    all_agents = list(all_agent_params.keys())

    # NOTE: Stage 1 trains the agents
    all_params_stage1 = list(it.product(min_amount_incidents, reward_fn, repeats_stage_1, all_agents))
    # NOTE: Stage 2 runs the models and picks the best of them
    all_params_stage2 = list(it.product(min_amount_incidents, reward_fn))
    # NOTE: Stage 3 Runs the picks against eachother
    all_params_stage3 = list(it.product(min_amount_incidents, reward_fn, repeats_stage_3))
    # Stage 1: Training
    training_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    all_parallel_params_stage1 = [(TIME_OUT, episodes_stage_1, episodes_stage_2, repeats_stage_2, Env, all_agent_params, *pms) for pms in all_params_stage1]

    with mp.Pool(8) as p: 
        for min_inc, rew_type, agent_number, Agent, results in p.imap_unordered(wrapper_for_experiment_stage1, tqdm(all_parallel_params_stage1, total=len(all_parallel_params_stage1))):
            training_results[min_inc][rew_type][Agent][agent_number] = results

    # for min_inc, rew_type, agent_number, Agent in tqdm(all_params_stage1):
    #     res = wrapper_for_experiment_stage1(TIME_OUT, episodes_stage_1, episodes_stage_2, repeats_stage_2, Env, all_agent_params, training_results, min_inc, rew_type, agent_number, Agent)
        

    # Stage 2: Selection
    best_agent_collection = defaultdict(partial(defaultdict, partial(defaultdict, dict)))
    for Agent in all_agents:    
        for min_inc, rew_type in all_params_stage2:
            for agent, stored_agent_info in training_results[min_inc][rew_type][Agent].items():
                best_agent = None
                best_value = -np.inf
                if stored_agent_info["last_reward"] > best_value:
                    best_value = stored_agent_info["last_reward"]
                    best_agent = stored_agent_info["agent"]
            best_agent_collection[min_inc][rew_type][Agent]["agent"] = best_agent
            best_agent_collection[min_inc][rew_type][Agent]["last_reward"] = best_value
            print("Done")
    # best_agent_collection = json.loads(json.dumps(best_agent_collection))
    # Stage 3: Validation

    pkl.dump(best_agent_collection, io.open('../data/best_agents_probablistic_reward_bart_reward.pkl', 'wb'))
    all_results = []


    all_parallel_params = [(Env, TIME_OUT, all_agents, best_agent_collection, *pms, episodes_stage_3) for pms in all_params_stage3]
    
    with mp.Pool(8) as p: 
        for results in p.imap_unordered(wrapper_for_experiment_stage3, tqdm(all_parallel_params, total=len(all_parallel_params))):
            all_results.extend(results)
    # for min_inc, rew_type, repetition in tqdm(all_params_stage3):
    #     # NOTE: env is initialized here so that every repeat has a different starting point but not every episode
    #     for res in run_stage_3(Env, TIME_OUT, all_agents, best_agent_collection, min_inc, rew_type, repetition, episodes_stage_3):
    #         all_results.append(res)

    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_pickle("../data/experiment_probablistic_time_reward_3_inference_bart_reward.pkl")

    
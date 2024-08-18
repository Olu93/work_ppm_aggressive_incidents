# %%
import pandas as pd
import ast
from collections import Counter, defaultdict
import io
from tqdm import tqdm
from scipy import stats
import numpy as np

# %%
experiment_1 = pd.read_pickle('../data/experiment_inference_experiment_1.pkl')
experiment_1

# %%
experiment_2_tmp = pd.read_pickle('../data/experiment_inference_policy_iteration.pkl')
experiment_2 = experiment_2_tmp[(experiment_2_tmp.min_inc == "../data/frequencies_final_3.csv") & (experiment_2_tmp.rew_type == "reward_bart")]
experiment_2
# %%
experiment_3_tmp = pd.read_pickle('../data/experiment_inference_correction.pkl')
experiment_3 = experiment_3_tmp[(experiment_3_tmp.min_inc == "../data/frequencies_final_3.csv") & (experiment_3_tmp.rew_type == "reward_all_actions_the_same")]
experiment_3
# %%
experiment_4_tmp = pd.read_pickle('../data/experiment_probablistic_time_reward_2_inference_correction.pkl')
experiment_4 = experiment_4_tmp[(experiment_4_tmp.min_inc == "../data/frequencies_final_3.csv") & (experiment_4_tmp.rew_type == "reward_all_actions_the_same")]
experiment_4
# %%
experiment_1["experiment"] = "Exp1"
experiment_1["num_steps"] = experiment_1["time"]
experiment_1["time"] = None
experiment_1["description"] = "Basic Experiment"
experiment_1.loc[experiment_1.agent=="MostFrequentPolicyAgentMod","agent"] = "MostFrequentPolicyAgent"

experiment_2["experiment"] = "Exp2"
experiment_2["num_steps"] = experiment_2["time"]
experiment_2["time"] = None
experiment_2["description"] = "Experiment on history lengths 3"

experiment_3["experiment"] = "Exp3"
experiment_3["num_steps"] = experiment_3["time"]
experiment_3["time"] = None
experiment_3["description"] = "all actions same penalty"

experiment_4["experiment"] = "Exp4"
experiment_4["description"] = "time aware reward function"
# %%
all_experiments = pd.concat([experiment_1, experiment_2, experiment_3, experiment_4])
all_experiments
# %%
results_agg = all_experiments.drop(["steps", "description"], axis=1).groupby([
            "experiment",
            "min_inc",
            "rew_type",
            "agent",
            "repetition"
        ]).mean().reset_index().groupby([
            "experiment",
            "min_inc",
            "rew_type",
            "agent",
        ]).agg(["mean", "std", "count"])

results_agg
# %%
all_experiments.to_csv('../saved_results/all_experiment_data.tar.gz', compression='tar')

# %%
all_experiments.to_pickle('../saved_results/all_experiment_data.pkl.gz', compression='gzip')
# %%

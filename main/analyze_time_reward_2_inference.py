# %%
import pandas as pd
import ast
from collections import Counter, defaultdict
import io
from tqdm import tqdm
from scipy import stats
import numpy as np
# %%
df_mean_results = pd.read_pickle('../data/experiment_probablistic_time_reward_2_inference_correction.pkl')
df_mean_results
# %%
results_agg = df_mean_results.drop(["steps"], axis=1).groupby([
            "min_inc",
            "rew_type",
            "agent",
            "episode"
        ]).mean().reset_index().groupby([
            "min_inc",
            "rew_type",
            "agent",
        ]).agg(["mean", "std", "count"])

results_agg

# %%
margin_of_err = (stats.norm.ppf((1 + 0.95) / 2)) * (results_agg[("total_reward", "std")]/np.sqrt(results_agg[("total_reward", "count")]))
results_agg[("total_reward", "confidence_intervall_left")] = results_agg[("total_reward", "mean")]-margin_of_err
results_agg[("total_reward", "confidence_intervall_right")] = results_agg[("total_reward", "mean")]+margin_of_err
results_agg
# %%
# results_agg.loc[("data/frequencies_final_1.csv", "reward_all_actions_the_same")]
# %%
results_agg.loc[("../data/frequencies_final_3.csv", "reward_all_actions_the_same")]
# # %%
# results_agg.loc[("data/frequencies_final_5.csv", "reward_all_actions_the_same")]
# # %%
# results_agg.loc[("data/frequencies_final_7.csv", "reward_all_actions_the_same")]
 # %%
# %%
agent = "RandomAgent"
def get_counts(df_mean_results, agent, history, rew_type):
    q_results = df_mean_results[(df_mean_results["agent"]==agent) & (df_mean_results["min_inc"] == history) & (df_mean_results["rew_type"] == rew_type)]
    transitions = q_results["steps"].explode().tolist()
    reactions = [t[:2] for t in transitions]
    cnt_t = Counter(transitions)
    cnt_r = Counter(reactions)
    cnt_x = Counter([tuple(l) for l in q_results["steps"].tolist()])
    return cnt_t, cnt_r, cnt_x


# %%
experiments = [
    #  "../data/frequencies_final_1.csv",
     "../data/frequencies_final_3.csv"
]

rew_types = [
"reward_all_actions_the_same",
# "reward_bart"
]
for agent in ["QAgent", "SarsaAgent"]:
    for history in experiments:
        for rew_type in rew_types:
            h = history.split("/")[-1].split(".")[0]
            cnt_t, cnt_r, cnt_x = get_counts(df_mean_results, agent, history, rew_type)
            with io.open(f'../data/prob_res_analysis_transitions_{agent}_{rew_type}_{h}.txt', 'w') as f:
                for transition, cnt in tqdm(cnt_t.most_common()):
                    f.write(f"Count {cnt} | {transition}\n")
            with io.open(f'../data/prob_res_analysis_epochs_{agent}_{rew_type}_{h}.txt', 'w') as f:
                for epoch_run, cnt in tqdm(cnt_x.most_common()):
                    f.write(f"Count {cnt} | {' -> '.join([str(x) for x in epoch_run])}\n")
            with io.open(f'../data/prob_res_analysis_reactions_{agent}_{rew_type}_{h}.txt', 'w') as f:
                for reaction, cnt in tqdm(cnt_r.most_common()):
                    f.write(f"Count {cnt} | {reaction}\n")
print("done")

# %%
cnt_x.most_common()[:15]
# %%
cnt_r.most_common()
# %%
cnt_t, cnt_r, cnt_x = get_counts(df_mean_results, "SarsaAgent")
cnt_t

# %%
import json
import io
json.dump(cnt_x.most_common(), io.open('../data/sarsa_example_episodes.json', "w"))

# %%
cnt_r.most_common()
# %%
# Qlearn
[((('sib', 'afzondering (deur op slot)', 'Tau'),), 541),
 ((('va', 'geen', 'Tau'),), 379),
 ((('po', 'naar andere kamer/ruimte gestuurd', 'Tau'),), 301),
 ((('pp', 'naar andere kamer/ruimte gestuurd', 'Tau'),), 256),
 ((('va', 'met kracht tegen- of vastgehouden', 'Tau'),), 223)]
# SARSA
[((('sib', 'afzondering (deur op slot)', 'Tau'),), 859),
 ((('va', 'naar andere kamer/ruimte gestuurd', 'Tau'),), 506),
 ((('pp', 'naar andere kamer/ruimte gestuurd', 'Tau'),), 346),
 ((('va', 'client toegesproken/gesprek met client', 'Tau'),), 333),
 ((('po', 'client afgeleid', 'Tau'),), 262)]
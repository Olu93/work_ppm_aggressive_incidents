# %%
import pandas as pd
import ast
from collections import Counter, defaultdict
# %%
df_mean_results = pd.read_csv('../data/experiment_inference_policy_iteration_3.csv')
df_mean_results
# %%
results_agg = (df_mean_results.drop(["steps"], axis=1).groupby([
            "min_inc",
            "rew_type",
            "agent",
        ]).mean())
results_agg

    # best_configs.to_csv("experiment_best_params.csv")
# %%
# results_agg.loc[("../data/frequencies_final_1.csv", "reward_all_actions_the_same")]
# %%
# results_agg.loc[("../data/frequencies_final_3.csv", "reward_all_actions_the_same")]
# # %%
# results_agg.loc[("data/frequencies_final_5.csv", "reward_all_actions_the_same")]
# # %%
results_agg.loc[("../data/frequencies_final_3.csv", "reward_bart")]
 # %%
df_mean_results["steps"] = df_mean_results["steps"].map(ast.literal_eval)
df_mean_results
# %%
agent = "QAgent"
def get_counts(df_mean_results, agent):
    q_results = df_mean_results[(df_mean_results["agent"]==agent) & (df_mean_results["min_inc"] == "data/frequencies_final_3.csv") & (df_mean_results["rew_type"] == "reward_all_actions_the_same")]
    transitions = q_results["steps"].explode().tolist()
    reactions = [t[:2] for t in transitions]
    cnt_t = Counter(transitions)
    cnt_r = Counter(reactions)
    cnt_x = Counter([tuple(l) for l in q_results["steps"].tolist()])
    return cnt_t, cnt_r, cnt_x

cnt_t, cnt_r, cnt_x = get_counts(df_mean_results, agent)
cnt_t

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
json.dump(cnt_x.most_common(), io.open('data/sarsa_example_episodes.json', "w"))

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
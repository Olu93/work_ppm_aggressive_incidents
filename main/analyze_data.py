# %%
import pandas as pd
import ast
from collections import Counter, defaultdict
import io
from tqdm import tqdm
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %%
all_experiments = pd.read_pickle('../saved_results/all_experiment_data.pkl.gz')
all_experiments

# %%
df_agg_f_test = all_experiments[all_experiments!="ExpectedSarsaAgent"].drop(["steps", "description"], axis=1).groupby([
            "experiment",
            "min_inc",
            "rew_type",
            "agent",
            "repetition"
        ]).mean().reset_index()
df_agg_f_test
# %%
exp = "Exp1"
agent_1 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="MostFrequentPolicyAgent")]["total_reward"]
agent_2 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="QAgent")]["total_reward"]
agent_3 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="SarsaAgent")]["total_reward"]
agent_4 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="PolicyIterationAgent")]["total_reward"]
print(stats.f_oneway(agent_1,agent_2,agent_3,agent_4))
print(pairwise_tukeyhsd(endog=df_agg_f_test[(df_agg_f_test.experiment==exp)]["total_reward"], groups=df_agg_f_test[(df_agg_f_test.experiment==exp)]["agent"], alpha=0.05))
# %%
exp = "Exp2"
agent_1 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="MostFrequentPolicyAgent")]["total_reward"]
agent_2 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="QAgent")]["total_reward"]
agent_3 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="SarsaAgent")]["total_reward"]
agent_4 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="PolicyIterationAgent")]["total_reward"]
print(stats.f_oneway(agent_1,agent_2,agent_3,agent_4))
print(pairwise_tukeyhsd(endog=df_agg_f_test[(df_agg_f_test.experiment==exp)]["total_reward"], groups=df_agg_f_test[(df_agg_f_test.experiment==exp)]["agent"], alpha=0.05))
# %%
exp = "Exp3"
agent_1 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="MostFrequentPolicyAgent")]["total_reward"]
agent_2 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="QAgent")]["total_reward"]
agent_3 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="SarsaAgent")]["total_reward"]
agent_4 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="PolicyIterationAgent")]["total_reward"]
print(stats.f_oneway(agent_1,agent_2,agent_3,agent_4))
print(pairwise_tukeyhsd(endog=df_agg_f_test[(df_agg_f_test.experiment==exp)]["total_reward"], groups=df_agg_f_test[(df_agg_f_test.experiment==exp)]["agent"], alpha=0.05))
# %%
exp = "Exp4"
agent_1 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="MostFrequentPolicyAgent")]["total_reward"]
agent_2 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="QAgent")]["total_reward"]
agent_3 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="SarsaAgent")]["total_reward"]
agent_4 = df_agg_f_test[(df_agg_f_test.experiment==exp)&(df_agg_f_test.agent=="PolicyIterationAgent")]["total_reward"]
print(stats.f_oneway(agent_1,agent_2,agent_3))
print(pairwise_tukeyhsd(endog=df_agg_f_test[(df_agg_f_test.experiment==exp)]["total_reward"], groups=df_agg_f_test[(df_agg_f_test.experiment==exp)]["agent"], alpha=0.05))
# %%
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
margin_of_err = (stats.norm.ppf((1 + 0.95) / 2)) * (results_agg[("total_reward", "std")]/np.sqrt(results_agg[("total_reward", "count")]))
results_agg[("total_reward", "confidence_intervall_left")] = results_agg[("total_reward", "mean")]-margin_of_err
results_agg[("total_reward", "confidence_intervall_right")] = results_agg[("total_reward", "mean")]+margin_of_err
results_agg
# %%

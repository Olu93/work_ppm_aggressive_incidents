# %%
from matplotlib.axes import Axes
import scipy as sc
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df_original = pd.read_excel('../data/full_data_output.xlsx')
df_original
# %%
col_s = 'Aggression_short'
col_s_prime = 'Next_Type_Eps'
col_env_reaction_days = 'Next_DaysToNext'
col_agent_action = 'reaction'
col_agent_action_orig = '[B09] Maatregelen om agressie te stoppen'
col_aao_mod = 'agent_action'
cols_important = [col_s, col_aao_mod, col_s_prime]
cols_all = [col_agent_action_orig, col_env_reaction_days, col_s, col_s_prime]

df = df_original.copy()[cols_all]
df[col_aao_mod] = df[cols_all][col_agent_action_orig].str.split(';')
df = df.explode(col_aao_mod).drop(columns=col_agent_action_orig)
df = df.dropna(subset=[col_aao_mod, col_env_reaction_days], how='any')
df[col_aao_mod] = df[col_aao_mod].str.strip()
df[col_aao_mod] = df[col_aao_mod].fillna('geen').replace('nan', 'geen') 
df
# %%
group_counts = df.groupby(cols_important).apply(lambda df: pd.Series({"cnt":len(df)})).reset_index()
group_counts

# %%
min_cnt = 2
group_counts["top_k"] = group_counts["cnt"] > min_cnt
topk_groups = group_counts[group_counts["top_k"]==True] 
other_groups = group_counts[group_counts["top_k"]!=True]
topk_groups
# %%
new_df = df.copy()
new_df.loc[(new_df[cols_important].sum(axis=1)).isin(other_groups[cols_important].sum(axis=1)), cols_important] = "Other"
new_df

# %%
groups = list(new_df.groupby(cols_important))

# %%
from collections import defaultdict
import json
params=defaultdict(lambda: defaultdict(dict))
for (incident, action, reaction), gdf in groups:
    t = gdf[col_env_reaction_days]
    fit_p, fit_x =  1 / np.mean(t), np.arange(1, np.max(t) + 1)
    params[incident][action][reaction] = fit_p

params
# %%
import io
json.dump(params, io.open('../data/prob_time_given_incident_action_reaction.json', 'w'), indent=2)
# %%

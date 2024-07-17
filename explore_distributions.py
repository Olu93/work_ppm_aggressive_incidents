# %%
from matplotlib.axes import Axes
import scipy as sc
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df_original = pd.read_excel('data/full_data_output.xlsx')
df_original
# %%
col_env_reaction = 'Next_Type_Eps'
col_env_reaction_days = 'Next_DaysToNext'
col_agent_action = 'reaction'
col_agent_action_orig = '[B09] Maatregelen om agressie te stoppen'
col_aao_mod = 'agent_action'
cols_important = [col_aao_mod, col_env_reaction]
cols_all = [col_agent_action_orig, col_env_reaction_days, col_env_reaction]

df = df_original.copy()[cols_all]
df[col_aao_mod] = df[cols_all][col_agent_action_orig].str.split(';')
df = df.explode(col_aao_mod).drop(columns=col_agent_action_orig)
df = df.dropna(subset=[col_aao_mod, col_env_reaction_days], how='any')
df[col_aao_mod] = df[col_aao_mod].str.strip()
df[col_aao_mod] = df[col_aao_mod].fillna('geen').replace('nan', 'geen') 
df
# %%
def mse_cdf(empirical_data, dist, params):
    ecdf = stats.ecdf(empirical_data).cdf.evaluate(np.sort(empirical_data))
    theoretical_cdf = dist.cdf(np.sort(empirical_data), *params)
    mse = np.mean((ecdf - theoretical_cdf) ** 2)
    return mse

# Function to compute Log-Likelihood
def log_likelihood(empirical_data, dist, params):
    ll = np.sum(dist.logpdf(empirical_data, *params))
    return ll

def plot_gamma(ax:Axes, t, x, idx, num):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(t)
    ax.plot(x, stats.gamma.pdf(x, a=fit_alpha, loc=fit_loc, scale=fit_beta), 'r-', lw=2, label=f'Gamma PDF\n $\\alpha$={fit_alpha:.2f}\n loc={fit_loc:.2f}\n $\\beta=${fit_beta:.2f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.gamma, (fit_alpha, fit_loc, fit_beta))
    ll = log_likelihood(t, stats.gamma, (fit_alpha, fit_loc, fit_beta))
    return mse, ll

def plot_geom(ax:Axes, t, x, idx, num):
    def log_likelihood(empirical_data, dist, params):
        ll = np.sum(dist.logpmf(empirical_data, *params))
        return ll    
    fit_p, fit_x =  1 / np.mean(t), np.arange(1, np.max(t) + 1)
    ax.plot(fit_x, stats.geom.pmf(fit_x, fit_p), 'r-', lw=2, label=f'Geometric PMF\np={fit_p:.2f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.geom, (fit_p, ))
    ll = log_likelihood(t, stats.geom, (fit_p,))
    return mse, ll

def plot_exp(ax:Axes, t, x, idx, num):
    fit_loc, fit_scale = stats.expon.fit(t)
    lambda_exp = 1/fit_scale
    ax.plot(x, stats.expon.pdf(x, loc=fit_loc, scale=fit_scale), 'r-', lw=2, label=f'Exponential PDF\nlambda={lambda_exp:.2f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.expon, (fit_loc, fit_scale))
    ll = log_likelihood(t, stats.expon, (fit_loc, fit_scale))
    return mse, ll

def plot_weibull(ax:Axes, t, x, idx, num):
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(t, floc=0)
    ax.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc=loc_weibull, scale=scale_weibull), 'r-', lw=2, label=f'Weibull PDF\nshape={shape_weibull:.2f}, scale={scale_weibull:.2f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.weibull_min, (shape_weibull, loc_weibull, scale_weibull))
    ll = log_likelihood(t, stats.weibull_min, (shape_weibull, loc_weibull, scale_weibull))
    return mse, ll


# %%
group_counts = df.groupby(cols_important).apply(lambda df: pd.Series({"cnt":len(df)})).reset_index()
group_counts
# %%
fs = []
iss = []
for i, l in enumerate(range(1, len(group_counts), 1)):
    f = np.sum(group_counts.sort_values('cnt')[::-1][:l].cnt)/group_counts.cnt.sum()
    fs.append(f)
    iss.append(i)

plt.plot(iss, fs)
plt.title('Fraction of data covered if groups are limited to top k')
plt.show()

# %%
min_cnt = 50
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
gamma_metrics =  []
geom_metrics =  []
exp_metrics =  []
weibull_metrics =  []
for idx, gdf in groups:
    # if len(gdf) < min_cnt:
    #     continue
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    t = gdf[col_env_reaction_days]
    x = np.linspace(0, np.max(t), 100)
    gamma_metrics.append(plot_gamma(ax[0], t, x, idx, len(gdf)))
    geom_metrics.append(plot_geom(ax[1], t, x, idx, len(gdf)))
    exp_metrics.append(plot_exp(ax[2], t, x, idx, len(gdf)))
    weibull_metrics.append(plot_weibull(ax[3], t, x, idx, len(gdf)))
    fig.suptitle(f"Distributions for {idx} with {len(gdf)} datapoints")
    fig.tight_layout()
    # break
    plt.show()

# %%
df_gamma_metrics = pd.DataFrame(gamma_metrics, columns=['mse', 'll']).assign(dist="gamma")
df_geom_metrics = pd.DataFrame(geom_metrics, columns=['mse', 'll']).assign(dist="geom")
df_exp_metrics = pd.DataFrame(exp_metrics, columns=['mse', 'll']).assign(dist="exp")
df_weibull_metrics = pd.DataFrame(weibull_metrics, columns=['mse', 'll']).assign(dist="weibull")

df_metrics = pd.concat([df_gamma_metrics, df_geom_metrics, df_exp_metrics, df_weibull_metrics])
df_metrics
# %%
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
sns.boxplot(data=df_metrics, x='dist', y='mse', ax=ax1)
# sns.boxplot(data=df_metrics, x='dist', y='mse', ax=ax1)

# %%
from collections import defaultdict
import json
params=defaultdict(dict)
for (action, reaction), gdf in groups:
    t = gdf[col_env_reaction_days]
    fit_p, fit_x =  1 / np.mean(t), np.arange(1, np.max(t) + 1)
    params[action][reaction] = fit_p

params
# %%
import io
json.dump(params, io.open('data/prob_time_given_action_reaction.json', 'w'), indent=2)
# %%

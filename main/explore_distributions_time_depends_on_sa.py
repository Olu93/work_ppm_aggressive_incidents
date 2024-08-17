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
cols_important = [col_s, col_aao_mod]
cols_all = [col_agent_action_orig, col_env_reaction_days, col_s, col_s_prime]

df = df_original.copy()[cols_all]
df[col_aao_mod] = df[cols_all][col_agent_action_orig].str.split(';')
df = df.explode(col_aao_mod).drop(columns=col_agent_action_orig)
df = df.dropna(subset=[col_aao_mod, col_env_reaction_days], how='any')
df[col_env_reaction_days] = (df[col_env_reaction_days] / 10).astype(int)
df[col_aao_mod] = df[col_aao_mod].str.strip()
df[col_aao_mod] = df[col_aao_mod].fillna('geen').replace('nan', 'geen') 
df[col_aao_mod] = df[col_aao_mod].replace({
    "client toegesproken/gesprek met client": "Talk with client",
    "contact beeindigd/weggegaan":"Contact terminated",
    "client afgeleid":"Client distracted",
    "geen":"No measure",
    "met kracht tegen- of vastgehouden":"Hold with force",
    "naar andere kamer/ruimte gestuurd":"Send to other room",
    "afzondering (deur op slot)":"Seclusion",
})
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
    ax.plot(x, stats.gamma.pdf(x, a=fit_alpha, loc=fit_loc, scale=fit_beta), 'r-', lw=2, label=f'Gamma PDF\n $\\alpha$={fit_alpha:.3f}\n loc={fit_loc:.3f}\n $\\beta=${fit_beta:.3f}')
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
    ax.plot(fit_x, stats.geom.pmf(fit_x, fit_p), 'r-', lw=2, label=f'Geometric PMF\np={fit_p:.3f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.geom, (fit_p, ))
    ll = log_likelihood(t, stats.geom, (fit_p,))
    return mse, ll

def plot_exp(ax:Axes, t, x, idx, num):
    fit_loc, fit_scale = stats.expon.fit(t)
    lambda_exp = 1/fit_scale
    ax.plot(x, stats.expon.pdf(x, loc=fit_loc, scale=fit_scale), 'r-', lw=2, label=f'Exponential PDF\nlambda={lambda_exp:.3f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.expon, (fit_loc, fit_scale))
    ll = log_likelihood(t, stats.expon, (fit_loc, fit_scale))
    return mse, ll

def plot_weibull(ax:Axes, t, x, idx, num):
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(t, floc=0)
    ax.plot(x, stats.weibull_min.pdf(x, shape_weibull, loc=loc_weibull, scale=scale_weibull), 'r-', lw=2, label=f'Weibull PDF\nshape={shape_weibull:.3f}\nscale={scale_weibull:.3f}')
    ax.hist(t, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.legend()
    mse = mse_cdf(t, stats.weibull_min, (shape_weibull, loc_weibull, scale_weibull))
    ll = log_likelihood(t, stats.weibull_min, (shape_weibull, loc_weibull, scale_weibull))
    return mse, ll


# %%
group_counts = df.groupby(cols_important).apply(lambda df: pd.Series({"cnt":len(df)})).reset_index().sort_values('cnt')
group_counts
# %%
fs = []
iss = []
# for i, l in enumerate(group_counts.cnt.values):
for i, l in enumerate(range(0, group_counts.cnt.max(), 10)):
    f = np.sum(group_counts[group_counts.cnt > l].cnt)/group_counts.cnt.sum()
    fs.append(f)
    iss.append(l)

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
groups = list(new_df.groupby(cols_important))[:10]
gamma_metrics =  []
geom_metrics =  []
exp_metrics =  []
weibull_metrics =  []
for idx, gdf in groups:
    # if len(gdf) < min_cnt:
    #     continue
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax = ax.flatten()
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

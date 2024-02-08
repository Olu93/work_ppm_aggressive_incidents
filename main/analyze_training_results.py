# %%
import pandas as pd
# %%
df_mean_results = pd.read_csv('experiment_parameter_search_mean.csv')
df_mean_results
# %%
best_configs = (df_mean_results.groupby([
            "agent",
            "epsilon",
            "alpha",
            "gamma",
            "min_inc",
            "rew_type",
        ]).mean().reset_index().sort_values(
        "total_reward", ascending=False).groupby([
            "min_inc",
            "rew_type",
            "agent",
        ]).apply(lambda df: df.head(1))).drop(["agent", "min_inc",
            "rew_type"], axis=1)
best_configs

    # best_configs.to_csv("experiment_best_params.csv")
# %%

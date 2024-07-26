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
df[col_aao_mod] = df[col_aao_mod].str.strip()
df[col_aao_mod] = df[col_aao_mod].fillna('geen').replace('nan', 'geen')
df[col_env_reaction_days] /= 10
df = df.reset_index()
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
params=defaultdict(dict)
for (incident, action), gdf in groups:
    t = gdf[col_env_reaction_days]
    shape_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(t, floc=0)
    params[incident][action] = {'shape':shape_weibull, 'loc':loc_weibull, 'scale': scale_weibull}

params
# %%
import io
json.dump(params, io.open('../data/prob_time_given_incident_action.json', 'w'), indent=2)
# %%

# %%
dataset = df[cols_important+[col_env_reaction_days, col_s_prime]]
dataset

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix

# Make a copy of the dataset
dataset_mod = dataset.copy()

# Define the preprocessing for the categorical features
categorical_features = cols_important
categorical_transformer = OneHotEncoder(drop='first')

# Define the preprocessing for the numerical features
numeric_features = [col_env_reaction_days]
numeric_transformer = StandardScaler()

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, numeric_features)
    ])

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegressionCV(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        cv=5,  # 5-fold cross-validation
        Cs=10  # 10 different values of C to try
    ))
])

# Splitting the data
X = dataset_mod.drop(columns=[col_s_prime])
y = dataset_mod[col_s_prime]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
print("Predicted Probabilities:\n", pipeline.predict_proba(X_test))

# %%
import joblib

# Save the pipeline
joblib_file = "../data/logistic_regression_pipeline.pkl"
joblib.dump(pipeline, joblib_file)

# Load the pipeline
loaded_pipeline = joblib.load(joblib_file)

# Verify the loaded pipeline
y_pred_loaded = loaded_pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred_loaded))
print(classification_report(y_test, y_pred_loaded))
# %%

# %%

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown] cell_id="77035188544a4ca5b186e24c88c7bf96" deepnote_cell_type="markdown" editable=true slideshow={"slide_type": ""}
# # Cirrhosis Classification
#
# Notebook for Cirrhosis classification model.
#

# %%
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_SEED = 42

# %% cell_id="a64ea2c303664a04bf48493c591c583f" deepnote_cell_type="code" execution_context_id="75d2a12e-fc16-40a9-93a6-587a82c272aa" execution_millis=678 execution_start=1742041716415 source_hash="ba2d2c9d"
raw_train = pd.read_csv("./KaggleData/train.csv")
raw_test = pd.read_csv("./KaggleData/test.csv")
sample_sub = pd.read_csv("./KaggleData/sample_submission.csv")

# %% [markdown] cell_id="75d280d6f68a4272b31f9f89259eb809" deepnote_cell_type="markdown"
# ## Explore data
#

# %% cell_id="3da7b5123a604c2195f1aeb626f06814" deepnote_cell_type="code" execution_context_id="75d2a12e-fc16-40a9-93a6-587a82c272aa" execution_millis=67 execution_start=1742041717139 source_hash="b1172c0c"
raw_train.describe().T

# %% cell_id="c201fd84ee954979b3f29f5b0d003b26" deepnote_cell_type="code" execution_context_id="75d2a12e-fc16-40a9-93a6-587a82c272aa" execution_millis=0 execution_start=1742041717260 source_hash="76ec7b87"
raw_train.describe(include="object").T

# %% [markdown] cell_id="764942a9ff1e4606b347a7dda2a271cc" deepnote_cell_type="markdown"
# ## Checking correlation
#

# %% cell_id="f7219e9014c24595a3dfb492f5ae973e" deepnote_cell_type="code" execution_context_id="75d2a12e-fc16-40a9-93a6-587a82c272aa" execution_millis=1278 execution_start=1742041717715 source_hash="b78412bc"
from dython.nominal import associations

cat_corr = associations(
    raw_train.select_dtypes(include=["object", "category"]), cmap="RdBu"
)

# %% cell_id="df91c8c53eab4e47b58bc12e55110ebf" deepnote_cell_type="code" execution_context_id="75d2a12e-fc16-40a9-93a6-587a82c272aa" execution_millis=1207 execution_start=1742041719043 source_hash="786e772b"
exclude_cols = list(
    raw_train.select_dtypes(include=["object", "category"]).columns.drop("Status")
)
exclude_cols += ["id"]

num_corr = associations(
    raw_train.drop(exclude_cols, axis=1), figsize=(10, 10), cmap="RdBu"
)

# %%
train_corr = pd.concat(
    [cat_corr["corr"]["Status"], num_corr["corr"]["Status"]], axis=0
).sort_values()

plt.figure(figsize=(10, 5))
plt.xticks(rotation=45, ha="right")
sns.barplot(train_corr)
plt.ylabel("Correlation with target")


# %% [markdown] cell_id="bdec4f23f5a24c85b49de5e6485f65d4" deepnote_cell_type="markdown"
# ## Handle missing values

# %%
def preprocess_X(raw_data, random_seed=0):
    """
    Preprocess data with numeric and categorical imputation.

    Parameters:
    -----------
    raw_data : pd.DataFrame
        The raw dataset to preprocess
    random_seed : int, default=0
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame: processed data
    """

    # Mapping dictionary embedded in function
    mapping_dict = {
        "Drug": {"Placebo": 0, "D-penicillamine": 1},
        "Sex": {"F": 0, "M": 1},
        "Ascites": {"N": 0, "Y": 1},
        "Hepatomegaly": {"N": 0, "Y": 1},
        "Spiders": {"N": 0, "Y": 1},
        "Edema": {"N": 0, "S": 1, "Y": 2},
    }

    # Create a copy and drop id
    data = raw_data.drop(["id"], axis=1)

    # Handle numeric features
    numeric_data = data.select_dtypes(include=["number"])
    num_imputer = IterativeImputer(max_iter=10, random_state=random_seed)
    num_imputer.set_output(transform="pandas")
    filled_numeric = num_imputer.fit_transform(numeric_data)

    # Handle categorical features
    categorical_data = data.select_dtypes(include=["object"])

    # Map categorical values to numeric
    # For each key in the mapping dictionary
    for col in mapping_dict.keys():
        categorical_data[col] = (
            categorical_data[col].map(mapping_dict[col]).astype(np.float32)
        )

    # Impute categorical features
    cat_imputer = IterativeImputer(
        estimator=RandomForestClassifier(), max_iter=10, random_state=random_seed
    )
    cat_imputer.set_output(transform="pandas")
    filled_categorical = cat_imputer.fit_transform(categorical_data).astype(np.int8)

    # Combine numeric and categorical data
    filled_data = pd.concat([filled_numeric, filled_categorical], axis=1).round(1)

    return filled_data


# %%
def preprocess_y(raw_data):
    """
    Preprocess the target data.

    Parameters:
    -----------
    raw_data : pd.Series
        The raw Series to preprocess

    Returns:
    --------
    pd.Series: processed target data
    """
    target_mapping = {"D": 0, "C": 1, "CL": 2}
    return raw_data.map(target_mapping)


# %%
for col in raw_train.select_dtypes(include=["object", "category"]).columns:
    print(raw_train[col].value_counts())

# %%
raw_train["Drug"] = raw_train["Drug"].replace("N", np.NaN)

# %%
raw_X = raw_train.drop("Status", axis=1)
raw_y = raw_train["Status"]

raw_X_train, raw_X_val, raw_y_train, raw_y_val = train_test_split(
    raw_X, raw_y, test_size=0.2, random_state=RANDOM_SEED
)

X_train = preprocess_X(raw_X_train, random_seed=RANDOM_SEED)
y_train = preprocess_y(raw_y_train)

X_val = preprocess_X(raw_X_val, random_seed=RANDOM_SEED)
y_val = preprocess_y(raw_y_val)

# %% [markdown]
# ## EDA after handling missing values
#
# EDA the data after handling the missing values to see the distribution of the data.
#

# %%
import seaborn as sns
from matplotlib import pyplot as plt

cat_cols = raw_train.drop("Status", axis=1).select_dtypes(include=["object", "category"]).columns

fig, ax = plt.subplots(2, 3, figsize=(10, 10))
plt.tight_layout(pad=3.0)
for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=X_train, ax=ax[i // 3, i % 3])

# %%
y_train.value_counts()

# %% [markdown]
# The distribution of the categorical features is very imbalanced in every feature.
#

# %%
num_cols = raw_train.drop(["Status", "id"], axis=1).select_dtypes(include=["number"]).columns

fig, ax = plt.subplots(4, 3, figsize=(15, 15))

for i, col in enumerate(num_cols):
    sns.histplot(X_train[col], ax=ax[i // 3, i % 3])

# %%
X_train[num_cols].describe().T

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)


# %%
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    
    # Create and train the model with the suggested hyperparameters
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred_proba = model.predict_proba(X_val)
    
    # Calculate multiclass log loss on test data
    # Note: lower log loss is better, but Optuna minimizes by default
    test_loss = log_loss(y_val, y_pred_proba, labels=model.classes_)
    
    # Return the test loss (Optuna will minimize this)
    return test_loss


# %%
print("Starting Optuna optimization...")
study = optuna.create_study(direction='minimize')  # Minimize log loss
study.optimize(objective, n_trials=100)  # 100 trials for thorough search

# %%
best_params = study.best_params

final_model = GradientBoostingClassifier(**best_params, random_state=RANDOM_SEED)
final_model.fit(X_train, y_train)

# %%
y_pred_proba = final_model.predict_proba(X_val)
y_pred = final_model.predict(X_val)

final_test_loss = log_loss(y_val, y_pred_proba, labels=final_model.classes_)
print(f"\nFinal Test Log Loss: {final_test_loss:.4f}")

# %%
plt.figure(figsize=(10, 6))
optimization_history = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')

# %%
feature_imp = pd.Series(
    final_model.feature_importances_, index=X_train.columns, name="feature_importance"
).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Feature Importance")
plt.ylabel("Features");

# %% [markdown]
# ## Testing on Test dataset

# %%
test_proba = final_model.predict_proba(X_val)
test_loss = log_loss(y_val, test_proba)
test_loss

# %% [markdown]
# ## Predicting on real validation set
#
# In this step we will get the prediction of the target on real data from test set. But we also need to clean and preprocess the test data as the test dataset is also dirty.
#

# %%
X_test = preprocess_X(raw_test, random_seed=RANDOM_SEED)
X_test

# %%
submit_pred = final_model.predict_proba(X_test)
submit_pred = pd.DataFrame(submit_pred, columns=["Status_C", "Status_CL", "Status_D"])
submit_pred = pd.concat([raw_test["id"], submit_pred], axis=1)
submit_pred

# %%
submit_pred.to_csv("submission.csv", index=False)

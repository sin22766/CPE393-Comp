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
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
        estimator=DecisionTreeClassifier(), max_iter=10, random_state=random_seed
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
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer, StandardScaler

def preprocess_num_data(num_data: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
    """
    Preprocess numerical data by scaling it.

    Parameters:
    -----------
    num_data : pd.DataFrame
        The numerical data to preprocess

    Returns:
    --------
    pd.DataFrame: preprocessed numerical data
    """


    num_data = num_data.copy()
    num_cols = num_data.columns
    
    def data_ploter(data, title):
        if not plot:
            return

        fig, ax = plt.subplots(4, 3, figsize=(15, 15))
        fig.suptitle(title)
        fig.tight_layout(pad=3.0)
        for i, col in enumerate(num_cols):
            sns.histplot(data[col], ax=ax[i // 3, i % 3])
    
    def get_skewed_cols(data):
        num_skew_abs = data.skew().abs().sort_values(ascending=False)
        skewed_cols = num_skew_abs[num_skew_abs > 0.5].index
        return skewed_cols
    
    data_ploter(num_data, "Before Scaling")

    # Scale the data
    skewed_cols = get_skewed_cols(num_data)
    num_data[skewed_cols] = num_data[skewed_cols].apply(lambda x: winsorize(x, limits=0.01))

    data_ploter(num_data, "After Winsorizing")

    skewed_cols = get_skewed_cols(num_data)
    pt = PowerTransformer()
    pt.set_output(transform="pandas")
    num_data[skewed_cols] = pt.fit_transform(num_data[skewed_cols])

    ss = StandardScaler()
    ss.set_output(transform="pandas")
    num_data = ss.fit_transform(num_data)

    data_ploter(num_data, "After Power Transforming & Standard Scaling")

    return num_data    

# %%
num_cols = raw_train.drop(["Status", "id"], axis=1).select_dtypes(include=["number"]).columns

X_scaled_train = X_train.copy()
X_scaled_train[num_cols] = preprocess_num_data(X_train[num_cols])
X_scaled_train[num_cols]

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
import optuna

# %%
X_scaled_val = X_val.copy()
X_scaled_val[num_cols] = preprocess_num_data(X_scaled_val[num_cols])


# %%
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        # Search around the current learning rate (±50%)
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.047831988317386595 * 0.5, 0.047831988317386595 * 1.5, log=True
        ),
        # Search around current n_estimators with some flexibility
        "n_estimators": trial.suggest_int("n_estimators", max(100, 468 - 100), 468 + 100),
        # Search depth around current value
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        # Try different feature selection strategies but include current
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        # Search around current min_samples_leaf
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", max(1, 10 - 5), 10 + 5),
        # Search around current min_samples_split
        "min_samples_split": trial.suggest_int("min_samples_split", max(2, 12 - 6), 12 + 6),
        # Search around current subsample rate
        "subsample": trial.suggest_float(
            "subsample", max(0.5, 0.6763658784493454 - 0.15), min(1.0, 0.6763658784493454 + 0.15)
        ),
        "random_state": RANDOM_SEED,
    }

    model = GradientBoostingClassifier(**params)

    X_train_fold = X_train
    y_train_fold = y_train

    use_smote = trial.suggest_categorical("use_smote", [True, False])
    if use_smote:
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

    # Fit the model
    model.fit(X_train_fold, y_train_fold)

    # Predict and calculate loss
    y_pred_val = model.predict_proba(X_scaled_val)
    val_loss = log_loss(y_val, y_pred_val)

    # Return the mean validation loss
    return val_loss


# Create a study object and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# %%
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_SEED)
model = GradientBoostingClassifier(
    learning_rate=0.047831988317386595,
    max_depth=4,
    max_features="log2",
    min_samples_leaf=10,
    min_samples_split=12,
    n_estimators=468,
    subsample=0.6763658784493454,
    random_state=RANDOM_SEED,
)

# %%
import os

if os.name == 'nt':
    os.environ['LOKY_MAX_CPU_COUNT'] = "6"

# %%
train_loss = []
val_loss = []

for train_index, val_index in rskf.split(X_scaled_train, y_train):
    X_train_fold, X_val_fold = X_scaled_train.iloc[train_index], X_scaled_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)

    model.fit(X_train_fold, y_train_fold)

    y_pred_train = model.predict_proba(X_train_fold)
    y_pred_val = model.predict_proba(X_val_fold)

    train_loss.append(log_loss(y_train_fold, y_pred_train))
    val_loss.append(log_loss(y_val_fold, y_pred_val))

# %%
cv_loss = pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss})
cv_loss

sns.lineplot(data=cv_loss)

# %%
feature_imp = pd.Series(
    model.feature_importances_, index=X_train.columns, name="feature_importance"
).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel("Feature Importance")
plt.ylabel("Features");

# %% [markdown]
# ## Testing on Test dataset

# %%
X_scaled_val = X_val.copy()
X_scaled_val[num_cols] = preprocess_num_data(X_scaled_val[num_cols])

# %%
y_pred_val = model.predict_proba(X_scaled_val)
log_loss(y_val, y_pred_val)

# %% [markdown]
# ## Predicting on real validation set
#
# In this step we will get the prediction of the target on real data from test set. But we also need to clean and preprocess the test data as the test dataset is also dirty.
#

# %%
X_test = preprocess_X(raw_test, random_seed=RANDOM_SEED)
X_test

# %%
X_scaled_test = X_test.copy()
X_scaled_test[num_cols] = preprocess_num_data(X_scaled_test[num_cols])

# %%
submit_pred = model.predict_proba(X_scaled_test)
# Reorder the columns from {"D": 0, "C": 1, "CL": 2} to {"C": 0, "CL": 1, "D": 2}
submit_pred = submit_pred[:, [1, 2, 0]]
submit_pred = pd.DataFrame(submit_pred, columns=["Status_C", "Status_CL", "Status_D"])
submit_pred = pd.concat([raw_test["id"], submit_pred], axis=1)
submit_pred

# %%
submit_pred.to_csv("submission.csv", index=False)

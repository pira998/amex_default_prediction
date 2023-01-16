import itertools
import random
from random import sample

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import optuna  # pip install optuna
import pandas as pd
import seaborn as sns
from category_encoders import TargetEncoder
from matplotlib.gridspec import GridSpec
from optuna.integration import LightGBMPruningCallback
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (  # we use F-TEST for numerical variables
    SelectKBest, chi2, f_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)
from sklearn.svm import SVC, LinearSVC
from tqdm.auto import tqdm

INT8_MIN = np.iinfo(np.int8).min
INT8_MAX = np.iinfo(np.int8).max
INT16_MIN = np.iinfo(np.int16).min
INT16_MAX = np.iinfo(np.int16).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max


def memory_usage(data, detail=1):
    if detail:
        display(data.memory_usage())
    memory = data.memory_usage().sum() / (1024 * 1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory


def compress_dataset(data):
    """
        Compress datatype as small as it can
        Parameters
        ----------
        path: pandas Dataframe

        Returns
        -------
            None
    """
    memory_before_compress = memory_usage(data, 0)
    print()
    length_interval = 50
    length_float_decimal = 4

    print("=" * length_interval)
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != "object":
            print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == "float64":
                print(
                    " variable min: {0:15s} max: {1:15s}".format(
                        str(np.round(col_min, length_float_decimal)),
                        str(np.round(col_max, length_float_decimal)),
                    )
                )
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    print(
                        "  float16 min: {0:15s} max: {1:15s}".format(
                            str(FLOAT16_MIN), str(FLOAT16_MAX)
                        )
                    )
                    print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    print(
                        "  float32 min: {0:15s} max: {1:15s}".format(
                            str(FLOAT32_MIN), str(FLOAT32_MAX)
                        )
                    )
                    print("compress float64 --> float32")
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print(
                    "Compress Rate: [{0:.2%}]".format(
                        (memory_before_compress - memory_after_compress)
                        / memory_before_compress
                    )
                )
                print("=" * length_interval)

            if col_dtype == "int64":
                print(
                    " variable min: {0:15s} max: {1:15s}".format(
                        str(col_min), str(col_max)
                    )
                )
                type_flag = 64
                if (col_min > INT8_MIN / 2) and (col_max < INT8_MAX / 2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    print(
                        "     int8 min: {0:15s} max: {1:15s}".format(
                            str(INT8_MIN), str(INT8_MAX)
                        )
                    )
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    print(
                        "    int16 min: {0:15s} max: {1:15s}".format(
                            str(INT16_MIN), str(INT16_MAX)
                        )
                    )
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    print(
                        "    int32 min: {0:15s} max: {1:15s}".format(
                            str(INT32_MIN), str(INT32_MAX)
                        )
                    )
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memory_usage(data, 0)
                print(
                    "Compress Rate: [{0:.2%}]".format(
                        (memory_before_compress - memory_after_compress)
                        / memory_before_compress
                    )
                )
                if type_flag == 32:
                    print("compress (int64) ==> (int32)")
                elif type_flag == 16:
                    print("compress (int64) ==> (int16)")
                else:
                    print("compress (int64) ==> (int8)")
                print("=" * length_interval)

    print()
    memory_after_compress = memory_usage(data, 0)
    print(
        "Compress Rate: [{0:.2%}]".format(
            (memory_before_compress - memory_after_compress) / memory_before_compress
        )
    )


# use dask to read in data

dev_data = dd.read_csv("train_data.csv")
dev_data_df = dev_data.compute()  # this converts dask to pandas
compress_dataset(dev_data_df)

# dev_data_df.to_hdf('dev_data_df.h5', key='df')
# dev_data_df = pd.read_hdf('dev_data_df.h5', 'df')
dev_label = dd.read_csv("train_labels.csv")
dev_label_df = dev_label.compute()  # this converts dask to pandas
dev_label_df = dev_label_df.set_index(["customer_ID"])


def missing(data):
    """
    Function: find number and percent of missing values in the data
    Input: data
    Output: variables with missing value and the missing percent
    """
    missing_number = data.isna().sum().sort_values(ascending=False)
    missing_percent = round(
        (data.isna().sum() / data.isna().count()).sort_values(ascending=False), 3
    )
    missing_values = pd.concat(
        [missing_number, missing_percent],
        axis=1,
        keys=["Missing_Number", "Missing_Percent"],
    )
    missing_df = missing_values[missing_values["Missing_Number"] > 0]
    return missing_df


missing_value_count = missing(dev_data_df)


cat_vars = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "month",
    "day_of_week",
]
features = dev_data_df.drop(["customer_ID", "S_2"], axis=1).columns.to_list()
num_vars = list(filter(lambda x: x not in cat_vars, features))

delequincy_vars = filter(lambda x: x.startswith("D") and x not in cat_vars, features)
spend_vars = filter(lambda x: (x.startswith("S")) and (x not in cat_vars), features)
payment_vars = filter(lambda x: x.startswith("P") and x not in cat_vars, features)
balance_vars = filter(lambda x: x.startswith("B") and x not in cat_vars, features)
risk_vars = filter(lambda x: x.startswith("R") and x not in cat_vars, features)

week_days = {1: "Mon", 2: "Tue", 3: "Wen", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def extract_date_vars(
    df, date_var="S_2", sort_by=["customer_ID", "S_2"], week_days=week_days
):
    # change to datetime
    df[date_var] = pd.to_datetime(df[date_var])
    # sort by customer by date
    df = df.sort_values(by=sort_by)
    # extract some date characteristics
    # month
    df["month"] = df[date_var].dt.month
    # day of week
    df["day_of_week"] = df[date_var].apply(lambda x: x.isocalendar()[-1])
    return df


dev_data_df = extract_date_vars(
    dev_data_df, date_var="S_2", sort_by=["customer_ID", "S_2"], week_days=week_days
)


def generate_column_names_num(vars=num_vars, agg=["mean", "std", "min", "max", "last"]):
    tmp = []
    for a in agg:
        tmp_i = pd.Series(vars).apply(lambda x: x + "_" + a).tolist()
        tmp.append(tmp_i)
    column_names_num = list(
        itertools.chain(*zip(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4]))
    )
    return column_names_num


def column_agg_num(data, vars=num_vars, agg=["mean", "std", "min", "max", "last"]):
    num_vars_agg = data.groupby("customer_ID")[vars].agg(agg)
    column_names_num = generate_column_names_num(vars=vars, agg=agg)
    num_vars_agg.columns = column_names_num
    return num_vars_agg


num_vars_agg = column_agg_num(
    data=dev_data_df, vars=num_vars, agg=["mean", "std", "min", "max", "last"]
)


def generate_column_names_cat(vars=cat_vars, agg=["count", "nunique", "mode"]):
    tmp = []
    for a in agg:
        tmp_i = pd.Series(vars).apply(lambda x: x + "_" + str(a)).tolist()
        tmp.append(tmp_i)
    column_names_cat = list(itertools.chain(*zip(tmp[0], tmp[1], tmp[2])))
    return column_names_cat


def column_agg_cat(data, vars=cat_vars, agg=["count", "nunique", pd.Series.mode]):
    cat_vars_agg = data.groupby("customer_ID")[vars].agg(agg)
    column_names_cat = ["_".join(x) for x in cat_vars_agg.columns]
    cat_vars_agg.columns = column_names_cat

    mode_cols = filter(lambda x: x.endswith("_mode"), cat_vars_agg.columns)
    mode_cols = list(mode_cols)
    for col in mode_cols:
        cat_vars_agg[col] = cat_vars_agg[col].apply(
            lambda x: x.tolist()[0]
            if (isinstance(x, np.ndarray)) and (len(x) > 0)
            else x
        )

    return cat_vars_agg


cat_vars_agg = column_agg_cat(
    data=dev_data_df, vars=cat_vars, agg=["count", "nunique", pd.Series.mode]
)

dev_agg_groupby = pd.concat([num_vars_agg, cat_vars_agg], axis=1)


# ====================================================
# Get the difference
# ====================================================


def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(["customer_ID"])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis=0)
    # Transform to dataframe
    df1 = pd.DataFrame(
        df1, columns=[col + "_diff1" for col in df[num_features].columns]
    )
    # Add customer id
    df1["customer_ID"] = customer_ids
    return df1


# preprocess the dev dataset
# dev_data, test_data = get_dev_test_set()
dev_agg_diff = get_difference(dev_data_df, num_vars)

dev_agg_diff = dev_agg_diff.set_index("customer_ID")

dev_agg = pd.concat([dev_agg_groupby, dev_agg_diff], axis=1)


data = dev_agg

# ====================================================

cs = filter(
    lambda x: (x.endswith("_mode"))
    or (x.endswith("_unique"))
    or (x.endswith("_count")),
    data.columns,
)
cs = list(cs)

for c in cs:
    data[c] = data[c].apply(lambda x: str(x).strip("[]"))

missing_features = missing(data)

features_to_drop = missing_features[
    missing_features.Missing_Percent > 0.5
].index.tolist()

data = data.drop(features_to_drop, axis=1)

cat_cols = filter(lambda x: x.endswith("_mode"), data.columns)
cat_cols = list(cat_cols)

data_cat = data[cat_cols + ["target"]]
data_cat = data_cat.reset_index()

for c in cat_cols:
    data_tmp = data[[c, "target"]]
    data_tmp["cnt"] = 1
    data_tmp = data_tmp.groupby([c, "target"], as_index=False).count()
    sns.factorplot(x=c, y="cnt", hue="target", data=data_tmp, kind="bar")


X = data.drop(columns=["target"])
y = np.array(data["target"])

# split
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


cat_cols = filter(lambda x: x.endswith("_mode"), X_dev.columns)
cat_cols = list(cat_cols)

num_cols = list(set(X_dev.columns) - set(cat_cols))

# Create pipelines for numerical and categorical features

te = TargetEncoder()

num_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())]
)
cat_pipeline = Pipeline(
    steps=[("impute", SimpleImputer(strategy="most_frequent")), ("target-encode", te)]
)

# Create ColumnTransformer to apply pipeline for each column type
col_trans = ColumnTransformer(
    transformers=[
        ("num_pipeline", num_pipeline, num_cols),
        ("cat_pipeline", cat_pipeline, cat_cols),
    ],
    remainder="drop",
    n_jobs=-1,
)

X_dev_new = col_trans.fit_transform(X_dev, y_dev)
X_test_new = col_trans.transform(X_test)


def get_column_names_from_ColumnTransformer(column_transformer):
    col_name = []
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:  # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name


columns_names = get_column_names_from_ColumnTransformer(col_trans)
columns_names[-11:] = cat_cols

X_dev_new = pd.DataFrame(X_dev_new, columns=columns_names)
X_test_new = pd.DataFrame(X_test_new, columns=columns_names)


import seaborn as sns


def select_features(X_train, y_train, X_test, k_value="all"):
    fs = SelectKBest(score_func=f_classif, k=k_value)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_dev_new, y_dev, X_test_new)
# what are scores for the features
for i in range(len(fs.scores_)):
    print("Feature %d: %f" % (i, fs.scores_[i]))

X_train_fs, X_test_fs, fs = select_features(X_dev_new, y_dev, X_test_new, 300)

features_selected = fs.get_feature_names_out()

X_dev_selected = X_dev_new[features_selected]

X_test_selected = X_test_new[features_selected]


import gc
import glob
import os
import pickle
import sys

## ESSENTIALS
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
import random

random.seed(75)
### warnings setting
import sys
import warnings
from functools import partial, reduce

from tqdm.notebook import tqdm_notebook

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

##### LOGGING Stettings #####

import pickle
import uuid

import catboost
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import auc, roc_auc_score, roc_curve
#### model
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm_notebook


from sklearn.model_selection import StratifiedKFold


def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", **param_grid)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "binary_logloss")
            ],  # Add a pruning callback
        )
        preds = model.predict_proba(X_test)
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)


study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")


def func(trial):
    return objective(trial, X_dev_selected, y_dev)


study.optimize(func, n_trials=20)

print(f"\tBest value (rmse): {study.best_value:.5f}")
print(f"\tBest params:")

for key, value in study.best_params.items():
    print(f"\t\t{key}: {value}")


model = lgb.LGBMClassifier(objective="binary", **study.best_params)
model.fit(X_dev_selected, y_dev, eval_metric="binary_logloss")
preds = model.predict_proba(X_dev_selected)
log_loss(y_dev, preds)

pred_class_dev = model.predict(X_dev_selected)
cm_dev = confusion_matrix(y_dev, pred_class_dev)


print("***** Development Set Analysis *****")

test_acc = sum(np.where(pred_class_dev == y_dev, 1, 0)) / y_dev.shape[0]
print("Accuracy on Test Set : {:0.4f}".format(test_acc))
prfs = precision_recall_fscore_support(y_dev, pred_class_dev, average="macro")
print("Precision : ", prfs[0])
print("Recall : ", prfs[1])
print("F1 Score : ", prfs[2])

from sklearn.metrics import average_precision_score

average_precision_score(y_dev, pred_class_dev_prob[:, 1])

pred_prob_dev = model.predict_proba(X_dev_selected)[:, 1]

pred_class_test = model.predict(X_test_selected)
cm_test = confusion_matrix(y_test, pred_class_test)

print("***** Test Set Analysis *****")

test_acc = sum(np.where(pred_class_test == y_test, 1, 0)) / y_test.shape[0]
print("Accuracy on Test Set : {:0.4f}".format(test_acc))
prfs = precision_recall_fscore_support(y_test, pred_class_test, average="macro")
print("Precision : ", prfs[0])
print("Recall : ", prfs[1])
print("F1 Score : ", prfs[2])

pred_class_test = model.predict(X_test_selected)

pred_class_test_prob = model.predict_proba(X_test_selected)

lightgbm_test_prediction = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred_class": pred_class_test,
        "y_pred_prob": pred_class_test_prob[:, 1],
    }
)
lightgbm_test_prediction.to_csv("lightgbm_test_prediction.csv")

pred_class_dev = model.predict(X_dev_selected)
pred_class_dev_prob = model.predict_proba(X_dev_selected)

lightgbm_dev_prediction = pd.DataFrame(
    {
        "y_dev": y_dev,
        "y_pred_class": pred_class_dev,
        "y_pred_prob": pred_class_dev_prob[:, 1],
    }
)

lightgbm_dev_prediction.to_csv("lightgbm_dev_prediction.csv")


def brier_score(y_test, y_pred):
    losses = np.subtract(y_test, y_pred) ** 2
    brier_score = losses.sum() / len(losses)
    return brier_score


brier_score(y_test, pred_prob_test)


# Isotonic Regression
cal_LGB_isotonic = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
cal_LGB_isotonic.fit(X_dev_selected, y_dev)
display_isotonic = CalibrationDisplay.from_estimator(
    cal_LGB_isotonic,
    X_test_selected,
    y_test,
    n_bins=10,
    name="Calibrated LGB (Isotonic)",
)

# Platt Scaling
cal_LGB_platt = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
cal_LGB_platt.fit(X_dev_selected, y_dev)
display_platt = CalibrationDisplay.from_estimator(
    cal_LGB_platt, X_test_selected, y_test, n_bins=10, name="Calibrated LGB (Platt)"
)

print(
    "Brier Score for Isotonic Regression : {:0.4f}".format(
        brier_score_loss(y_test, cal_LGB_isotonic.predict_proba(X_test_selected)[:, 1])
    )
)
print(
    "Brier Score for Platt Scaling       : {:0.4f}".format(
        brier_score_loss(y_test, cal_LGB_platt.predict_proba(X_test_selected)[:, 1])
    )
)


pred_class_test = cal_LGB_isotonic.predict(X_test_selected)
cm_test = confusion_matrix(y_test, pred_class_test)

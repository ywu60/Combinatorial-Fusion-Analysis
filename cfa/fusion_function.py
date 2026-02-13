
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools # use combination function
from typing import Literal # function definition
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.ticker as ticker # for RSC function graph


def normalization(df: pd.DataFrame) -> pd.DataFrame:    
    minv = df.min(axis = 0)
    maxv = df.max(axis = 0)
    denom = (maxv - minv).replace(0, np.nan)  # Avoid division by zero
    if denom.isna().any():
        print("At leaset one column of the data frame is constant!")

    scaled = (df - minv) / denom
    scaled = scaled.fillna(0.0)
    
    return scaled


def compute_cd_ds(df):
    # Step 1: Scale each column to [0,1] (min–max scaling)
    minv = df.min(axis = 0)
    maxv = df.max(axis = 0)
    denom = (maxv - minv).replace(0, np.nan)  # Avoid division by zero
    if denom.isna().any():
        print("At leaset one column of the data frame is constant!")

    scaled = (df - minv) / denom
    scaled = scaled.fillna(0.0)

    # Step 2: Sort each column descending independently, which is rank-score function
    df_1 = pd.DataFrame({
        c: scaled[c].sort_values(ascending=False).reset_index(drop=True)
        for c in scaled.columns
    })

    # Step 3: Compute CD for each pair (root mean squared difference)
    pairs = list(itertools.combinations(df_1.columns, 2))
    cds = []
    pair_labels = []
    for a, b in pairs:
        x = df_1[a].to_numpy()
        y = df_1[b].to_numpy()
        x_size = x.size
        if x_size == 1:
            x_size = 2 # avoid x_size -1 = 0
        cd = np.sqrt(np.sum( (x - y) ** 2 )/ (x_size - 1))
        cds.append(cd)
        pair_labels.append((a, b))  # tuple for easier filtering later for calcualting ds
    df_2 = pd.DataFrame({"CD": cds}, index=pd.MultiIndex.from_tuples(pair_labels, names=["Col1", "Col2"]))

    # Step 4: For each column, compute ds = average CD over all pairs containing it
    ds_vals = {}
    for col in df_1.columns:
        mask = (df_2.index.get_level_values(0) == col) | (df_2.index.get_level_values(1) == col)
        ds_vals[col] = df_2.loc[mask, "CD"].mean()
    df_3 = pd.DataFrame({"ds": pd.Series(ds_vals)})

    return df_1, df_2, df_3


def average_score_combination(df):
    scoring_sys = {}

    # Generate combinations of 2 to n columns (n-> number of columns in df)
    for r in range(2, df.shape[1]+1):
        for cols in itertools.combinations(df.columns, r):
            combo_name = ''.join(cols)
            avg_scores = df[list(cols)].mean(axis=1)
            scoring_sys[combo_name] = avg_scores

    # Convert to DataFrame
    score_df = pd.DataFrame(scoring_sys, index=df.index)
    return score_df


def average_rank_combination(df):
    # only changed one place from average score combination
    # add rank(ascending = False)
    scoring_sys = {}

    # Generate combinations of 2 to (n+1) columns (n-> number of columns in df)
    for r in range(2, df.shape[1]+1):
        for cols in itertools.combinations(df.columns, r):
            combo_name = ''.join(cols)
            avg_scores = df[list(cols)].rank(ascending = False).mean(axis=1)
            scoring_sys[combo_name] = avg_scores

    # Convert to DataFrame
    score_df = pd.DataFrame(scoring_sys, index=df.index)
    return score_df


def weighted_score_combination_by_ds(df):
    scoring_sys = {}

    # Generate combinations of 2 to (n+1) columns (n-> number of columns in df)
    for r in range(2, df.shape[1]+1):
        for cols in itertools.combinations(df.columns, r):
            combo_name = ''.join(cols)
            df_subset = df[list(cols)]
            _, _, df_weights = compute_cd_ds(df_subset) # only need diversity strength
            weights = df_weights.loc[df_subset.columns, 'ds'] # weights is pd.Series here
            weights = weights/sum(weights)
            weighted_scores = df_subset.mul(weights, axis = 1).sum(axis = 1)
            scoring_sys[combo_name] = weighted_scores

    # Convert to DataFrame
    score_df = pd.DataFrame(scoring_sys, index=df.index)
    return score_df


def weighted_rank_combination_by_ds(df):
    # only changed two places from weighted score combination
    # add rank(ascending = False)
    scoring_sys = {}

    # Generate combinations of 2 to (n+1) columns (n-> number of columns in df)
    for r in range(2, df.shape[1]+1):
        for cols in itertools.combinations(df.columns, r):
            combo_name = ''.join(cols)
            df_subset = df[list(cols)]
            _, _, df_weights = compute_cd_ds(df_subset)
            weights = df_weights.loc[df_subset.columns, 'ds'] # weights is pd.Series here
            weights = weights/sum(weights)
            weighted_scores = df_subset.rank(ascending = False).mul(weights, axis = 1).sum(axis = 1) # pd.Series
            scoring_sys[combo_name] = weighted_scores

    # Convert to DataFrame
    score_df = pd.DataFrame(scoring_sys, index=df.index)
    return score_df


def compute_performance(df, perf_metric: Literal["accuracy", "auroc"], y_true, score = True):
    # the reason that we need to distinguish score or rank when computing performance is
    # for ranks, we covnert to values between 0 and 1 and then compare to 0.5
    # since rank 1 is the highest and will be converted to 0, 
    # converted values less than 0.5 should have label 1.
    
    if perf_metric not in ("accuracy", "auroc"):
        raise ValueError("Performance metric we support now is 'accuracy' or 'auroc'!")
    if score:
        if perf_metric == 'accuracy':
            acc_dict = {}
            for col in df.columns:
                y_binary = (df[col] >= 0.5).astype(int)
                acc_dict[col] = accuracy_score(y_true, y_binary)
            perf = pd.Series(acc_dict, name = 'accuracy')
        elif perf_metric == 'auroc':
            auc_dict = {}
            for col in df.columns:
                auc_dict[col] = roc_auc_score(y_true, df[col])
                
            perf = pd.Series(auc_dict, name = 'auroc')
    else: 
        df_n = (df - df.min()) / (df.max() - df.min()) # normalize ranks to [0, 1]

        if perf_metric == 'accuracy':
            acc_dict = {}
            for col in df_n.columns:
                y_binary = (df_n[col] <= 0.65).astype(int)
                acc_dict[col] = accuracy_score(y_true, y_binary)
            perf = pd.Series(acc_dict, name = 'accuracy')
        
        elif perf_metric == 'auroc': # here df or df_n is fine because auroc is based on ranks
            # normalization doesn't change ranks
            auc_dict = {}
            for col in df.columns:
                auc_dict[col] = 1- roc_auc_score(y_true, df[col]) # auc is based on ascending order
                
            perf = pd.Series(auc_dict, name = 'auroc')
        
    return perf


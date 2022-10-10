import cv2
import numpy as np
import os, os.path as osp
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def create_dir(d):
    if not osp.exists(d): os.makedirs(d)


def create_double_cv(df, id_column, num_inner, num_outer, stratified=None, seed=88):
    np.random.seed(seed)
    df = df.reset_index(drop=True)
    df["outer"] = -1
    kfold_class = GroupKFold if not isinstance(stratified, str) else StratifiedGroupKFold
    if not isinstance(stratified, str): 
        stratified = id_column
    else:
        print(f"Stratifying CV folds based on `{stratified}` ...")
    outer_kfold = kfold_class(n_splits=num_outer)
    outer_split = outer_kfold.split(X=df[id_column], y=df[stratified], groups=df[id_column])
    for outer_fold, (outer_train, outer_valid) in enumerate(outer_split):
        df.loc[outer_valid, "outer"] = outer_fold 
        df[f"inner{outer_fold}"] = -1
        inner_df = df[df.outer != outer_fold].copy()
        inner_kfold = kfold_class(n_splits=num_inner)
        inner_split = inner_kfold.split(X=inner_df[id_column], y=inner_df[stratified], groups=inner_df[id_column])
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_split):
            inner_df = inner_df.reset_index(drop=True)
            inner_valid_ids = inner_df.loc[inner_valid, id_column].tolist()
            df.loc[df[id_column].isin(inner_valid_ids), f"inner{outer_fold}"] = inner_fold
    # Do a few checks
    for outer_fold in df.outer.unique():
        train_df = df.loc[df.outer != outer_fold]
        valid_df = df.loc[df.outer == outer_fold]
        assert len(set(train_df[id_column].tolist()) & set(valid_df[id_column].tolist())) == 0
        inner_col = f"inner{outer_fold}"
        for inner_fold in df[inner_col].unique():
            inner_train = train_df[train_df[inner_col] != inner_fold]
            inner_valid = train_df[train_df[inner_col] == inner_fold]
            assert len(set(inner_train[id_column].tolist()) & set(inner_valid[id_column].tolist())) == 0
        assert valid_df[f"inner{outer_fold}"].unique() == np.asarray([-1])
    return df


def overlay_images(image, overlay, weights=[1.0, 0.4]):
    overlaid_image = cv2.addWeighted(image, weights[0], overlay, weights[1], 0)
    return overlaid_image

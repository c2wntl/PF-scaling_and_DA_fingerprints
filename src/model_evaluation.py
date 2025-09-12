import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os


def get_eval_results(
    model, X_train, X_test, y_train, y_test, SAVEPATH=None, weights=None
):
    y_pred_trian = model.predict(X_train)
    MAE_train = mean_absolute_error(
        y_train, y_pred_trian, sample_weight=weights.loc[y_train.index]
    )
    MSE_train = mean_squared_error(
        y_train, y_pred_trian, sample_weight=weights.loc[y_train.index]
    )
    R2_train = r2_score(y_train, y_pred_trian, sample_weight=weights.loc[y_train.index])

    y_pred_test = model.predict(X_test)
    MAE_test = mean_absolute_error(
        y_test, y_pred_test, sample_weight=weights.loc[y_test.index]
    )
    MSE_test = mean_squared_error(
        y_test, y_pred_test, sample_weight=weights.loc[y_test.index]
    )
    R2_test = r2_score(y_test, y_pred_test, sample_weight=weights.loc[y_test.index])

    results = pd.DataFrame(
        {
            "R2_train": [R2_train],
            "R2_test": [R2_test],
            "MAE_train": [MAE_train],
            "MAE_test": [MAE_test],
            "MSE_train": [MSE_train],
            "MSE_test": [MSE_test],
        }
    )
    if SAVEPATH is not None:
        results.to_csv(SAVEPATH)

    return results, y_pred_test


def get_eval_results_from_cv(cv_score, SAVEPATH=None):
    # Mean values
    test_means = [
        cv_score["test_r2"].mean(),
        -cv_score["test_neg_mean_absolute_error"].mean(),
        -cv_score["test_weighted_mae"].mean(),
    ]
    train_means = [
        cv_score["train_r2"].mean(),
        -cv_score["train_neg_mean_absolute_error"].mean(),
        -cv_score["train_weighted_mae"].mean(),
    ]

    # Standard deviations
    test_stds = [
        cv_score["test_r2"].std(),
        cv_score["test_neg_mean_absolute_error"].std(),
        cv_score["test_weighted_mae"].std(),
    ]
    train_stds = [
        cv_score["train_r2"].std(),
        cv_score["train_neg_mean_absolute_error"].std(),
        cv_score["train_weighted_mae"].std(),
    ]

    eval_df = pd.DataFrame(
        {
            "train_r2_mean": [train_means[0]],
            "test_r2_mean": [test_means[0]],
            "train_r2_std": [train_stds[0]],
            "test_r2_std": [test_stds[0]],
            "train_mae_mean": [train_means[1]],
            "test_mae_mean": [test_means[1]],
            "train_mae_std": [train_stds[1]],
            "test_mae_std": [test_stds[1]],
            "train_weighted_mae_mean": [train_means[2]],
            "test_weighted_mae_mean": [test_means[2]],
            "train_weighted_mae_std": [train_stds[2]],
            "test_weighted_mae_std": [test_stds[2]],
        }
    )
    if SAVEPATH is not None:
        eval_df.to_csv(SAVEPATH)
    return train_means, test_means, train_stds, test_stds


def get_feature_importance(model, features, N=15, SAVEPATH=None):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": features, "Importance": feature_importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    if SAVEPATH is not None:

        feature_importance_df.to_csv(SAVEPATH)

    return feature_importance_df.head(N)

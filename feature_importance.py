import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

def compute_perm_feature_importance(model: BaseEstimator, cfg: Config, X_test: pd.DataFrame, y_test: pd.Series):
    """ Compute feature importance using sk-learn permutation_importance"""

    feat_import = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=cfg.random_state, n_jobs=-1)

    # Print the feature importances
    for i in feat_import.importances_mean.argsort()[::-1]:
        if feat_import.importances_mean[i] - 2 * feat_import.importances_std[i] > 0:
            print(f"{X_test.columns[i]:<8}"
                f"{feat_import.importances_mean[i]:.3f}"
                f" +/- {feat_import.importances_std[i]:.3f}")

    # Sort feature importances in descending order
    indices = np.argsort(feat_import.importances_mean)[::-1]

    # save feature importance
    cfg.feat_importance = feat_import
    
    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(X_test.shape[1]), feat_import.importances_mean[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_test.shape[1]), X_test.columns, rotation=90)

    # Save plot
    plt.tight_layout()
    plt.savefig(f'{cfg.path}/feature_importance.png')
    plt.cla()
    
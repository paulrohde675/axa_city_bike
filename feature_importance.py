import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import Config
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance


def compute_perm_feature_importance(
    model: BaseEstimator, cfg: Config, X_test: pd.DataFrame, y_test: pd.Series
):
    """Compute feature importance using sk-learn permutation_importance"""

    feat_import = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=cfg.random_state, n_jobs=-1
    )

    # Print the feature importances
    for i in feat_import.importances_mean.argsort()[::-1]:
        if feat_import.importances_mean[i] - 2 * feat_import.importances_std[i] > 0:
            print(
                f"{X_test.columns[i]:<8}"
                f"{feat_import.importances_mean[i]:.3f}"
                f" +/- {feat_import.importances_std[i]:.3f}"
            )

    # Sort feature importances in descending order
    indices = np.argsort(feat_import.importances_mean)[::-1]

    # save feature importance
    cfg.feat_importance = feat_import

    # Create plot
    fig, ax = plt.subplots()

    # Create plot title
    ax.set_title("Feature Importance", fontsize=16)

    # Add bars with color palette
    bars = ax.bar(
        range(X_test.shape[1]),
        feat_import.importances_mean[indices],
        color=sns.color_palette("hsv", X_test.shape[1]),
    )

    # Add feature names as x-axis labels
    ax.set_xticks(range(X_test.shape[1]))
    ax.set_xticklabels(X_test.columns[indices], rotation=90, fontsize=12)

    # Increase y-tick font size
    ax.yaxis.set_tick_params(labelsize=12)

    # Add y-axis label
    ax.set_ylabel("Importance", fontsize=14)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add padding
    plt.gcf().subplots_adjust(bottom=0.15)

    # Save plot
    fig.tight_layout()
    plt.savefig(f"{cfg.path}/feature_importance.png")
    cfg.plt_feat_importance = fig
    plt.close(fig)

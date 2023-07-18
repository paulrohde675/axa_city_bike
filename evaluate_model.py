import seaborn as sns
from config import Config
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model: BaseEstimator, cfg: Config, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """ Evaluates a trained model and stores the metrics in the experiment config """

    # Prediction on test split
    y_pred = model.predict(X_test)
    
    # generate classification report
    cfg.model_report = classification_report(y_test, y_pred)
    cfg.model_report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(cfg.model_report)

    cfg.accuracy = accuracy_score(y_test, y_pred)
    cfg.precision = precision_score(y_test, y_pred)
    cfg.recall = recall_score(y_test, y_pred)
    cfg.f1 = f1_score(y_test, y_pred)
    cfg.auc_roc = roc_auc_score(y_test, y_pred)
    
    # Plot ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc), linewidth=2)

    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)

    # Set labels, title and ticks
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)

    # Customize legend
    legend = ax.legend(loc='lower right', fontsize=12, frameon=True)
    legend.get_frame().set_edgecolor('black')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save plot
    fig.tight_layout()
    plt.savefig(f'{cfg.path}/roc_curve.png')
    cfg.plt_roc = fig
    plt.close(fig)

    
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    # Create a heatmap for the confusion matrix
    cax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, square=True)

    # Set labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Negative', 'Positive'], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Negative', 'Positive'], fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=16)

    # Add a colorbar
    fig.colorbar(cax.get_children()[0], ax=ax, orientation="vertical", pad=0.05)

    # Save plot
    fig.tight_layout()
    plt.savefig(f'{cfg.path}/confusion_matrix.png')
    cfg.plt_confusion_matrix = fig
    plt.close(fig)

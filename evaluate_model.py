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
    
    # plot ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    fig.savefig(f'{cfg.path}/roc_curve.png')
    cfg.plt_roc = fig
    plt.cla()
    
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    cax = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.set_title('Confusion Matrix')
    fig.savefig(f'{cfg.path}/confusion_matrix.png')
    cfg.plt_confusion_matrix = fig
    plt.cla()

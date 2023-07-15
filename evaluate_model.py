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
    classification_refport = classification_report(y_test, y_pred)
    cfg.classification_refport = classification_refport
    print(classification_refport)

    cfg.accuracy = accuracy_score(y_test, y_pred)
    cfg.precision = precision_score(y_test, y_pred)
    cfg.recall = recall_score(y_test, y_pred)
    cfg.f1 = f1_score(y_test, y_pred)
    cfg.auc_roc = roc_auc_score(y_test, y_pred)
    
    # plot ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{cfg.path}/roc_curve.png')     
    plt.cla()
    
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.savefig(f'{cfg.path}/confusion_matrix.png')     
    plt.cla()

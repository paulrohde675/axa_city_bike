import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from evaluate_model import evaluate_model
import multiprocessing

def xg_boost(df: pd.DataFrame):
    """ This functions modifies the featurs accroding to the ML method """

    # get the number of logical cpu cores
    n_cores = multiprocessing.cpu_count()
    print(f'num avail cores = {n_cores}')

    scaler = StandardScaler()
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=df.columns)

    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init logistic Regression model
    model = GradientBoostingClassifier()
    
    # Define the hyperparameter grid
    param_grid = {
        "loss":["log_loss"],
        "learning_rate": [0.01, 0.1], # , 0.2
        "min_samples_split": np.linspace(0.1, 0.5, 2),
        "min_samples_leaf": np.linspace(0.1, 0.5, 2),
        "max_depth":[5], #3,5,8
        "max_features":["log2"], #,"sqrt"
        "criterion": ["friedman_mse"], #,  "mae"
        "subsample":[0.75], # [0.5, 0.75, 1.0],
        "n_estimators":[10]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(model, param_grid, refit=True, cv=5, n_jobs=-1, verbose=2, scoring='f1')
    
    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)    
    
    # Print the best hyperparameters found
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Get the best model found
    best_model = grid_search.best_estimator_

    evaluate_model(best_model, 'xgBoost', X_test, y_test)
    
    return


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/feature_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    xg_boost(df)
    
    
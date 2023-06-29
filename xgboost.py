import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
import numpy as np

def logistic_reg(df: pd.DataFrame):
    """ This functions modifies the featurs accroding to the ML method """

    scaler = StandardScaler()
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=df.columns)

    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init logistic Regression model
    logisticRegr = GradientBoostingClassifier()
    
    scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),'recall':make_scorer(recall_score)}
    
    # Define the hyperparameter grid
    param_grid = {
        "loss":["log_loss"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        "min_samples_split": np.linspace(0.1, 0.5, 12),
        "min_samples_leaf": np.linspace(0.1, 0.5, 12),
        "max_depth":[3,5,8],
        "max_features":["log2","sqrt"],
        "criterion": ["friedman_mse",  "mae"],
        "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        "n_estimators":[10]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(logisticRegr, param_grid, scoring=scoring, refit=False, cv=5, n_jobs=-1)
    
    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)    
    
    # Print the best hyperparameters found
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Get the best model found
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    accuracy = best_model.score(X_test, y_test)
    print("Accuracy on Test Set: {:.3f}".format(accuracy))
    
    return


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/feature_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    df = logistic_reg(df)
    
    
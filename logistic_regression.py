import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def logistic_reg(df: pd.DataFrame):
    """ This functions modifies the featurs accroding to the ML method """

    scaler = StandardScaler()
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=df.columns)

    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init logistic Regression model
    logisticRegr = LogisticRegression(solver='saga')
    
    # Define the hyperparameter grid
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.5, 1, 5],
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(logisticRegr, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    
    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)    
    
    # Print the best hyperparameters found
    print("Best Hyperparameters: ", grid_search.best_params_)

    # Get the best model found
    best_model = grid_search.best_estimator_

    # Prediction on test split
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)
    print("AUC-ROC:", auc_roc)
    
    return


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/feature_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    df = logistic_reg(df)
    
    
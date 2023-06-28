import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def logistic_reg(df: pd.DataFrame):
    """ This functions modifies the featurs accroding to the ML method """

    scaler = StandardScaler()
    data = scaler.fit_transform(df)
    df = pd.DataFrame(data, columns=df.columns)

    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    
    # Use score method to get accuracy of model
    score = logisticRegr.score(X_test, y_test)
    print(score)
    
    return


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/feature_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    df = logistic_reg(df)
    
    
import pandas as pd
import numpy as np
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import Config, imb_learn_options, model_options
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean data by removing outlier rows """
    
    # rename columns
    new_col_names = {
        'start station id': 'start_station_id',
        'start station name': 'start_station_name',
        'start station latitude': 'start_station_lat',
        'start station longitude': 'start_station_long',
        'end station id': 'end_station_id',
        'end station name': 'end_station_name',
        'end station latitude': 'end_station_lat',
        'end station longitude': 'end_station_long', 
        'birth year': 'birth_year',       
    }
    df = df.rename(columns=new_col_names)
    
    # drop columns
    df = df.drop(columns=['start_station_name', 'end_station_name'])

    # remove nan values
    df = df.dropna()

    # remove trips > 1 day and trips < 1 min
    df = df[df['tripduration'] <= 86400]
    df = df[df['tripduration'] >= 60]
    
    # remove stations with oddly off longitude and lattitude
    df = df[df['start_station_lat'] < 42]
    df = df[df['end_station_lat'] < 42]
    df = df[df['start_station_long'] < -73.7]
    df = df[df['end_station_long'] < -73.7]
    
    # remove people older than 100y
    df = df[df['birth_year'] > 1918]

    # fix types
    df['start_station_id'] = df['start_station_id'].astype(int)
    df['end_station_id'] = df['end_station_id'].astype(int)
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['stoptime'] = pd.to_datetime(df['stoptime'])


    df = df.reset_index(drop=True)    
    return df

def mod_feature(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """ This functions modifies the featurs accroding to the ML method """
    
    # convert datime to categorical feature
    df['day'] = df['starttime'].dt.dayofweek
    
    if cfg.model_type == model_options.LOGISTIC:
        df['time_elapsed'] = df['starttime'].dt.hour*3600 + df['starttime'].dt.minute*60 + df['starttime'].dt.second
        df['day_sin'] = (2 * np.pi * df['time_elapsed'] / 86400).apply(math.sin)
        df['day_cos'] = (2 * np.pi * df['time_elapsed'] / 86400).apply(math.cos)
        df = df.drop(columns=['time_elapsed'])
        
        df['month_sin'] = (2 * np.pi * df['starttime'].dt.month / 12).apply(math.sin)
        df['month_cos'] = (2 * np.pi * df['starttime'].dt.month / 12).apply(math.cos)
        
    elif cfg.model_type == model_options.GRAD_BOOST or cfg.model_type == model_options.RANDOM_FOREST:
        df['month'] = df['starttime'].dt.month
        df['time'] = df['starttime'].dt.hour
        df = pd.get_dummies(df, columns=['gender'], prefix='gender')
    
    # target to int
    df.loc[df['usertype'] == 'Customer', 'usertype'] = 0
    df.loc[df['usertype'] == 'Subscriber', 'usertype'] = 1
    df['usertype'] = df['usertype'].astype(int)
    
    # drop datetimes
    df = df.drop(columns=['starttime', 'stoptime'])

    return df


def test_train_split(df: pd.DataFrame, cfg: Config) -> list[pd.DataFrame]:
    """ Perfrom the test train split according to the parameter set in config """
    
    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.test_split, random_state=cfg.random_state)
    
    # save test data
    cfg.X_test = X_test
    cfg.y_test = y_test
    
    return [X_train, X_test, y_train, y_test]


def scale_data(cfg: Config, 
               X_train: pd.DataFrame, 
               y_train: pd.Series, 
               X_test: pd.DataFrame) -> list[pd.DataFrame | pd.Series]:
    """ Scale data by removing the mean and scaling to unit variance using sklearn StandardScaler """
    
    # store column names
    x_cols = X_train.columns
    y_col = y_train.name
    
    # perform standard scaling by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler = scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # rebuild pd.Dataframes
    X_train = pd.DataFrame(X_train, columns=x_cols)
    y_train = pd.Series(y_train, name=y_col)
    X_test = pd.DataFrame(X_test, columns=x_cols)
    
    # save scaler
    cfg.scaler = scaler
    
    return [X_train, y_train, X_test]


def handle_skewed_data(cfg: Config, X_train: pd.DataFrame, y_train: pd.DataFrame):
    """ Handle skewed by either applying over- or undersampling """
    
    if cfg.imb_mode == imb_learn_options.SMOTE:
        # Apply SMOTE to oversample the minority class
        smote = SMOTE(random_state=cfg.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    elif cfg.imb_mode == imb_learn_options.UNDERSAMPLING: 
        # Apply undersampling to the majority class
        undersampler = RandomUnderSampler(random_state=cfg.random_state)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
          
    return X_train_resampled, y_train_resampled
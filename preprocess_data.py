import pandas as pd
import numpy as np
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import Config, imb_learn_options, model_options

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
        
    elif cfg.model_type == model_options.XGBOOST:
        df['month'] = df['starttime'].dt.month
        df['time'] = df['starttime'].dt.hour
    
    # target to int
    df.loc[df['usertype'] == 'Customer', 'usertype'] = 0
    df.loc[df['usertype'] == 'Subscriber', 'usertype'] = 1
    df['usertype'] = df['usertype'].astype(int)
    
    
    # drop datetimes
    df = df.drop(columns=['starttime', 'stoptime'])

    return df

def handle_scewd_data(cfg: Config, X_train: pd.DataFrame, y_train: pd.DataFrame):
    if cfg.imb_mode == imb_learn_options.SMOTE:
        # Apply SMOTE to oversample the minority class
        smote = SMOTE(random_state=cfg.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    elif cfg.imb_mode == imb_learn_options.UNDERSAMPLING: 
        # Apply undersampling to the majority class
        undersampler = RandomUnderSampler(random_state=cfg.random_state)
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
          
    return X_train_resampled, y_train_resampled
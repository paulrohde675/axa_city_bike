import pandas as pd
from sklearn.utils import resample
import math
import numpy as np

def mod_feature(df: pd.DataFrame, periodic_feature: bool = False) -> pd.DataFrame:
    """ This functions modifies the featurs accroding to the ML method """
    
    # convert datime to categorical feature
    df['day'] = df['starttime'].dt.dayofweek
    
    if periodic_feature:
        df['time_elapsed'] = df['starttime'].dt.hour*3600 + df['starttime'].dt.minute*60 + df['starttime'].dt.second
        df['day_sin'] = (2 * np.pi * df['time_elapsed'] / 86400).apply(math.sin)
        df['day_cos'] = (2 * np.pi * df['time_elapsed'] / 86400).apply(math.cos)
        df = df.drop(columns=['time_elapsed'])
        
        df['month_sin'] = (2 * np.pi * df['starttime'].dt.month / 12).apply(math.sin)
        df['month_cos'] = (2 * np.pi * df['starttime'].dt.month / 12).apply(math.cos)
        
    else:
        df['month'] = df['starttime'].dt.month
        df['time'] = df['starttime'].dt.hour
    
    # target to int
    df.loc[df['usertype'] == 'Customer', 'usertype'] = 0
    df.loc[df['usertype'] == 'Subscriber', 'usertype'] = 1
    df['usertype'] = df['usertype'].astype(int)
    
    
    # drop datetimes
    df = df.drop(columns=['starttime', 'stoptime'])

    return df


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/balanced_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    df = mod_feature(df)
    print(f'n_rows after undersampling: {len(df.index)}')
    
    # save balanced data
    df.to_feather('data/intermediate/feature_data.feather')
    
import pandas as pd


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

    # convert float ids to int
    df['start_station_id'] = df['start_station_id'].astype(int)
    df['end_station_id'] = df['end_station_id'].astype(int)

    df = df.reset_index(drop=True)    
    return df


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/raw/2018-citibike-tripdata.feather')
    print(f'n_rows before cleaning: {len(df.index)}')
    
    # clean data
    df = clean_data(df)
    print(f'n_rows after cleaning: {len(df.index)}')
    
    # save cleaned data
    df.to_feather('data/intermediate/cleaned_data.feather')
    
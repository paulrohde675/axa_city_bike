import pandas as pd


# load data by looping over all 12 shards
def read_data_to_feather():
    df_list = []
    for i in range(12):
        filename = f'/workspaces/axa_city_bike/data/raw/2018{i+1:02d}-citibike-tripdata.csv'
        df_list.append(pd.read_csv(filename))
        print(filename)
    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df.to_feather('/workspaces/axa_city_bike/data/raw/2018-citibike-tripdata.feather')


def create_sample_data_set():
    df = pd.read_feather('data/raw/2018-citibike-tripdata.feather')
    df = df.sample(frac=0.001, random_state=42)
    df = df.reset_index(drop=True)
    df.to_feather('data/raw/2018-citibike-tripdata_xxs.feather')
    df.to_csv('data/raw/2018-citibike-tripdata_xxs.csv')


if __name__ == '__main__':
    create_sample_data_set()
import pandas as pd


# load data by looping over all 12 shards
df_list = []
for i in range(2):
    filename = f'/workspaces/axa_city_bike/data/raw/2018{i+1:02d}-citibike-tripdata.csv'
    df_list.append(pd.read_csv(filename))
    print(filename)
df = pd.concat(df_list)

print(df)
df.shape
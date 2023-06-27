import pandas as pd
from sklearn.utils import resample

def undersample_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Undersamples the dataset
    
    The dataset is heavily scewed towards subscriber which might
    cause problems while taining and interpreting the results.
    
    The most straight forward approch is undersampling by randomling 
    reducing the majority class.
    """
    
    # Split the DataFrame into majority and minority classes
    df_majority = df[df['usertype'] == 'Subscriber']
    df_minority = df[df['usertype'] == 'Customer']

    # Undersample the majority class
    df_majority_undersampled: pd.DataFrame = resample(df_majority, 
                                                      replace=False, 
                                                      n_samples=len(df_minority), 
                                                      random_state=42)

    # Combine the undersampled majority class with the minority class
    df_undersampled = pd.concat([df_majority_undersampled, df_minority])

    # Shuffle the combined DataFrame to randomize the order of the samples
    df_undersampled = df_undersampled.sample(frac=1, random_state=42)
    df_undersampled = df_undersampled.reset_index(drop= True)

    return df_undersampled


if __name__ == '__main__':
    # load data
    df = pd.read_feather('data/intermediate/cleaned_data.feather')
    print(f'n_rows before undersampling: {len(df.index)}')
    
    # clean data
    df = undersample_data(df)
    print(f'n_rows after undersampling: {len(df.index)}')
    
    # save balanced data
    df.to_feather('data/intermediate/balanced_data.feather')
    
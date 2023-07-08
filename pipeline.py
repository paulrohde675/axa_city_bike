from preprocess_data import handle_scewd_data, clean_data, mod_feature
from evaluate_model import evaluate_model
from train_model import init_model, train_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config, model_options, imb_learn_options
import pandas as pd
import logging

def pipeline(cfg: Config):
    """ Main pipe"""
    logging.info('Run Pipeline')
    
    # load data
    df = pd.read_feather('data/raw/2018-citibike-tripdata_sample.feather')
    logging.info('Data loaded')
    
    # clean data
    df = clean_data(df)
    logging.info('Data cleaned')
    
    # argument features
    df = mod_feature(df)
    logging.info('Features argumented')
    
    # test train split
    y = df['usertype']
    X = df.drop(columns=['usertype'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info('Data splitted')
    
    # scale data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train, y_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    logging.info('Data scaled')
    
    # user over- or undersampling for scewed data
    handle_scewd_data(cfg, X_train, y_train)
    logging.info('Data corrected for scewnes')
    
    # init the model spcified in config
    model = init_model(cfg)
    logging.info('Model initialized')
    
    # train the model-grid and pick the best one
    model = train_model(model, cfg, X_train, y_train)
    logging.info('Model trained')
    
    # evalute the best trained model
    evaluate_model(model, cfg, X_test, y_test)
    

if __name__ == '__main__':

    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO)
    
    cfg = Config(model_options.XGBOOST, imb_mode=imb_learn_options.UNDERSAMPLING)
    pipeline(cfg)
import logging
import os

import pandas as pd

from config import Config
from config import imb_learn_options
from config import model_options
from evaluate_model import evaluate_model
from feature_importance import compute_perm_feature_importance
from io_functions import save_run_to_pickle
from preprocess_data import clean_data
from preprocess_data import handle_skewed_data
from preprocess_data import mod_feature
from preprocess_data import scale_data
from preprocess_data import test_train_split
from train_model import init_model
from train_model import train_model


def pipeline(cfg: Config):
    """Main pipe"""
    logging.info(f"Run Experimnet {cfg.run_name}")

    # create folder to store results
    if not os.path.exists(cfg.path):
        os.makedirs(cfg.path)

    # load data
    df = pd.read_feather("data/raw/2018-citibike-tripdata_02.feather")
    logging.info("Data loaded")

    # clean data
    df = clean_data(df)
    logging.info("Data cleaned")

    # argument features
    df = mod_feature(df, cfg)
    logging.info("Features argumented")

    # test train split
    X_train, X_test, y_train, y_test = test_train_split(df, cfg)
    logging.info("Data splitted")

    # scale data
    X_train, y_train, X_test = scale_data(cfg, X_train, y_train, X_test)
    logging.info("Data scaled")

    # user over- or undersampling for scewed data
    handle_skewed_data(cfg, X_train, y_train)
    logging.info("Data corrected for scewnes")

    # init the model spcified in config
    model = init_model(cfg)
    logging.info("Model initialized")

    # train the model-grid and pick the best one
    model = train_model(model, cfg, X_train, y_train)
    logging.info("Model trained")

    # evalute the best trained model
    evaluate_model(model, cfg, X_test, y_test)
    logging.info("Model evaluated")

    # compute the feeature importance
    compute_perm_feature_importance(model, cfg, X_test, y_test)
    logging.info("Feature evaluated")

    # save experiment results to file
    save_run_to_pickle(cfg)
    logging.info("Experiment saved to file")
    logging.info(f"Experimnet {cfg.run_name} finished")


if __name__ == "__main__":
    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO)

    cfg = Config(
        "svm_01",
        model_options.SVM,
        imb_mode=imb_learn_options.UNDERSAMPLING,
        scoring="f1",
        cv_folds=5,
    )
    pipeline(cfg)

import numpy as np
from enum import Enum

imb_learn_options = Enum('smote', 'undersampling')
model_options = Enum('logistic', 'xgboost')

class imb_learn_options(Enum):
    SMOTE = 'smote'
    UNDERSAMPLING = 'undersampling'

class model_options(Enum):
    LOGISTIC = 'logistic'
    XGBOOST = 'xgboost'
    RANDOM_FOREST = 'random_forest'

class Config:
    
    def __init__(self, 
                 model_type: model_options,
                 scoring = 'f1',
                 test_split = 0.3,
                 cv_folds = 5,
                 imb_mode: imb_learn_options = imb_learn_options.UNDERSAMPLING) -> None:
        self.random_state = 42
        self.test_split = test_split
        self.model_type: model_options = model_type
        self.scoring = scoring
        self.imb_mode: imb_learn_options = imb_mode
        self.cv_folds = cv_folds
        
        # Define the hyperparameter grid
        if model_type == model_options.XGBOOST:
        #    self.hyperparam_grid = {
        #         "loss":["log_loss"],
        #         "learning_rate": [0.01], #0.01,  , 0.2
        #         "min_samples_split": np.linspace(0.1, 0.5, 2),
        #         "min_samples_leaf": np.linspace(0.1, 0.5, 2),
        #         "max_depth":[3, 5], #3,5,8
        #         "max_features":["sqrt"], #,"sqrt"
        #         "criterion": ["friedman_mse"], #,  "mae"
        #         "subsample":[0.5, 0.75, 1.0], # [0.5, 0.75, 1.0],
        #         "n_estimators":[10]
        #     }  
            self.hyperparam_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01]
            }
           
        elif model_type == model_options.LOGISTIC:
            self.hyperparam_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.5, 1, 5],
            } 
        elif model_type == model_options.RANDOM_FOREST:
            self.hyperparam_grid = {
                'bootstrap': [False], #, False
                'max_depth': [10, 50],  #, 20, 30, 40, 50, 100, 60, 70, 80, 90, , None
                'max_features': ['sqrt'], #'auto', 
                'min_samples_leaf': [1, 2], #1, 2, 4
                'min_samples_split': [2, 5],#2, 5, 10
                'n_estimators': [2, 5, 10] #200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000
            }
    

    
    

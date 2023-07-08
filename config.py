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

class Config:
    
    def __init__(self, 
                 model_type: model_options,
                 scoring = 'f1',
                 imb_mode: imb_learn_options = imb_learn_options.UNDERSAMPLING) -> None:
        self.random_state = 42
        self.model_type: model_options = model_type
        self.scoring = scoring
        self.imb_mode: imb_learn_options = imb_mode
        
        # Define the hyperparameter grid
        if model_type == model_options.XGBOOST:
           self.hyperparam_grid = {
                "loss":["log_loss"],
                "learning_rate": [0.01, 0.1], # , 0.2
                "min_samples_split": np.linspace(0.1, 0.5, 2),
                "min_samples_leaf": np.linspace(0.1, 0.5, 2),
                "max_depth":[5], #3,5,8
                "max_features":["log2"], #,"sqrt"
                "criterion": ["friedman_mse"], #,  "mae"
                "subsample":[0.75], # [0.5, 0.75, 1.0],
                "n_estimators":[10]
            }  
           
        elif model_type == model_options.LOGISTIC:
            self.hyperparam_grid = {
                'penalty': ['l1', 'l2'],
                'C': [0.5, 1, 5],
            } 
    
    

    
    

import pickle
import streamlit as st
from config import Config

@st.cache_resource
def load_experiment_from_file(model_name: str) -> Config:
    """ Load an experiment including the trained model, parameters, test data, scaler and evaluation results from file """
    
    # get session state
    state = st.session_state
    
    path: str = state.models[model_name]
    print(f'load: {path}')
    
    with open(f'{path}/results.pickle', "rb") as input_file:
        cfg: Config = pickle.load(input_file)
        
    print(f'loaded: {path}')
        
    return cfg

def side_bar() -> None:
    
    # get session state
    state = st.session_state
    
    # add global parameter to session state
    if "models" not in state:
        state.models: dict[str, str] = {
            'Logistic Regression': 'data/final/log_regression_01',
            'Gradient Boost': 'data/final/grad_boost_01',
            'Random Forest': 'data/final/rnd_forest_01',
            'SVM': 'data/final/svm_01',
            'Naive Bayes': 'data/final/naive_bayes_01',
            'Neural Network': 'data/final/neural_network_01',
        }    
    
    # Create a sidebar for model selection
    st.sidebar.title("Model Selection")   
    selected_model = st.sidebar.selectbox("Select a model", state.models)
 
    state.cfg = load_experiment_from_file(selected_model)
    
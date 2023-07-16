import pickle
import sys
import streamlit as st
from config import Config
from sklearn.base import BaseEstimator
from dataclasses import dataclass


@dataclass
class DashboadData:
    page: str
    cfg: Config

def main():
    print('Rerender---------------')

    state = st.session_state

    if "page" not in state:
        state.page = 'Introduction'

    # Create a sidebar for model selection
    st.sidebar.title("Model Selection")   
    selected_model = st.sidebar.selectbox("Select a model", models)
    state.cfg = load_experiment_from_file(selected_model)
    
    st.sidebar.title("Outline")
    pages = {
        "Introduction": page1,
        "Data Exploration": page2,
        "Preprocessing": page3,
        "Model Evaluation": page4,
        "Conclusion": page5,
        "Outlook": page6,
    }    

    selected_page = st.sidebar.button("Introduction", use_container_width=True)
    if selected_page:
        state.page = "Introduction"

    selected_page = st.sidebar.button("Data Exploration", use_container_width=True)
    if selected_page:
        state.page = "Data Exploration"

    selected_page = st.sidebar.button("Preprocessing", use_container_width=True)
    if selected_page:
        state.page = "Preprocessing"

    selected_page = st.sidebar.button("Model Evaluation", use_container_width=True)
    if selected_page:
        state.page = "Model Evaluation"

    selected_page = st.sidebar.button("Conclusion", use_container_width=True)
    if selected_page:
        state.page = "Conclusion"
        
    selected_page = st.sidebar.button("Outlook", use_container_width=True)
    if selected_page:
        state.page = "Outlook"                    
    
    pages[state.page]()


def page1():
    st.title("Introduction")

    # Split the page into two columns
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:
        st.header("Column 1")
        st.write("This is the content of column 1.")

    # Content for the second column
    with col2:
        st.image('assets/shared-bike-g36484585c_1280.jpg', use_column_width=True)

        
    # Display the image

def page2():
    cfg = st.session_state.cfg
    st.title("Data Exploration")
    
    st.text(f'Model selected {cfg.run_name}')
    st.text(cfg.model_report)

def page3():
    st.title("Preprocessing")

def page4():
    st.title("Model Evaluation")

def page5():
    st.title("Conclusion")
    
def page6():
    st.title("Outlook")    
    

models: dict[str, str] = {
    'Gradient Boost': 'data/final/grad_boost_01',
    'Random Forest': 'data/final/rnd_forest_01',
}

def load_experiment_from_file(model_name: str) -> Config:
    """ Load an experiment including the trained model, parameters, test data, scaler and evaluation results from file """
    path: str = models[model_name]
    print(f'load: {path}')
    
    with open(f'{path}/results.pickle', "rb") as input_file:
        cfg: Config = pickle.load(input_file)
        
    print(f'loaded: {path}')
        
    return cfg

if __name__ == "__main__":
    main()
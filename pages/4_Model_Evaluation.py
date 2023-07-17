import streamlit as st
from config import Config
from dashboard_sidebar import side_bar

def page_model_evaluation():
    cfg: Config = st.session_state.cfg
    st.title("Model Evaluation")
    
    # render side_bar
    side_bar()

    st.text(f'Model selected {cfg.run_name}')
    st.text(cfg.model_report)
    
if __name__ == '__main__':
    page_model_evaluation()
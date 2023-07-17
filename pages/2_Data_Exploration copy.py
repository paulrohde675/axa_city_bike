import streamlit as st
from dashboard_sidebar import side_bar

def page_data_exploration():
    st.title("Data Exploration")
    
    # render side_bar
    side_bar()

    
if __name__ == '__main__':
    page_data_exploration()
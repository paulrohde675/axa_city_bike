import pickle
import sys
import streamlit as st
from config import Config
from sklearn.base import BaseEstimator
from dataclasses import dataclass
from dashboard_sidebar import side_bar

@dataclass
class DashboadData:
    page: str
    cfg: Config



def main():
    st.title("Introduction")

    # get session state
    state = st.session_state
    
    # add global parameter to session state
    if "models" not in state:
        state.models: dict[str, str] = {
            'Gradient Boost': 'data/final/grad_boost_01',
            'Random Forest': 'data/final/rnd_forest_01',
        }

    # render side_bar
    side_bar()

    # Split the page into two columns
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:
        st.header("Column 1")
        st.write("This is the content of column 1.")

    # Content for the second column
    with col2:
        st.image('assets/shared-bike-g36484585c_1280.jpg', use_column_width=True)


if __name__ == "__main__":
    main()
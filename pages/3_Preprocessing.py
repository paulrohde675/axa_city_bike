import streamlit as st
from dashboard_sidebar import side_bar
from config import Config

def page_preprocessing():
    st.title("Preprocessing")
    state = st.session_state
    cfg: Config = state.cfg

    # render side_bar
    side_bar()

    # clean data
    st.subheader('Clean data')
    st.markdown("- rename columns")
    st.markdown("- remove missing values")
    st.markdown("- drop **station names**")
    st.markdown("- drop **bikeid**")
    st.markdown("- remove **birth year** < 1918")
    st.markdown("- remove **tripduration** > 1 day")
    st.markdown("- remove **tripduration** < 1 min")
    st.markdown("- remove **long/latitude** far off")
    st.markdown('#')
    
    # feature engineering
    st.subheader('Feature engineering')    
    st.markdown("- add **day of the week**")
    st.markdown("- add **month**")
    st.markdown("    - cyclic for non tree based model")
    st.markdown("    - int for tree based model")
    st.markdown("- add **daytime**")
    st.markdown("    - cyclic for non tree based model")
    st.markdown("    - int for tree based model")
    st.markdown("- one hot encode **gender** for tree based model")
        
    st.markdown("- convert **usertype to int**")
    
    st.dataframe(cfg.X_test.head())
    
if __name__ == '__main__':
    page_preprocessing()
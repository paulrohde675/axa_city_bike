import streamlit as st
import pandas as pd
from dashboard_sidebar import side_bar
from config import Config


def page_preprocessing():
    """This page shows all preprocessing steps applied to the data"""
    st.title("Preprocessing")
    st.markdown('#')
    st.write("""Data preprocessing is essential for model success. The first step is cleaning data 
             to remove inconsistencies and outliers, followed by feature engineering to create 
             meaningful attributes like rider age and time of day. We then partitioned the data into 
             training and test sets for model validation. Lastly, we addressed class imbalance through 
             techniques like over-sampling and under-sampling to avoid model bias.
             """)
    st.markdown('#')


    # render side_bar
    side_bar()

    state = st.session_state
    cfg: Config = state.cfg

    # clean data
    st.subheader("Clean data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- rename columns")
        st.markdown("- remove missing values")
        st.markdown("- drop **station names**")
        st.markdown("- drop **bikeid**")

    with col2:
        st.markdown("- remove **birth year** < 1918")
        st.markdown("- remove **tripduration** > 1 day")
        st.markdown("- remove **tripduration** < 1 min")
        st.markdown("- remove **long/latitude** far off")
    st.markdown("#")

    # feature engineering
    st.subheader("Feature engineering")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        - add **month** 
            - cyclic for non tree based model
            - int for tree based model"""
        )
        st.markdown(
            """
        - add **daytime**
            - cyclic for non tree based model
            - int for tree based model"""
        )

    with col2:
        st.markdown("- add **day of the week**")
        st.markdown("- one hot encode **gender** for tree based model")
        st.markdown("- convert **usertype** to int")
        st.markdown("- convert **station id** to int")

    st.markdown(
        """
    - further steps:
        - **distance** between start/stop using an API
        - **weather** at the starttime using an API"""
    )
    st.markdown("#")

    # show example data
    st.dataframe(cfg.X_test.head())

    # feature engineering
    st.subheader("Test/train split")
    st.markdown(
        """
    - apply test/train split 
        - *f_split* = 0.3"""
    )
    st.markdown("#")

    # scale data
    st.subheader("Scale data")
    st.markdown(
        """
    - apply standard scaler 
        - remove mean
        - scale to unit variance"""
    )

    # show example data
    scaled_data = pd.DataFrame(
        cfg.scaler.transform(cfg.X_test.head()), columns=cfg.X_test.columns
    )
    st.dataframe(scaled_data)
    st.markdown("#")

    # imbalanced data
    st.subheader("Sample data")
    st.markdown(
        """
    - **undersample** data due to imbalance
        - use **undersampling** since the dataset is large
    - alternatives:
        - **oversampling** using smote
        - adopting model weights"""
    )


if __name__ == "__main__":
    page_preprocessing()

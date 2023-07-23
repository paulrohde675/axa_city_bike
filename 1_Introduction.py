import streamlit as st
from dashboard_sidebar import side_bar


def main():
    st.title("NYC Bike User Classification Dashboard")

    # get session state
    state = st.session_state

    # render side_bar
    if "models" in state:
        side_bar()

    # Split the page into two columns
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:
        st.header("Abstract")
        st.write(
            "Welcome to our NYC Bike User Classification Dashboard! This platform showcases the comprehensive process of classifying user types from the NYC Bike dataset of 2018. This procedure involves several stages including data exploration, data cleaning, feature engineering, model evaluation, live model execution, and an outlook on future improvements."
        )

    # Content for the second column
    with col2:
        st.image("assets/shared-bike-g36484585c_1280.jpg", use_column_width=True)


if __name__ == "__main__":
    main()

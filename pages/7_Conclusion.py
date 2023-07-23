import streamlit as st

from dashboard_sidebar import side_bar


def page_conclusion():
    st.title("Conclusion")

    st.write(
        """This conclusion offers a reflection on the accomplishments and 
             key learnings from the exploration of the NYC Bike dataset. The 
             journey involved the creation of robust models to classify user 
             types, incorporating numerous steps of a typical machine learning 
             project. The following summarizes these achievements and outlines 
             potential future developments."""
    )
    st.markdown("#")
    st.markdown(
        """
        - successfully built high-performance models to classify user types in the NYC Bike dataset.
        - Gradient Boosting and Neural Networks proved to be the best models.
        - through preprocessing, including addressing class imbalance, we ensured the robustness of our models.
        - identified gender and birth year as crucial features in user classification
            - especially given their often 'unknown' or default status among short-term subscribers.
        - live model execution showcased our models' high performance and provided an interactive experience.
        - future work includes refining our models and incorporating additional features for even better performance.
        """
    )

    # render side_bar
    side_bar()


if __name__ == "__main__":
    page_conclusion()

import streamlit as st
from dashboard_sidebar import side_bar


def page_outlook():
    st.title("Outlook")
    st.markdown("#")

    # render side_bar
    side_bar()

    # Split the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Possible Next Steps")
        st.markdown(
            """
            - feature engineering
                - create distance feature
                - create velocity feature
                - create weather feature
            - retrain model on whole dataset
            - deploy model 
            """
        )
    with col2:
        st.image("assets/DALLE_Outlook.png")
    st.markdown("#")

    st.subheader("Cooperation between NYC Bike and an insurace company")
    st.markdown(
        """
        - movement data
            - people riding bikes regularly might be more healthy
            - people riding bikes regularly drive less car
            - people riding bikes regularly might have higher accident risk
                - especially if they drive fast
            - average velocity might correlate with fintess and health
            - start / stop locations might correspond to the customers home / workplace
                - might be interessting to double check the insurance company data
        - fraut
            - the insurance compy might inform NYC Bike about fraut risks
        - customer benefits
            - customer is inshured while riding if he is a customer of the insurance company
            - free rides if the customer is also customer of the insurance company
        - insurance
            - insuranc of the bikes
        """
    )


if __name__ == "__main__":
    page_outlook()

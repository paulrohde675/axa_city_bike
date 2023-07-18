import pandas as pd
import streamlit as st
from config import Config
from config import model_options
from dashboard_sidebar import side_bar

def page_model_evaluation():
    """ On this page, model evaluations are comapred """
    
    # render side_bar
    side_bar()
    
    # get session state and cfg
    state = st.session_state
    cfg: Config = state.cfg
    
    st.title("Model Evaluation")
    st.markdown('#')
    st.markdown('#')
    
    st.subheader('Statistics')
    report_dict:dict = cfg.model_report_dict
    if 'accuracy' in report_dict:
        del report_dict['accuracy']
    
    # Convert the dictionary into a DataFrame
    report_df = pd.DataFrame(report_dict).T
    report_df = report_df.rename(index={'0': 'Customer', '1': 'Subscriber'})

    # Change the index name to empty
    report_df.index.name = ''
    report_df = report_df.round(2)
    st.dataframe(report_df, width=1000)
    st.markdown('- **subscriber** are more accurately classified')
    st.markdown('- **customer** recall is rather low')
    st.markdown('- **subscriber** f1 score is much higher')
    st.markdown('#')
    
    # confusion matrix
    st.subheader('Confusion matrix')
    st.pyplot(cfg.plt_confusion_matrix)
    st.markdown('Negative (0) = Customer')
    st.markdown('Positive (1) = Subscriber')
    st.markdown("""
        - the models often misclassify customers as subscriber
            - these might be cases where customers have other than **unkown** gender
        """)
    st.markdown('#')
    
    # roc curve
    st.subheader('Roc curve')
    st.pyplot(cfg.plt_roc)
    
if __name__ == '__main__':
    page_model_evaluation()
    
import random
import math
import numpy as np
import pandas as pd
import streamlit as st
from config import Config
from config import model_options
from dashboard_sidebar import side_bar

def predict_from_rnd(event_idx: int) -> None:
        """ make a prediction from a random index """
    
        # get session state and cfg
        cfg: Config = st.session_state.cfg
        state = st.session_state
        
        event = cfg.X_test.iloc[[event_idx]]
        state.event_df = event.copy()
        
        # scale data
        scaled_event = cfg.scaler.transform(event)
        scaled_event = pd.DataFrame(scaled_event, columns=event.columns)
        
        # perform prediction
        pred_val = cfg.model.predict(scaled_event)[0]
        true_val = cfg.y_test.iloc[[event_idx]].values[0]

        # check if predicted correctly
        if pred_val != true_val:
            state.outcome_bool = False
        else:
            state.outcome_bool = True

        print('Prediction results:')
        print(f'pred: {pred_val} | true: {true_val}')

        # convert prediction to string (dict would be better!)
        if pred_val == 1:
            state.pred_val_str = 'Subscriber'
        elif pred_val == 0:
            state.pred_val_str = 'Customer'
        else:
            state.pred_val_str = 'Something Weird'

        if true_val == 1:
            state.true_val_str = 'Subscriber'
        elif true_val == 0:
            state.true_val_str = 'Customer'
        else:
            state.true_val_str = 'Something Weird'

def page_model_evaluation():
    cfg: Config = st.session_state.cfg
    st.title("Model Evaluation")
    st.markdown('#')
    st.markdown('#')
    
    # get session state
    state = st.session_state
    
    # render side_bar
    side_bar()
    
    st.subheader('Predict a random sample')

    # create an initial prediction
    btn = st.button('     Random sample     ')
    if 'event_idx' not in state:
        state.event_idx: int = 42
        predict_from_rnd(state.event_idx)

    # predict a random sample on btn click
    if btn:
        # choose a random sample
        n_events = len(cfg.X_test.index)
        state.event_idx = random.randint(0, n_events-1)
        
        # perfrom prediction
        predict_from_rnd(state.event_idx)
    
    # Show the event
    st.subheader('Selected sample')
    st.dataframe(state.event_df)
    st.markdown('#')

    # Split the page into two columns
    col1, col2 = st.columns(2)

    # Show the prediction results
    with col1:
        st.subheader('Prediction')
        if state.outcome_bool:
            st.success(state.pred_val_str)
        else:
            st.error(state.pred_val_str)

    with col2:
        st.subheader('True value')
        if state.outcome_bool:
            st.success(state.true_val_str)
        else:
            st.error(state.true_val_str)
    st.markdown('#')
        
    st.subheader('Predict on custom events')
    
    container = st.container()
    
    with st.expander("Create an event"):
        gender = st.slider('gender', min_value=1, value=0, max_value=2, step=1)
        birth_year = st.slider('birth_year', min_value=1918, value=1991, max_value=2005, step=1)
        time = st.slider('time', min_value=0, value=14, max_value=23, step=1)
        day = st.slider('day', min_value=0, value=1, max_value=6, step=1)
        month = st.slider('month', min_value=0, value=7, max_value=11, step=1)
        tripduration = st.slider('tripduration', min_value=60, value=900, max_value=86400, step=10)
        start_station_id = st.slider('start_station_id', min_value=0, value=3100, max_value=4000, step=10)
        end_station_id = st.slider('end_station_id', min_value=0, value=442, max_value=4000, step=10)
        start_station_lat = st.slider('start_station_lat', min_value=40.0, value=40.7, max_value=45.0, step=0.1)
        end_station_lat = st.slider('end_station_lat', min_value=40.0, value=40.7, max_value=45.0, step=0.1)
        start_station_long = st.slider('start_station_long', min_value=-75.0, value=-74.0, max_value=-73.0, step=0.1)
        end_station_long = st.slider('end_station_long', min_value=-75.0, value=-73.9, max_value=-73.0, step=0.1)
    
    event = pd.DataFrame({
        'tripduration': tripduration,
        'start_station_id': start_station_id,
        'start_station_lat': start_station_lat,
        'start_station_long': start_station_long,
        'end_station_id': end_station_id,
        'end_station_lat': end_station_lat,
        'end_station_long': end_station_long,
        'birth_year': birth_year,
        'gender': gender,
        'time': time,
        'day': day,
        'month': month,
    }, index=[0])
    
    # account for model type
    if cfg.model_type.value == model_options.LOGISTIC.value:
        event['time_elapsed'] = event['time']*3600
        event['time_sin'] = (2 * np.pi * event['time_elapsed'] / 86400).apply(math.sin)
        event['time_cos'] = (2 * np.pi * event['time_elapsed'] / 86400).apply(math.cos)
        
        event['month_sin'] = (2 * np.pi * event['month'] / 12).apply(math.sin)
        event['month_cos'] = (2 * np.pi * event['month'] / 12).apply(math.cos)
        event = event.drop(columns=['time_elapsed'])       
        event = event.drop(columns=['time'])       
        event = event.drop(columns=['month'])    

    # scale event
    scaled_event = cfg.scaler.transform(event)
    scaled_event = pd.DataFrame(scaled_event, columns=event.columns)
    
    # perform prediction
    pred_val = cfg.model.predict(scaled_event)[0]

    # convert prediction to string (dict would be better!)
    if pred_val == 1:
        pred_val_str = 'Subscriber'
    elif pred_val == 0:
        pred_val_str = 'Customer'
    else:
        pred_val_str = 'Something Weird'
        
    container.success(pred_val_str)
    
        
if __name__ == '__main__':
    page_model_evaluation()
# axa_city_bike

## Init project

### Devcontainer
Run this project as a devcontainer using Docker
The Docker image used is:
"mcr.microsoft.com/devcontainers/python:1-3.11-bullseye"

For more information on devcontainer see:
https://code.visualstudio.com/docs/devcontainers/containers

### Pip
If devcontainers are not working for you, a simple 
```
pip insall -r requirements.txt
```

should work.

## Train a model
- To train a model go to pipeline.py and and adjust the parameter in the __main__ section
- To adjust the hyperparameter grid, go to config.py
- To add a new model, add a new model to model_options in config.py and add a hyperparamter grid for this model. Check out preprocess_data aswell and your model to the feature engineering part. To display your model in the dashboard, add it to state.models in the dashboard_sidebar.py.

## Dashboard
To run the Dashboard type
```
streamlit run 1_Introduction.py
```

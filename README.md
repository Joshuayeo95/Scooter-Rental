# Scooter-Rental

In this project, the goal was to predict the number of scooter rental users to help __demand forecasting__. As this was a regression problem, we trained and evaluated the performance of the following models:

__Linear Models__
* Linear Regression
* Lasso Regression
* Ridge Regression

__Ensemble Models__
* Random Forest Regression
* Gradient Boosting Regression

These models were chosen as they allow us to compare the performance between Linear Regression, Bootstrap Aggregation (Random Forest) and Boosting algorithms. Given that the dataset is not very large, we did not choose to implement deep learning models.

## Data Preparation and Model Selection Process

The dataset was split into training (__80%__) and holdout (__20%__) datasets.

The individual models were trained using __5-Fold cross validation__ on the training data before being evaluated on the holdout data.

The evaluation metric used to measure model performance is the __Root Mean Squared Error__ (RMSE). We chose RMSE as it penalises larger error as well as being in the same units of our target variable which makes it more intepretable.

The training results and hyper-parameter tuning can be viewed in the __`modelling.ipynb`__ Jupyter Notebook.

## Selecting Model & Hyper Parameter Tuning 

To change the model being trained, please go to __`mlp\config.py`__ and change the __`model_type`__.
__Available options are: 'linear', 'lasso', 'ridge', 'rforest' and 'gboost'.__

Hyper parameters for the models have been set to the optimal parameters found during tuning and each hyper paramter for their respective models have been saved in a dictionary. However, if you wish to change the parameters, just edit the value within the key-value for the repective model hyper parameter dictionary.

## Pipeline

Firstly, the data is ingested and cleaned using __`preprocessing.py`__. This will create __`training_dataset.pickle`__ which is the dataset that has been cleaned and ready to use for model training.

Next, we will use the functions in __`build.py`__ to create our model. 

The model is then fit with the clean data we have obtained from __`preprocessing.py`__.

Finally, the model is trained and outputs the evaluation metrics onto the command line. 

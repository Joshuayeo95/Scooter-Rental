# Scooter-Rental

In this project, the goal was to predict the number of scooter rental users to help demand forecasting. As this was a regression problem, we trained and evaluated the performance of the following models:

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

The individual models were trained using 5-Fold cross validation on the training data before being evaluated on the holdout data.

The evaluation metric used to measure model performance is the Root Mean Squared Error (RMSE). We chose RMSE as it penalises larger error as well as being in the same units of our target variable which makes it more intepretable.

The training results and hyper-parameter tuning can be viewed in the 'modelling.ipynb' Jupyter Notebook.

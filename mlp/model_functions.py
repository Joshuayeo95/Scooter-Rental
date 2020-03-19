''' Machine Learning Functions '''

# Standard Libraries
from scipy import stats
import pandas as pd
import numpy as np
import pickle
import time

# Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

import visualisation as viz # user defined functions

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



def adj_r_square(X, model_r2):
    '''
    Calculates the adjusted R-Square of the model.
    
    :params:
        X (array) :
            Array of independent variables' values.
        model_r2 (float): 
            R-Square of the model fitted on the same array x as the independent variables.
        
    :returns:
        adj_r2 (float): 
            Adjusted R-Square of the model fitted. 
        
    '''
    n = len(X) # Number of observations #
    k = X[0].size # Number of parameters # 
    adj_r2 = 1 - (1-model_r2) * ((n-1)/(n-k-1))
    
    return adj_r2



def get_rmse(model, X, y):
    '''
    Function to generate the RMSE from fitted model. 
    '''

    resid = y - model.predict(X)
    rmse = np.sqrt(np.mean(resid**2))

    return rmse



def get_scores(model, X, y):
    '''
    Function to obtain the R Squared and Adjusted R Squared from the fitted model.
    '''

    score = model.score(X, y)
    adj_score = adj_r_square(X, score)

    return score, adj_score



def alpha_plots(alphas, X_train, y_train, X_valid, y_valid, model_type, fig_size=(6,6), return_series=False):
    '''
    Function that plots the RMSEs of the training and validation datasets across different alpha values for Ridge Regression.
    
    :params:
        alphas (array) : 
            Vector of different alpha values to try.
        model_type (str) :
            Model to plot the RMSEs values. Can be 'ridge' or 'lasso'.
        return_series (bool) : 
            Default is False. If True, returns the training and validation series used to plot the graphs. 
        
    '''

    rmse_training = []
    rmse_validation = []

    for alpha in alphas:

        if model_type == 'ridge':
            pipe = Pipeline(steps=[
                    ('Standardise', StandardScaler()),
                    ('Ridge', Ridge(alpha=alpha))
                ]
            )

        if model_type == 'lasso':
            pipe = Pipeline(steps=[
                    ('Standardise', StandardScaler()),
                    ('Lasso', Lasso(alpha=alpha))
                ]
            )

        model = pipe.fit(X_train, y_train)
        
        # Calls function to obtain RMSE for training and validation
        rmse_training.append(get_rmse(model, X_train, y_train))
        rmse_validation.append(get_rmse(model, X_valid, y_valid))

    rmse_training = pd.Series(rmse_training, index=alphas)
    rmse_validation = pd.Series(rmse_validation, index=alphas)
    
    # Plotting the RMSE Scores 
    viz.hyper_param_plot(rmse_training, rmse_validation, fig_size=fig_size)
    
    if return_series:
        return rmse_training, rmse_validation



def kfold_cv(model, X, y, cv=5):
    '''
    Function to perform K-Fold cross validation and returns the RMSEs, R Squared and Adjusted R Sqaures.
    '''

    training_rmse = []
    validation_rmse = []
    
    training_r2 = []
    validation_r2 = []
    
    training_adj_r2 = []
    validation_adj_r2 = []

    kf = KFold(n_splits=cv, shuffle=True)

    for train_index, valid_index in kf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]
        
        model.fit(X_train, y_train)
        
        # Calls function to obtain RMSE for training and validation
        training_rmse.append(get_rmse(model, X_train, y_train))
        validation_rmse.append(get_rmse(model, X_valid, y_valid))
        
        # Calls function to obtain training and validation scores 
        train_r2_score, train_adj_r2_score = get_scores(model, X_train, y_train)
        valid_r2_score, valid_adj_r2_score = get_scores(model, X_valid, y_valid)

        # Saving R Squared
        training_r2.append(train_r2_score)
        validation_r2.append(valid_r2_score)
        
        # Saving Adjusted R Squared
        training_adj_r2.append(train_adj_r2_score)
        validation_adj_r2.append(valid_adj_r2_score)
    
    RMSE = {'Training' : training_rmse,
            'Validation' : validation_rmse}
    
    R_Squared = {'Training' : training_r2,
                 'Validation' : validation_r2}
    
    Adj_R_Squared = {'Training' : training_adj_r2,
                     'Validation' : validation_adj_r2}
    
    return RMSE, R_Squared, Adj_R_Squared



def kfold_cv_glm(X, y, optimal_alpha, model_type, cv=5):
    '''
    Function that performs cross validation and returns the RMSE of the training and validation datasets.

    :params:
        optimal_alpha (int) : 
            Optimal alpha value determined during tuning. 
        model_type (str) :
            Linear Model to be fitted. Options include 'linear', 'lasso' or 'ridge'.
    
    :returns:
        RMSE (dict) : 
            Dictionary of the training and validation RMSEs scores obtained during cross validation.
        R_Squared (dict)
            Dictionary of the training and validation R-Squared scores obtained during cross validation.
        Adj_R_Squared (dict)
            Dictionary of the training and validation Adjusted R-Squared scores obtained during cross validation.
    '''
    
    # Converting data into ndarrays
    X = np.array(X)
    y = np.array(y)

    if model_type == 'linear':
        model = LinearRegression(fit_intercept=True, n_jobs=-1)

    if model_type == 'lasso':
        model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Lasso', Lasso(alpha=optimal_alpha))
            ]
        )

    if model_type == 'ridge':
        model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Ridge', Ridge(alpha=optimal_alpha))
            ]
        )
    
    # Performs K-Fold cross validation and saves the results
    RMSE, R_Squared, Adj_R_Squared = kfold_cv(model, X, y, cv=cv)
    
    return RMSE, R_Squared, Adj_R_Squared



def rforest_parameter_tuning(X, y, parameter_type, parameter_tuning_list, n_trees=10, max_depth=None,
                             min_split=2, min_leaf=1, max_features='sqrt', fig_size=(6,6), cv=5, return_series=False):
    '''
    Function that fits a random forest model with the specified hyper-parameter before doing cross validation.
    The function returns a plot of the training and validation RMSEs.

    :params:
        parameter_type (str) :
            Hyper-parameter we wish to tune.
        parameter_tuning_list (list) :
            List of parameters we wish to iterate over.
        return_series (bool) : 
            Default is False. If True, returns the training and validation series used to plot the graphs. 
        
        Other params are the hyper-paramters of a Random Forest set as the default values.
    
    :returns:
        training_rmse : 
            List of training RMSE scores.
        validation_rmse : 
            List of validation RMSE scores.
        
    '''
    # Lists to store our results
    param_tuning_train_rmse = []
    param_tuning_valid_rmse = []
    
    # Converting data into ndarrays
    X = np.array(X)
    y = np.array(y)
    
    kwargs = {
        'n_estimators' : n_trees,
        'max_depth' : max_depth,
        'min_samples_split' : min_split,
        'min_samples_leaf' : min_leaf,
        'max_features' : max_features,
        'n_jobs' : -1 
    }

    for parameter in parameter_tuning_list:
        if parameter_type == 'n_estimators':
            kwargs['n_estimators'] = parameter

        if parameter_type == 'max_depth':
            kwargs['max_depth'] = parameter

        if parameter_type == 'min_samples_leaf':
            kwargs['min_samples_leaf'] = parameter

        if parameter_type == 'max_features':
            kwargs['max_features'] = parameter

        model = RandomForestRegressor(**kwargs)

        RMSE, R_Squared, Adj_R_Squared = kfold_cv(model, X, y, cv=cv)

        param_tuning_train_rmse.append(np.mean(RMSE['Training']))
        param_tuning_valid_rmse.append(np.mean(RMSE['Validation']))
        
        kwargs = {
            'n_estimators' : n_trees,
            'max_depth' : max_depth,
            'min_samples_split' : min_split,
            'min_samples_leaf' : min_leaf,
            'max_features' : max_features,
            'n_jobs' : -1 
            }

    param_tuning_train_rmse = pd.Series(param_tuning_train_rmse, index=parameter_tuning_list)
    param_tuning_valid_rmse = pd.Series(param_tuning_valid_rmse, index=parameter_tuning_list)     
    
    print('For {}, the best value is {}.'.format(
        parameter_type,
        param_tuning_valid_rmse.nsmallest(n=1).index[0])
    )

    # Plotting the RMSE Scores 
    viz.hyper_param_plot(param_tuning_train_rmse, param_tuning_valid_rmse, fig_size=fig_size)
    
    if return_series:
        return param_tuning_train_rmse, param_tuning_valid_rmse



def gboost_parameter_tuning(X, y, parameter_type, parameter_tuning_list, learning_rate=0.1, n_trees=100,
                            max_depth=None, min_split=2, min_leaf=1, max_features='sqrt', subsample=1.0, verbose=0,
                            fig_size=(6,6), cv=5, return_series=False):

    '''
    Function that fits a gradient boosting model with the specified hyper-parameter before doing cross validation.
    The function returns a plot of the training and validation RMSEs.

    :params:
        parameter_type (str) :
            Hyper-parameter we wish to tune.
        parameter_tuning_list (list) :
            List of parameters we wish to iterate over.
        return_series (bool) : 
            Default is False. If True, returns the training and validation series used to plot the graphs. 
        
        Other params are the hyper-paramters of a Random Forest set as the default values.
    
    :returns:
        training_rmse : 
            List of training RMSE scores.
        validation_rmse : 
            List of validation RMSE scores.
        
    '''
        # Lists to store our results
    param_tuning_train_rmse = []
    param_tuning_valid_rmse = []
    
    # Converting data into ndarrays
    X = np.array(X)
    y = np.array(y)
    
    kwargs = {
        'learning_rate' : learning_rate,
        'n_estimators' : n_trees,
        'max_depth' : max_depth,
        'min_samples_split' : min_split,
        'min_samples_leaf' : min_leaf,
        'max_features' : max_features,
        'subsample' : subsample,
        'verbose' : verbose
    }

    for parameter in parameter_tuning_list:
        if parameter_type == 'learning_rate':
            kwargs['learning_rate'] = parameter

        if parameter_type == 'n_estimators':
            kwargs['n_estimators'] = parameter

        if parameter_type == 'max_depth':
            kwargs['max_depth'] = parameter

        if parameter_type == 'min_samples_leaf':
            kwargs['min_samples_leaf'] = parameter

        if parameter_type == 'max_features':
            kwargs['max_features'] = parameter

        if parameter_type == 'subsample':
            kwargs['subsample'] = parameter

        model = GradientBoostingRegressor(**kwargs)

        RMSE, R_Squared, Adj_R_Squared = kfold_cv(model, X, y, cv=cv)

        param_tuning_train_rmse.append(np.mean(RMSE['Training']))
        param_tuning_valid_rmse.append(np.mean(RMSE['Validation']))
        
        kwargs = {
            'learning_rate' : learning_rate,
            'n_estimators' : n_trees,
            'max_depth' : max_depth,
            'min_samples_split' : min_split,
            'min_samples_leaf' : min_leaf,
            'max_features' : max_features,
            'subsample' : subsample,
            'verbose' : verbose
        }

    param_tuning_train_rmse = pd.Series(param_tuning_train_rmse, index=parameter_tuning_list)
    param_tuning_valid_rmse = pd.Series(param_tuning_valid_rmse, index=parameter_tuning_list)        
    
    print('For {}, the best value is {}.'.format(
        parameter_type,
        param_tuning_valid_rmse.nsmallest(n=1).index[0])
    )

    # Plotting the RMSE Scores 
    viz.hyper_param_plot(param_tuning_train_rmse, param_tuning_valid_rmse, fig_size=fig_size)
    
    if return_series:
        return param_tuning_train_rmse, param_tuning_valid_rmse



def kfold_cv_ensemble(X, y, model, cv=5):
    '''
    Function that performs cross validation and returns the metrics of the training and validation datasets for ensemble models.
    
    :params:
        model (estimator) :
            Sklearn ensemble model estimator with the optimal hyper-parameters found during tuning.

    :returns:
        RMSE (dict) : 
            Dictionary of the training and validation RMSEs scores obtained during cross validation.
        R_Squared (dict)
            Dictionary of the training and validation R-Squared scores obtained during cross validation.
        Adj_R_Squared (dict)
            Dictionary of the training and validation Adjusted R-Squared scores obtained during cross validation.
        
    '''

    # Converting data into ndarrays
    X = np.array(X)
    y = np.array(y)
    
    # Performs K-Fold cross validation and saves the results
    RMSE, R_Squared, Adj_R_Squared = kfold_cv(model, X, y, cv=cv)
    
    return RMSE, R_Squared, Adj_R_Squared



def model_results_summary(RMSE, R_Squared, Adj_R_Squared, model_type):
    '''
    Function to format results and display them.

    :params:
        RMSE (dict) : 
            Dictionary of the training and validation RMSEs scores obtained during cross validation.
        R_Squared (dict)
            Dictionary of the training and validation R-Squared scores obtained during cross validation.
        Adj_R_Squared (dict)
            Dictionary of the training and validation Adjusted R-Squared scores obtained during cross validation.
        model_type (str) :
            Name of model whose results are being evaluated.
    '''
    print(' '*13, '{} Results Summary'.format(model_type))
    print('=' * 61)
    print('\t'*2,'Mean RMSE', '\t', 'Mean R^2', '\t', 'Mean Adj R^2')
    
    print('Training',
          '\t', '{:.4f}'.format(np.mean(RMSE['Training'])),
          '\t', '{:.4f}'.format(np.mean(R_Squared['Training'])),
          '\t', '{:.4f}'.format(np.mean(Adj_R_Squared['Training'])))
    
    print('Validation',
          '\t','{:.4f}'.format(np.mean(RMSE['Validation'])),
          '\t', '{:.4f}'.format(np.mean(R_Squared['Validation'])),
          '\t', '{:.4f}'.format(np.mean(Adj_R_Squared['Validation'])))



def append_results(result_df, model_name, rmse, score, print_results=False):
    '''
    Function to append results to a dataframe that keeps track of all model performance.

    :params:
        result_df (dataframe) :
            Dataframe that we wish to append the results to.
        model_name (str) :
            Name of the model whose results we wish to append.
        rmse (dict) : 
            Dictionary of the training and validation RMSEs scores obtained during cross validation.
        score (dict)
            Dictionary of the training and validation R-Squared scores obtained during cross validation.
        print_results (bool) :
            Default is False. Prints the dataframe if True.
    '''
    
    results_to_append = pd.DataFrame(
        data={
            'Model' : model_name,
            'RMSE' : np.mean(rmse['Validation']),
            'R-Squared' : np.mean(score['Validation'])
        },
        index=[0]
    )
    if print_results:    
        print(results_to_append)
    
    result_df = result_df.append(results_to_append, ignore_index=True)
    
    return result_df
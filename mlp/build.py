

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


import model_functions as modf
import pickle
import config


model_type = config.model


def split_data(df):

    X = df.drop(['total_users', 'log_total_users', 'guest_users', 'registered_users'], axis=1)
    y = df.log_total_users
    X_train, X_valid, y_train, y_valid = train_test_split(X ,y, test_size=0.2)

    return X_train, X_valid, y_train, y_valid


def create_model():

    if model_type == 'linear':
        model = LinearRegression(**config.linear_params)

    if model_type == 'lasso':
        model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Lasso', Lasso(**config.lasso_params))
            ]
        )

    if model_type == 'ridge':
        model = Pipeline(steps=[
                ('Standardise', StandardScaler()),
                ('Ridge', Ridge(**config.ridge_params))
            ]
        )

    if model_type == 'rforest':
        model = RandomForestRegressor(**config.rforest_params)

    if model_type == 'gboost':
        model = GradientBoostingRegressor(**config.gboost_params)

    return model


def fit_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model



def score_model(model, X_valid, y_valid):

    rmse = modf.get_rmse(model, X_valid, y_valid)
    r2 = model.score(X_valid, y_valid)

    return rmse, r2



def print_metrics(rmse, r2):
    print('\n')
    print(' '*13, '{} Results Summary'.format(model_type.capitalize()))
    print('=' * 61)
    print('Model RMSE : {}'.format(rmse))
    print('Model R Squared : {}'.format(r2))



def build_model(model, df):

    X_train, X_valid, y_train, y_valid = split_data(df)
    model = create_model()
    model = fit_model(model, X_train, y_train)
    rmse, r2 = score_model(model, X_valid, y_valid)

    print_metrics(rmse, r2)




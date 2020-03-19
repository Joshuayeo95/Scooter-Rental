''' Config for hyper parameter tuning ''' 

# Model to train and do predictions
# Available options are: 'linear', 'lasso', 'ridge', 'rforest' and 'gboost'
model = 'gboost'

linear_params = {
	'fit_intercept' : True,
	'n_jobs' : -1
}

lasso_params = {
	'alpha' : 0.001,
	'fit_intercept' : True
}

ridge_params = {
	'alpha' : 11,
	'fit_intercept' : True,
}

rforest_params = {
	'n_estimators' : 60,
	'max_depth' : 30,
	'min_samples_split' : 2,
	'min_samples_leaf' : 1,
	'n_jobs' : -1 
}

gboost_params = {
	'learning_rate' : 0.1,
	'n_estimators' : 50,
	'max_depth' : 10,
	'max_features' : 21,
	'subsample' : 0.75
}

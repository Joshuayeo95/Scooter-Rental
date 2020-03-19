''' Script to preprocess the dataset such that it is ready for model training or predictions. '''

from io import StringIO
import pandas as pd
import numpy as np
import requests
import pickle

def download_data():
	url = 'https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv'
	response = requests.get(url)
	df = pd.DataFrame(pd.read_csv(StringIO(response.text)))

	return df

def process_data(df):


	# Formating columns
	df.columns = df.columns.map(str.strip)
	df.columns = df.columns.str.replace("-", "_")

	# Filling potential missing values for numeric columns with their median
	for col in df.select_dtypes(include=np.number):
		df[col] = df[col].fillna(df[col].median())

	# Creating target variable
	df['total_users'] = df.guest_users + df.registered_users

	# Creating log transformed target variable
	df['log_total_users'] = np.log(df.total_users)

	# Fixing typos
	df.weather = df.weather.str.lower()

	df.weather = df.weather.apply(lambda x:
	                              'clear' if x == 'lear'
	                              else 'cloudy' if x == 'loudy'
	                              else 'light snow/rain' if x == 'heavy snow/rain'
	                              else x)

	# Changing date to datetime format
	df.date = pd.to_datetime(df.date, format='%Y-%m-%d')

	# Creating additional time related features
	df['year'] = df['date'].dt.year
	df['month'] = df['date'].dt.month_name()
	df['dayofweek'] = df['date'].dt.day_name()

	# Creating a variable for the season of the year
	seasons = ['winter', 'winter', 'spring', 'spring', 'spring', 'summer',
	           'summer', 'summer', 'autumn', 'autumn', 'autumn', 'winter']

	month_to_season = dict(zip(range(1,13), seasons))

	df['season'] = df['date'].dt.month.map(month_to_season)

	# Creating a variable for the difference between temp and feels-like temp
	df['temp_diff'] = np.abs(df.temperature - df.feels_like_temperature)

	# Changing feature types to categorical
	categorical_vars = ['hr', 'weather', 'month', 'dayofweek', 'season']

	for var in categorical_vars:
	    df[var] = df[var].astype('category')

	# Saving a dataset for EDA before we one hot encode our dataset
	eda_dataset_filename = 'eda_dataset.pickle'
	eda_dataset_filepath = './data/eda_dataset.pickle'
	with open (eda_dataset_filename, 'wb') as f:
	    pickle.dump(df, f)

	print('The dataset which is used for EDA has been saved as "{}".'.format(eda_dataset_filename))

	# One Hot Encoding
	df = pd.get_dummies(df, drop_first=True)

	# Creating lagged terms up to the 12th order for the logged target
	df['l1_log_total_users'] = df.log_total_users.shift(1)
	df['l2_log_total_users'] = df.log_total_users.shift(2)
	df['l3_log_total_users'] = df.log_total_users.shift(3)
	df['l4_log_total_users'] = df.log_total_users.shift(4)
	df['l5_log_total_users'] = df.log_total_users.shift(5)
	df['l6_log_total_users'] = df.log_total_users.shift(6)
	df['l7_log_total_users'] = df.log_total_users.shift(7)
	df['l8_log_total_users'] = df.log_total_users.shift(8)
	df['l9_log_total_users'] = df.log_total_users.shift(9)
	df['l10_log_total_users'] = df.log_total_users.shift(10)
	df['l11_log_total_users'] = df.log_total_users.shift(11)
	df['l12_log_total_users'] = df.log_total_users.shift(12)

	# Droping rows with NaN values due to lagged term creation 
	df = df.dropna().reset_index(drop=True)

	# Dropping redundant variables
	df = df.drop(['date', 'feels_like_temperature'], axis=1)

	trainset_filename = 'training_dataset.pickle'
	trainset_filepath = './data/training_dataset.pickle'

	with open(trainset_filename, 'wb') as f:
		pickle.dump(df, f)

	print('The cleaned training dataset has been saved as "{}".'.format(trainset_filename))


	return df

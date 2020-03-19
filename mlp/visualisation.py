'''Visualisation Functions Using Plotly '''

### Importing Libraries ###

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

# Plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import tools
from plotly.offline import init_notebook_mode


def hyper_param_plot(train_rmse, valid_rmse, title='Training and Validation RMSE over different Alphas', fig_size=(6,6)):
	'''
	Function to plot the RMSE of the training and validation results.
	'''

	plt.figure(figsize=fig_size)
	plt.title(title)

	train_rmse.plot(color='blue', grid=True, label='Training RMSE')
	valid_rmse.plot(color='red', grid=True, label='Validation RMSE')

	plt.legend(loc='center right')
	plt.show()
    


def var_distribution_plots(df, x_var, transform=False, transform_type=None):
    '''
		Function that plots the distribution of the variable using Histograms and Boxplots.
		
		:params:
			df (dataframe): Dataset. 
			x_var (str) : Name of the variable we want to visualise.
			transform (bool) : Default is False. Set to True if we want to plot the distributions for the transformed x variable.
			transform_type (str) : Transformation to be applied to x variable. Options include 'log', 'log1p' and 'sqrt'.

	'''
	# Distribution plots of variable and transformed variable
    if transform:
        if transform_type == 'log':
            x_trans = np.log(df[x_var])
        
        if transform_type == 'log1p':
            x_trans = np.log1p(df[x_var])
            
        if transform_type == 'sqrt':
            x_trans = np.sqrt(df[x_var])
            
        # Creating subplot
        fig = tools.make_subplots(rows=2, cols=2, print_grid=False, vertical_spacing=0.25,
                                  subplot_titles=['Histogram',
                                                  'Histogram after {} Transformation'.format(transform_type.capitalize()),
                                                  'Boxplot',
                                                  'Boxplot after {} Transformation'.format(transform_type.capitalize())])
       	
       	# Histogram                            
        fig.add_trace(go.Histogram(x=df[x_var], nbinsx=100, opacity=0.8, hoverinfo='x'),
                      row=1, col=1)
        # Boxplot
        fig.add_trace(go.Box(x=df[x_var], opacity=0.8, hoverinfo='x', ),
                      row=2, col=1)
        # Histogram of Transformed Variable
        fig.add_trace(go.Histogram(x=x_trans, nbinsx=100, opacity=0.8, hoverinfo='x'),
                      row=1, col=2)
        # Boxplot of Transformed Variable
        fig.add_trace(go.Box(x=x_trans, opacity=0.8, hoverinfo='x'),
                      row=2, col=2)
        
        # Updating layout of our figure
        fig.update_layout(
            height=600,
            width=800,
            title='Plots for the Variable: {}'.format(x_var.capitalize().replace('_', ' ')),
            showlegend=False,
            yaxis3_showticklabels=False,
            yaxis4_showticklabels=False,
            paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='rgb(243, 243, 243)')
       	
       	# Updating the labels for y-axes 
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        # Updating the labels for x-axes
        fig.update_xaxes(title_text="Daily Users", row=1, col=1)
        fig.update_xaxes(title_text="{} Daily Users".format(transform_type.capitalize()), row=1, col=2)
        fig.update_xaxes(title_text="Daily Users", row=2, col=1)
        fig.update_xaxes(title_text="{} Daily Users".format(transform_type.capitalize()), row=2, col=2)
    
    # Distribution plots of variable only    
    else:
        
    	# Creating subplot
        fig = tools.make_subplots(rows=2, cols=1, print_grid=False, vertical_spacing=0.20,
                                  subplot_titles=['Histogram', 'Boxplot'])

        # Histogram
        fig.add_trace(go.Histogram(x=df[x_var], nbinsx=100, opacity=0.8, hoverinfo='x'),
                      row=1, col=1)
        # Boxplot
        fig.add_trace(go.Box(x=df[x_var], opacity=0.8, hoverinfo='x'),
                      row=2, col=1)
    	
    	# Updating layout of out figure
        fig.update_layout(
            height=600,
            width=450,
            title='Plots for the Variable: {}'.format(x_var.capitalize().replace('_', ' ')),
            showlegend=False,
            yaxis2_showticklabels=False,
            paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='rgb(243, 243, 243)'
        )
        
        # Updating the labels for y-axes
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        # Updating the labels for x-axes
        fig.update_xaxes(title_text="Daily Users", row=1, col=1)
        fig.update_xaxes(title_text="Daily Users", row=2, col=1)
    
    # Display Figure                         
    fig.show()



def scatter_plot(df, x_var, target, transform_type='log'):
	'''
		Function to plot the scatter plots of the x variable against the target as well as log-transformed target.
		This function only works if the 

		:params:
			df (dataframe): Dataset. 
			x_var (str) : Name of the variable we want to visualise.
			target (str) : Name of our target variable. 
			transform_type (str) : Default is 'log'. Transformation to be applied to x variable. Options include 'log', 'log1p' and 'sqrt'.

	'''
	if transform_type == 'log':
		y_trans = np.log(df[target])

	if transform_type == 'log1p':
		y_trans = np.log1p(df[target])

	if transform_type == 'sqrt':
		y_trans = np.sqrt(df[target])

	corr = df[x_var].corr(df[target]) # Correlation of x variable and target

	corr_trans = df[x_var].corr(y_trans) # Correlation of x variable and transformed target

	# Creating subplot
	fig = tools.make_subplots(
    	rows=1, cols=2, print_grid=False,
    	subplot_titles=[
    	'{} vs. {}<br>Correlation = {:.2f}'.format(
    		x_var.capitalize(),
    		target.capitalize(),
    		corr),
    	'{} vs. Log({})<br>Correlation = {:.2f}'.format(
    		x_var.capitalize(),
    		target.capitalize(),
    		corr_trans)
    	]
                             )
	
	# Adding scatterplot of x variable against target
	fig.add_trace(
        go.Scatter(x=df[x_var], y=df[target], mode='markers', opacity=0.8),
        row=1, col=1
        )
    # Adding scatterplot of x variable against transformed target
	fig.add_trace(go.Scatter(x=df[x_var], y=y_trans, mode='markers', opacity=0.8), row=1, col=2)

    # Updating figure layout
	fig.update_layout(
        height=400,
        width=800,
        title='Scatter Plots for Variable: {}'.format(x_var.capitalize().replace('_', ' ')),
        showlegend=False,
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)'
    )

    # Updating the labels for y-axes
	fig.update_yaxes(title_text='{}'.format(target.capitalize().replace('_', ' ')),
                     row=1, col=1)
	fig.update_yaxes(title_text='Log({})'.format(target.capitalize().replace('_', ' ')),
                     row=1, col=2)
    
    # Updating the labels for x-axes
	fig.update_xaxes(title_text='{}'.format(x_var.capitalize().replace('_', ' ')),
                     row=1, col=1)
	fig.update_xaxes(title_text='{}'.format(x_var.capitalize().replace('_', ' ')),
                     row=1, col=2)
    
	fig.show()


def numeric_var_plots(df, x_var, target):
    '''
    Function that plots the distribution of the variable as well as its distribution against the Target LogSales=Price.
    
    :params:
        x_var : Feature column in the dataset.
        target : Set to true to plot variable against target variable. Default is True.
    '''

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,8))
    f.tight_layout(pad=5.0)

    ax1.set_title('Distribution for Variable: {}'.format(x_var.capitalize()), size=12)
    _, p_val = stats.normaltest(df[x_var])
    anchored_text = AnchoredText('Skew = {:.2f}\nKurtosis = {:.2f}\nP-val = {:.4f}'.format(stats.skew(df[x_var]),
                                                                                           stats.kurtosis(df[x_var]),
                                                                                           p_val), loc='center right', frameon=False)
    ax1.add_artist(anchored_text)
    
    sns.distplot(df[x_var], bins=100, ax=ax1)

    stats.probplot(df[x_var], plot=ax2)
    
    sns.boxplot(df[x_var], ax=ax3)
    ax3.set_title('Boxplot for {}'.format(x_var.capitalize()), size=12, y=1.05)
    
    sns.scatterplot(x=df[x_var], y=df[target], alpha=0.5, ax=ax4)
    correlation = df[x_var].corr(df[target])
    ax4.set_title('Correlation = {:.2f}'.format(correlation), size=12, y=1.05)

    sns.despine();



def sorted_categorical_boxplots(df, cat_var, target):
    
    category_groupby = df.groupby([cat_var])
    
    # Sort order by median values
    sorted_categories = category_groupby[target].median().sort_values().keys().to_list()
        
    data = []

    for cat in sorted_categories:
        if type(cat) == str:
            cat_name = cat.capitalize()
        else:
            cat_name = cat
        
        data.append(go.Box(
            y = df.loc[df[cat_var] == cat][target],
            name = cat_name)
                   )

    fig = go.Figure(data=data)
    
    fig.update_layout(
        title = 'Categorical Box Plots for variable: {}'.format(cat_var.capitalize()),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False)
    
    fig.show()
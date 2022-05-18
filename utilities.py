import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
#from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.graph_objs import Layout

pd.options.display.float_format = ' {:,.2f}'.format 
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

def plotly_scatter(df, x, y, title="", name=None):
    layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(
        l=10,
        r=10,
        b=50,
        t=50,
        pad=1
    ),
    xaxis_title=x,
    yaxis_title=y,
    title=title
)
    fig = go.Figure(data=go.Scatter(x=df[x], y=df[y], mode='markers', name=name), layout=layout)
    fig.show()
    return fig

def plotly_boxplot(df, x, y, title="", points='all'):
    layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(
        l=10,
        r=10,
        b=50,
        t=50,
        pad=1
    ),
    xaxis_title=x,
    yaxis_title=y,
    title=title
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Box(y=df[y], x=df[x], name="All Points",
                    jitter=.3,
                    pointpos=-1.8,
                    boxpoints=points, # represent all points
                        ))
    fig.show()
    return fig

def plotly_bar(df, x, y, title=""):
    layout = Layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(
                        l=10,
                        r=10,
                        b=50,
                        t=50,
                        pad=1
                        ),
                    xaxis_title=x,
                    yaxis_title=y,
                    title=title
                    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(
                        x=df[x],
                        y=df[y],
                        marker_line_width=3
                        )
                 )
    fig.show()
    return fig



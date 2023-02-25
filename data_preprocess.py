
# Load all the necessary libraries
import numpy as np  # numerical computation with arrays
import pandas as pd # library to manipulate datasets using dataframes

# Statistical libraries
from scipy.stats import norm 
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# Load plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize":(8, 4), "figure.dpi":300}) #width=8, height=4

# Load sklearn libraries for machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgbm

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore") 

# Load the dataset using pandas
df = pd.read_csv("train_data.csv.zip")
# Set column 'index' as the index of rows
df = df.set_index('index')

# Load the test dataset using pandas
test_df = pd.read_csv("test_data.csv.zip")
# Set column 'index' as the index of rows
test_df = test_df.set_index('index')


#  Scatter plot of latitude and longitude, colored using variation in temperature. 
p = plt.scatter(x=df['lon'], y=df['lat'], c=df['contest-tmp2m-14d__tmp2m'])
plt.colorbar(p)
plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.show()


'''
Observation: Target variable is dependent on both longitude and latitude.
High longitude and low latitdes have much high temperture compared to high latitudes and low longitudes
'''
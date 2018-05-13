from scipy.stats import t, norm
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.weightstats import *

df = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week3/regresion/botswana.tsv",# na_filter=False,
                   na_values=["NA"], delim_whitespace=True)
# print(df["religion"].describe())
# q2
df.replace('NA',np.NaN)
df.dropna(axis=0, how='any', inplace=True)
print(df.describe())

# q3
p=0.0000000000000000000000000000000000000003
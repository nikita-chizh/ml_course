from scipy.stats import t, norm
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.weightstats import *

# q4
data = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week3/illiteracy.txt")
answ4 = data.corr().values[0][1]
answ4 = round(answ4, 4)

# q5
answ5 = data.corr(method="spearman").values[0][1]
answ5 = round(answ5, 4)

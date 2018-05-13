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
n = 160
Z = (9.57 - 9.5) *  (n ** 0.5) / 0.4
p1 = 2 * (1 - norm.cdf(abs(Z)))
answ1 = round(p1, 4)

# q5
data = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week2/diamonds.txt")
collist = data.columns.tolist()
collist.remove("price")
X_train, X_test, y_train, y_test = \
    train_test_split(data[collist], data["price"], test_size=0.25, random_state=1)

lreg = LinearRegression()
lreg.fit(X_train, y_train)
freg = RandomForestRegressor(random_state=1)
freg.fit(X_train, y_train)
###
Lpred = lreg.predict(X_test)
Fpred = freg.predict(X_test)
##
Lerr = np.abs(y_test - Lpred)
Ferr = np.abs(y_test - Fpred)
res = scipy.stats.ttest_rel(Lerr, Ferr)
print(res)
answ = CompareMeans(DescrStatsW(Lerr), DescrStatsW(Ferr)).tconfint_diff()
print(round(answ[0], 10))





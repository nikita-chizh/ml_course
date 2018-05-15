import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

milk_data = pd.read_csv('/Users/nikita/PycharmProjects/ML_Tasks/course5/week1/timelines/monthly-milk-production.csv',
                   index_col=['month'], parse_dates=['month'], dayfirst=True, sep=';')
milk_data.plot()
plt.show()
import statsmodels.api as sm
answ4 = sm.tsa.stattools.adfuller(milk_data["milk"])

import calendar
months = list(range(1, 13))
years = list(range(1962, 1975))
alldays = []
for year in years:
    ydays = []
    for month in months:
        days = calendar.monthrange(year, month)
        alldays.append(days[1])

# division
i = 0
for idx, row in milk_data.iterrows():
    milk_data.loc[idx,'milk'] = milk_data.loc[idx,'milk']/alldays[0]
    i = i + 1
answ5 = round(milk_data[["milk"]].sum()[0] ,2)
milk_data.plot()
plt.show()

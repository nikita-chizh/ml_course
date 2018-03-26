import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


path = "/Users/nikita/PycharmProjects/ML_Tasks/course2/week2/"
df = pd.read_csv(path + "bikes_rent.csv")
#
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
a = 0 / 4
for idx, feature in enumerate(df.columns[:-1]):
    df.plot(feature, "cnt", subplots=True, kind="scatter", ax=axes[int(idx / 4), int(idx % 4)])
# __Блок 1. Ответьте на вопросы (каждый 0.5 балла):__
# 1. Каков характер зависимости числа прокатов от месяца?
#   Парабола с отрицательным коф у квадрата
# 1. Укажите один или два признака, от которых число прокатов скорее всего зависит линейно
#    температура и ощущаемая температура
df1 = df.loc[:, df.columns != "cnt"]
df2 = df["cnt"]
# print(df1.corr())
# print(df1.corrwith(df2))
#
df3 = df.loc[:, "temp":"cnt"]
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
df_shuffled = shuffle(df, random_state=123)
X = scale(df_shuffled[df_shuffled.columns[:-1]])
y = df_shuffled["cnt"]
from sklearn.linear_model import LinearRegression

lregr  = LinearRegression()
lregr.fit(X, y)
wghts_feat  = zip(lregr.coef_, df.columns)
#
# Код 3.1 (1 балл)
alphas = np.arange(1, 500, 50)
coefs_lasso = np.zeros((alphas.shape[0], X.shape[1]))  # матрица весов размера (число регрессоров) x (число признаков)
coefs_ridge = np.zeros((alphas.shape[0], X.shape[1]))
# Для каждого значения коэффициента из alphas обучите регрессор Lasso
# и запишите веса в соответствующую строку матрицы coefs_lasso (вспомните встроенную в python функцию enumerate),
# а затем обучите Ridge и запишите веса в coefs_ridge.
from sklearn.linear_model import Lasso, Ridge

for i in range(0, alphas.shape[0]):
    lasso = Lasso(alpha=alphas[i])
    lasso.fit(X, y)
    coefs_lasso[i] = lasso.coef_
    #
    ridge = Ridge(alpha=alphas[i])
    ridge.fit(X, y)
    coefs_ridge[i] = ridge.coef_
import pandas as pd
import numpy as np
from scipy.stats import t, norm


def interval(array):
    mean = array.mean()
    S_n = (array.std(ddof=1)) / (array.shape[0] ** 0.5)
    y = 0.05 / 2
    df = array.shape[0] - 1
    T = t.ppf(1 - y, df)
    value = S_n * T
    return (mean - value, mean + value)


data = pd.read_table("/home/nikita/Desktop/some/help/bl/course/1/2.1/4_COURSE/week1/quiz1.txt")
# q2
# Постройте 95% доверительный интервал для средней
# годовой смертности в больших городах.
# Чему равна его нижняя граница? Округлите ответ до 4 знаков после десятичной точки.
allint = interval(data["mortality"])
answ1 = round(allint[0], 4)
# q3
# На данных из предыдущего вопроса постройте 95% доверительный интервал для средней
# годовой смертности по всем южным городам.
# Чему равна его верхняя граница? Округлите ответ до 4 знаков после десятичной точки.
sint = interval(data[data['location'] == 'South']["mortality"])
answ2 = round(allint[1], 4)
# _tconfint_generic(data['mortality'].mean(),
#                   data['mortality'].std(ddof=1) / np.sqrt(len(data)),                  len(data) - 1, 0.05, 'two-sided')
# q4
# На тех же данных постройте 95% доверительный интервал для средней годовой смертности по
# всем северным городам. Пересекается ли этот интервал с предыдущим?
# Как вы думаете, какой из этого можно сделать вывод?
nint = interval(data[data['location'] == 'North']["mortality"])
# q5
# Пересекаются ли 95% доверительные интервалы для средней жёсткости воды в северных и южных городах?
nintW = interval(data[data['location'] == 'North']["hardness"])
sintW = interval(data[data['location'] == 'South']["hardness"])
# q6
# При σ=1 какой нужен объём выборки, чтобы на уровне доверия 95% оценить среднее с точностью ±0.1?
# y=0.95 pres = 0.1
# pres = t * sigma /n**0.5;  n**0.5 = t * sigma /pres
# ppf -> обратная функция функции распределения cdf (quantile function)
np.ceil((norm.ppf(1 - 0.05 / 2) / 0.1) ** 2)
y = 0.05 / 2
t = norm.ppf(1 - 0.05 / 2)
n = (t / 0.1) ** 2
# _tconfint_generic(data[data.location == 'South'].hardness.mean(),
#                   data[data.location == 'South'].hardness.std(ddof=1) / np.sqrt(len(data[data.location == 'South'])),
#                   len(data[data.location == 'South']) - 1, 0.05, 'two-sided')
# _tconfint_generic(data[data.location == 'North'].hardness.mean(),
#
#                   data[data.location == 'North'].hardness.std(ddof=1) / np.sqrt(len(data[data.location == 'North'])),
#                   len(data[data.location == 'North']) - 1, 0.05, 'two-sided')

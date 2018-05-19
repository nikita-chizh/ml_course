import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from itertools import product

def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))
# Визуальный анализ ряда
df = pd.read_csv('/Users/nikita/PycharmProjects/ML_Tasks/course5/week1/timelines/WAG_C_M.csv',
                 index_col=['month'], parse_dates=['month'], dayfirst=True, sep=';')
df.plot()
plt.ylabel('Salary(rubles')
plt.show()
print("Критерий Дики-Фуллера Оригинал: p=%f" % sm.tsa.stattools.adfuller(df.WAG_C_M)[1])
# Ряд Нестационарный, с трендом на повышение и сезонностью с периодом в год
sm.tsa.seasonal_decompose(df.WAG_C_M).plot()
plt.show()
# Сделаем преобразование Бокса-Кокса для стабилизации дисперсии:
df["salary_bxcx"], lmbda = stats.boxcox(df.WAG_C_M)
df["salary_bxcx"].plot()
plt.ylabel(u'Transformed Salaries')
print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
print("Критерий Дики-Фуллера После преобразования Бокса-Кокса: p=%f" % sm.tsa.stattools.adfuller(df["salary_bxcx"])[1])
plt.show()
# Ряд сгладился, но все ще существенно не стационарный
# Попробуем сезонное дифференцирование c Сезонным лагом 12
df['salary_box_diff'] = df.salary_bxcx - df.salary_bxcx.shift(12)
sm.tsa.seasonal_decompose(df.salary_box_diff[12:]).plot()
print("Критерий Дики-Фуллера преобразования Бокса-Кокса и Сезонного дифференцирования: p=%f" %
      sm.tsa.stattools.adfuller(df.salary_box_diff[12:])[1])
plt.show()
# Критерий Дики-Фуллера <0.05 но тренд виден
# добавим обычное дифференцирование
df['salary_box_diff1'] = df.salary_box_diff - df.salary_box_diff.shift(1)
sm.tsa.seasonal_decompose(df.salary_box_diff1[13:]).plot()
print("Критерий Дики-Фуллера После еще одного сдига в дифференцирование: p=%f" %
      sm.tsa.stattools.adfuller(df.salary_box_diff1[13:])[1])
plt.show()
# Теперь тренд побежден будем работать с этим рядом далее

# Часть 2 Выбор начальных приближений для p, q, P, Qp,q,P,Q
sm.graphics.tsa.plot_acf(df.salary_box_diff1[13:].values.squeeze(), lags=48)
plt.xticks(np.arange(1, 48, 2))
plt.show()
#
sm.graphics.tsa.plot_pacf(df.salary_box_diff1[13:].values.squeeze(), lags=48)
plt.xticks(np.arange(1, 48, 2))
plt.figure(figsize=(8,4))
plt.show()
# Начальные приближения: Q, q, P и p
# Номер последнего значимого СЕЗОННОГО лага = 0 -> Q=0, НЕ СЕЗОННОГО n = 26 -> q =  26
# Номер последнего значимого СЕЗОННОГО лага = 12 -> P=1 p = 28
Q = 0
Qs = range(0, 2)

q = 4
qs = range(1, q)

P = 1
Ps = range(0, 2)

p = 5
ps = range(2, 6)
#
d=1
D=1

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print("PARAMS LIST LEN=", len(parameters_list))
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
#Поиск лучшей модели
best_model = None
for param in parameters_list:
    # try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model = sm.tsa.statespace.SARIMAX(df.salary_bxcx, order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
    # выводим параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        continue
    aic = model.aic
    # сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
warnings.filterwarnings('default')
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by = 'aic', ascending=True).head())
print(best_model.summary())

#отстатки
plt.figure(figsize = (15,8))
plt.subplot(211)
best_model.resid[13:].plot()
plt.ylabel(u'Residuals')

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)

print("Критерий Стьюдента для модели: p=%f" % stats.ttest_1samp(best_model.resid[13:], 0)[1])
print("Критерий Дики-Фуллера для модели: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

df['model'] = invboxcox(best_model.fittedvalues, lmbda)
plt.figure(figsize = (15,7))
df.WAG_C_M.plot()
df.model[13:].plot(color='r')
plt.ylabel('Wine sales')
plt.show()

#Построим Прогноз
import datetime
import dateutil.relativedelta as relativedelta
df2 = df[['WAG_C_M']]
date_list = [datetime.datetime.strptime("2016-09-01", "%Y-%m-%d") + relativedelta.relativedelta(months=x) for x in range(0, 24)]
future = pd.DataFrame(index=date_list, columns= df2.columns)
df2 = pd.concat([df2, future])
start = df.shape[0]
df2['forecast'] = invboxcox(best_model.predict(start=start, end=start+24), lmbda)

plt.figure(figsize=(15,7))
df2.WAG_C_M.plot()
df2.forecast.plot(color='r')
plt.ylabel('Salaries')
plt.show()
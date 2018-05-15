import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import warnings
from itertools import product

df = pd.read_csv('/Users/nikita/PycharmProjects/ML_Tasks/course5/week1/timelines/WAG_C_M.csv',
                 index_col=['month'], parse_dates=['month'], dayfirst=True, sep=';')
df.rename(index=str, columns={"WAG_C_M": "salary"})
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
# Попробуем сезонное дифференцирование c очевидным лагом 12
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
sm.graphics.tsa.plot_pacf(df.salary_box_diff1[13:].values.squeeze(), lags=48)
plt.show()
# Начальные приближения: Q, q, P и p
# Номер последнего значимого СЕЗОННОГО лага = 0 -> Q=0, НЕ СЕЗОННОГО n = 26 -> q =  26
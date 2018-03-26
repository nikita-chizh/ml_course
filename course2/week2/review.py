import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/2_course/week2/"
df = pd.read_csv(path + "bikes_rent.csv")
print(df.head())
#
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
a = 0 / 4
for idx, feature in enumerate(df.columns[:-1]):
    df.plot(feature, "cnt", subplots=True, kind="scatter", ax=axes[int(idx / 4), int(idx % 4)])
plt.show()
# __Блок 1. Ответьте на вопросы (каждый 0.5 балла):__
# 1. Каков характер зависимости числа прокатов от месяца?
#   Парабола с отрицательным коф у квадрата
# 1. Укажите один или два признака, от которых число прокатов скорее всего зависит линейно
#    температура и ощущаемая температура
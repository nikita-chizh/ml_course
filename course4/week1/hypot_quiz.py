import pandas as pd
import numpy as np
from scipy.stats import t, norm
from scipy import stats

# q1
answ1 = round(stats.binom_test(67, 100, 0.75, alternative = 'greater'), 4)

# q3
data = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week1/34_pines.txt")
x = data["sn"]
y = data["we"]
binx = list(range(0, 240, 40))
biny = binx
ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx,biny])
print(ret.statistic)
answ2 = round(ret.statistic.mean(),2)
# q4
answ4 = round(stats.chisquare(ret.statistic, axis=None)[0], 2)


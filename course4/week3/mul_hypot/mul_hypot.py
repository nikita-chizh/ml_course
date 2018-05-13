from scipy.stats import t, norm
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.weightstats import *

df = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week3/mul_hypot/AUCs.txt")
columns = list(df.columns.values)
columns.remove("Unnamed: 0")
cpairs = []

for i in range(len(columns) - 1):
    for j in range(i + 1, len(columns)):
        cpairs.append((columns[i], columns[j]))

wilcoxons = []
for p in cpairs:
    wilcoxons.append(stats.wilcoxon(df[p[0]], df[p[1]])[1])
maxid = wilcoxons.index(min(wilcoxons))
answ1 = cpairs[maxid]

# q3
significant = list(filter(lambda x: x < 0.05, wilcoxons))
answ3 = len(significant)

# q4
from statsmodels.sandbox.stats.multicomp import multipletests
reject, p_corrected, a1, a2 = multipletests(wilcoxons,
                                        alpha = 0.05,
                                            method = 'holm')
answ4 = 0
# q5
reject, p_corrected, a1, a2 = multipletests(wilcoxons,
                                        alpha = 0.05,
                                            method = 'fdr_bh')
answ5 = 3

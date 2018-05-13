import numpy as np
import pandas as pd
from scipy.stats import t, norm
import scipy
import scipy.stats
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)

    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))

    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
    P = float(p1 * n1 + p2 * n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative='greater'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)


# q3
s1 = [1] * 10 + [0] * 24
s2 = [1] * 4 + [0] * 12
pv1 = proportions_diff_z_test(proportions_diff_z_stat_ind(s1, s2))
answ3 = round(pv1, 4)

# q4
data = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week2/banknotes.txt")
collist = data.columns.tolist()
collist.remove("real")

X_train1, X_test1, y_train1, y_test1 = \
    train_test_split(data[["X1", "X2", "X3"]], data["real"], test_size=0.25, random_state=1)

reg1 = LogisticRegression()
reg1.fit(X_train1, y_train1)
pred1 = reg1.predict(X_test1)
err1 = mean_absolute_error(y_test1, pred1)
y_test1 = y_test1.values
s1 = [1 if pred1[i] == y_test1[i] else 0 for i in range(len(pred1))]
###############
X_train2, X_test2, y_train2, y_test2 = \
    train_test_split(data[["X4", "X5", "X6"]], data["real"], test_size=0.25, random_state=1)

reg2 = LogisticRegression()
reg2.fit(X_train2, y_train2)
pred2 = reg2.predict(X_test2)
err2 = mean_absolute_error(y_test2, pred2)
y_test2 = y_test2.values
s2 = [1 if pred2[i] == y_test2[i] else 0 for i in range(len(pred2))]


def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = zip(sample1, sample2)
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    left_boundary = float(f - g) / n - z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    right_boundary = float(f - g) / n + z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    return (left_boundary, right_boundary)


def proportions_diff_z_stat_rel(sample1, sample2):
    sample = zip(sample1, sample2)
    n = len(sample)
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    return float(f - g) / np.sqrt(f + g - float((f - g) ** 2) / n)

pv2 = proportions_diff_z_test(proportions_diff_z_stat_rel(s1, s2))
import math
answ4 = pv2

# q5
int5 = proportions_diff_confint_rel(s1, s2)
answ5 = round(max(int5), 4)

#q6
n = 100
Z = (541.4 - 525) *  (n ** 0.5) / 100
p1 = 1 - norm.cdf(abs(Z))
answ6 = round(p1, 4)

# q7
n = 100
Z = (541.5 - 525) *  (n ** 0.5) / 100
p2 = 1 - norm.cdf(abs(Z))
answ7 = round(p2, 4)


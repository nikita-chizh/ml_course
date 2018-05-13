from scipy.stats import t, norm
import scipy.stats
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.weightstats import *

df = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week3/corr_analysis/water.txt")

# q1
answ1 = df.corr().values[0][1]
answ1 = round(answ1, 4)

# q2
answ2 = df.corr(method="spearman").values[0][1]
answ2 = round(answ2, 4)

# q3
south = df[df["location"] == "South"]
north = df[df["location"] == "North"]
scorr = south.corr()
ncorr = north.corr()
SandNcorr = np.array([south.corr().values[0][1], north.corr().values[0][1]])
answ3 = min(np.abs(SandNcorr))
answ3 = round(answ3 * -1, 4)

# q4
from sklearn.metrics import matthews_corrcoef
m1 = 239
m0 = 515
f1 = 203
f0 = 718
#     1
a = min(m0, f0)
b = min(m0, f1)
c = min(m1, f0)
d = min( m1, f1)
matthews = (a * d - b * c) / (((a + b) * (a + c)* (b + d) * (c + d)) ** 0.5)
answ4 = round(matthews, 4)

# q5
from scipy.stats import chi2_contingency
table = np.array([[a, b], [c, d]])
chi2 = chi2_contingency(table)
answ5 = round(chi2[0], 4)

# q6
#

males = [1] * m1 + [0] * m0
females = [1] * f1 + [0] * f0

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

def proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)

int6 = proportions_diff_confint_ind(males, females)
answ6 = round(int6[0], 4)

# q7
answ7 = proportions_diff_z_test(proportions_diff_z_stat_ind(males, females))

# q8
table_happines = [[197, 111, 33],
                  [382, 685, 331],
                  [110, 342, 333]]
chi2 = chi2_contingency(table_happines)
answ8 = round(chi2[0], 4)

# q9
answ9 = chi2[1]

# q10
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

table_happines = np.array(table_happines)
answ10 = cramers_corrected_stat(table_happines)
answ10 = round(answ10, 4)
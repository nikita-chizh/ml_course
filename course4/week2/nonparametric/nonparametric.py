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

# q4
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.weightstats import zconfint

sample = np.array([49, 58, 75, 110, 112, 132, 151, 276, 281, 362])
m0 = 200
int4 = stats.wilcoxon(sample - m0)
ans4 = round(int4[1], 4)
# q5
s1 = np.array([22, 22, 15, 13, 19, 19, 18, 20, 21, 13, 13, 15])
s2 = np.array([10.17, 18, 18, 15, 12, 4, 14, 15, 10])
int5 = stats.mannwhitneyu(s1, s2, alternative="less")
ans5 = round(int5[1], 4)

# q6
data = pd.read_table("/Users/nikita/PycharmProjects/ML_Tasks/course4/week2/nonparametric/challenger.txt")
def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries
fail = data["Temperature"][data['Incident'] == 1]
fail = np.array(fail)
success = data["Temperature"][data['Incident'] == 0]
success = np.array(success)
np.random.seed(0)
success_median_scores = map(np.mean, get_bootstrap_samples(success, 1000))
fail_median_scores = map(np.mean, get_bootstrap_samples(fail, 1000))
delta_median_scores = map(lambda x: x[1] - x[0], zip(success_median_scores, fail_median_scores))
int6 = stat_intervals(delta_median_scores, 0.05)
answ6 = round(min(np.abs(int6)), 4)

# q7
def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)

def get_random_combinations(n1, n2, max_combinations):
    index = range(n1 + n2)
    indices = set([tuple(index)])
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def permutation_zero_dist_ind(sample1, sample2, max_combinations=None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                   for index in itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr


def permutation_test(sample, mean, max_permutations=None, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_ind(sample, mean)

    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)
np.random.seed(0)
answ7 = permutation_test(success, fail, max_permutations = 10000)
answ7 = round(answ7, 4)
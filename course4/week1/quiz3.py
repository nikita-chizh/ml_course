import scipy.stats
import numpy as np
from scipy.stats import t, norm

# q1
y = 0.003
np.ceil((norm.ppf(1 - y / 2) / 0.1) ** 2)
answ1 = round(norm.ppf(1 - y / 2), 4)

#q5
def proportions_confint_diff_ind(p1, p2, l1, l2, alpha=0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    dif = z * np.sqrt(p1 * (1 - p1) / l1 + p2 * (1 - p2) / l2)
    left_boundary = (p2 - p1) - dif
    right_boundary = (p2 - p1) + dif

    return (left_boundary, right_boundary, p2 - p1)

l1, l2 = 11037, 11034
p1, p2 = 104/l1,  189/l2
res = proportions_confint_diff_ind(p1, p2, l1, l2)
answ5 = round(res[2], 4)
answ6 = round(res[1], 4)
answ7 = round((p2/(1-p2))/(p1/(1-p1)), 4)
# q8


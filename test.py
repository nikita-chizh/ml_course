import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t, norm

Z = norm.ppf(1 - float(0.04/2))
p = 0.6
n = 100
P = (p*(1-p)/n) ** 0.5
res = (p - Z * P, p + Z * P)
import numpy as np
from scipy.stats import chi2, norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# k degrees of freedom
k = 5
th_mean, th_var, _, _ = chi2.stats(k, moments='mvsk')
print("THEORY MEAN=", th_mean, "THEORY VARIANCE=", th_var)
# prob dens function
# what X is in 0.01 and 0.99 quantiles
start = chi2.ppf(0.01, k)
stop = chi2.ppf(0.99, k)
# theoretical dist
x = np.linspace(start, stop, 100)
plt.plot(x, chi2.pdf(x, k), 'r-', lw=5, alpha=0.6, label='chi2 pdf')
# histogram
# random nums from dist
r = chi2.rvs(k, size=1000)
plt.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
plt.legend(loc='best', frameon=False)
plt.title('Theoretical Dist')
plt.ylabel('Prob Dens')
plt.xlabel('X')
plt.show()


# calculate means for 1000 expirements with n samples in each
def sample_mean(n):
    smeans = []
    for x in range(1000):
        x = chi2.rvs(k, size=n)
        smeans.append(x.mean())
    return smeans


def show_smean_dist(n):
    data = sample_mean(n)
    plt.hist(data, normed=True)
    # calculating normal dist params
    mean = np.mean(data)
    var = np.var(data)
    print("Sample MEAN=", mean, "Sample VAR=", var)
    th_norm = norm(mean, var**0.5)
    x = np.linspace(1, 10, 10000)
    plt.plot(x, th_norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
    patch = mpatches.Patch(color='red', label='Нормальное')
    plt.legend(handles=[patch])


    #
    plt.title(str(n) + ' means')
    plt.ylabel('Гистограмма и Нормальное распределение')
    plt.xlabel('X Среднее')
    plt.show()


show_smean_dist(5)
show_smean_dist(50)
show_smean_dist(1000)

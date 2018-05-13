from statsmodels.stats.proportion import proportion_confint

# q1
res1 = proportion_confint(1, 50)
answ1 = round(res1[0], 4)
# q2
res2 = proportion_confint(1, 50, method='wilson')
answ2 = round(res2[0], 4)

from statsmodels.stats.proportion import samplesize_confint_proportion
import numpy as np
# q5
n_samples = int(np.ceil(samplesize_confint_proportion(0.98, 0.01)))

# q6
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
P = np.linspace(0, 1, 100)
N = [int(np.ceil(samplesize_confint_proportion(p, 0.01))) for p in P]
plt.plot(P, N)
patch = mpatches.Patch(color='red', label='N(P)')
plt.legend(handles=[patch])
plt.show()
n=max(N)
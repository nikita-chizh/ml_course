import numpy as np
from matplotlib import pyplot as plt

y = [1, 1, 3, 3, 6, 8, 11]
def Q(med):
    err = 0
    for yi in y:
        err = err + abs(med - yi)
    return err / len(y)

medians = range(11)
all_Q = [Q(med) for med in medians]
#
plt.plot(medians, all_Q, 'r-', lw=5, alpha=0.6, color="red")
plt.ylabel('Q')
plt.xlabel('median')
plt.show()
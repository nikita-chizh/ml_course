import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t, norm
import os
import pickle
from collections import Counter
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC



PATH_TO_DATA = "/Users/nikita/PycharmProjects/ML_Tasks/course6/capstone_user_identification/sparce_res"


with open(os.path.join(PATH_TO_DATA, 'X17_10_10.dat'), 'rb') as X_sparse_150users_pkl:
    X_sparse_150users = pickle.load(X_sparse_150users_pkl)
with open(os.path.join(PATH_TO_DATA, 'Y17_10_10.dat'), 'rb') as y_150users_pkl:
    y_150users = pickle.load(y_150users_pkl)

from collections import Counter
y_binary_128 =  []
for x in y_150users.values:
    k = x[0]
    if k == 128:
       y_binary_128.append(1)
    else:
        y_binary_128.append(0)
y_binary_128 = np.array(y_binary_128)
print(y_binary_128)
# new_shape = (y_binary_128.shape[0], 1)
# y_binary_128 = np.reshape(y_binary_128 , newshape = new_shape)
C = Counter(y_binary_128)
print(C)
train_sizes = np.linspace(0.25, 1, 20)
n_train, val_train, val_test =  learning_curve(SVC(kernel='linear')
                                                ,X = X_sparse_150users,
                                                y = y_binary_128, train_sizes=train_sizes, n_jobs=4)
print(1)
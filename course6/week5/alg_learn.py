import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
import random
from sklearn.linear_model import SGDClassifier



pd.set_option('display.max.columns', 100)

data_path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/data/"
model_path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/models/"

def under_samplling(df):
    # under sampling
    class1_df = df[df.target == 1]
    class0_df = df[df.target == 0]
    class0_df = class0_df.sample(n=class1_df.shape[0])
    res_df = pd.concat([class1_df, class0_df])
    res_df = shuffle(res_df)
    return res_df

def XY(data_type, unders = False):
    fname = "TRAIN_" + data_type + ".pkl"
    df = pd.read_pickle(data_path + fname)
    df = shuffle(df)
    cols = list(df.columns.values)
    cols.remove('target')
    if unders:
        df = under_samplling(df)
    X = df[cols]
    Y = df["target"]
    return X, Y

def tuning(csl, param_grid):
    grid_search = GridSearchCV(csl, param_grid, cv=4, scoring="f1" , n_jobs=4)
    grid_search.fit(X, Y)
    return grid_search

def write_res(fname, grid_search):
    info_file = open(model_path + fname + ".txt", "w")
    info_file.write(str(grid_search.best_params_))
    info_file.write("\n\n********************\n\n")
    info_file.write(str(grid_search.best_score_))
    info_file.close()
    best_model = grid_search.best_estimator_
    pickle.dump(best_model, open(model_path + fname + ".pkl", 'wb'))


def compute_SGD(data_type):
    fname = "SGD_" + data_type
    alpha_step = 0.0001 / 5
    param_grid = {
        'alpha': [alpha_step + alpha_step * i for i in range(10)],  # learning rate
        'n_iter': [1000],  # number of epochs
        'loss': ['log'],  # logistic regression,
        'penalty': ['l1', 'l2']
    }
    grid_search = tuning(SGDClassifier(), param_grid)
    grid_search.fit(X, Y)
    write_res(fname, grid_search)

def compute_SVM(data_type):
    fname = "SVM_" + data_type
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = tuning(svm.SVC(), param_grid)
    grid_search.fit(X, Y)
    write_res(fname, grid_search)


data_types = ["UNX", "UNX_SCALED", "DELTAS", "DELTAS_SCALED"]
for d in data_types:
    X, Y = XY(d)
    compute_SGD(d)
    compute_SVM(d)



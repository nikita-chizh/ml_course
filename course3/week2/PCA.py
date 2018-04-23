from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score as cv_score
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn import datasets
import matplotlib.patches as mpatches

matplotlib.style.use('ggplot')

path = "/Users/nikita/PycharmProjects/ML_Tasks/course3/week2/"

# ЗАДНИЕ 1
#
def plot_scores(d_scores):
    n_components = np.arange(1, d_scores.size + 1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()

def write_answer_1(optimal_d):
    with open(path + "pca_answer1.txt", "w") as fout:
        fout.write(str(optimal_d))


data = pd.read_csv('data_task1.csv')

D = data.shape[1]
d_scores = []
for d in range(1, D):
    model = PCA(n_components=d, svd_solver='full')
    scores = cv_score(model, data, n_jobs=4, cv = 3)
    d_score = scores.mean()
    d_scores.append(d_score)
id = d_scores.index(max(d_scores))
write_answer_1(id)
plot_scores(np.array(d_scores))

# # ЗАДАНИЕ 2
#
# def plot_variances(d_variances):
#     n_components = np.arange(1, d_variances.size + 1)
#     plt.plot(n_components, d_variances, 'b', label='Component variances')
#     plt.xlim(n_components[0], n_components[-1])
#     plt.xlabel('n components')
#     plt.ylabel('variance')
#     plt.legend(loc='upper right')
#     plt.show()
#
#
# def write_answer_2(optimal_d):
#     with open(path + "pca_answer2.txt", "w") as fout:
#         fout.write(str(optimal_d))
#
# data = pd.read_csv('data_task2.csv')
# D = data.shape[1]
# model = PCA(n_components=D)
# model.fit(data)
# pca_data = model.transform(data)
# vars = []
# for i in range(1, D):
#     tg = pca_data[:, i]
#     var = tg.var()
#     vars.append(var)
# vars.sort(reverse=True)
# diffs = [vars[i] - vars[i+1] for i in range(len(vars) - 1)]
# max_diff = max(diffs)
# max_diff_id = diffs.index(max_diff)
# plot_variances(np.array(vars))
# write_answer_2(max_diff_id)

#
# # ЗАДАНИЕ 3
# def plot_iris(transformed_data, target, target_names):
#     plt.figure()
#     for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
#         plt.scatter(transformed_data[target == i, 0],
#                     transformed_data[target == i, 1], c=c, label=target_name)
#     plt.legend()
#     plt.show()
#
# def write_answer_3(list_pc1, list_pc2):
#     with open("pca_answer3.txt", "w") as fout:
#         fout.write(" ".join([str(num) for num in list_pc1]))
#         fout.write(" ")
#         fout.write(" ".join([str(num) for num in list_pc2]))
#
# # загрузим датасет iris
# iris = datasets.load_iris()
# data = iris.data
# target = iris.target
# target_names = iris.target_names
# # PCA fitting
# D = data.shape[1]
# model = PCA(n_components=2)
# model.fit(data)
# pca_data = model.transform(data)
# f_cors = []
# s_cors = []
# from math import fabs
# list_pc1 = []
# list_pc2 = []
# for i in range(D):
#     fcor = fabs(np.corrcoef(data[:, i],pca_data[:, 0])[0][1])
#     scor = fabs(np.corrcoef(data[:, i],pca_data[:, 1])[0][1])
#     if fcor > scor:
#         list_pc1.append(i + 1)
#     else:
#         list_pc2.append(i + 1)
#
# write_answer_3(list_pc1, list_pc2)
#
# # ЗАДАНИЕ 4
#
# from sklearn.datasets import fetch_olivetti_faces
# from sklearn.decomposition import RandomizedPCA
#
#
# def write_answer_4(list_pc):
#     with open(path + "pca_answer4.txt", "w") as fout:
#         fout.write(" ".join([str(num) for num in list_pc]))
#
#
# data = fetch_olivetti_faces(shuffle=True, random_state=0).data
# image_shape = (64, 64)
# # PCA fitting
# D = data.shape[1]
# model = RandomizedPCA(n_components=10)
# model.fit(data)
# pca_data = model.transform(data)
#
#
# def cos_2(data, j, i):
#     squarer = lambda t: t ** 2
#     return data[j, i] ** 2 / np.sum(np.array([squarer(xi) for xi in data[j, :]]))
#
# list_pc = []
# for i in range(pca_data.shape[1]):
#     coses = []
#     for j in range(pca_data.shape[0]):
#         cos = cos_2(pca_data, j, i)
#         coses.append(cos)
#     list_pc.append(coses.index(max(coses)))
#
# write_answer_4(list_pc)

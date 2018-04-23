import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn

path = "/Users/nikita/PycharmProjects/ML_Tasks/course3/week3/"
data = pd.read_csv(path + 'train.csv', na_values="NaN")
real_features = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                 "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
discrete_features = ["Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24",
                     "Medical_History_32"]
cat_features = data.columns.drop(real_features).drop(discrete_features).drop(["Id", "Response"]).tolist()

# print(data[real_features].describe())
# print("DISCRETE-------------------------------------------\n")
# print(data[discrete_features].describe())
# print("CATEGORIAL-------------------------------------------\n")
# print(data[cat_features].describe())
# print(data.shape)
def nil_part(df, col):
    isnan = df[col].isnull().sum()
    return isnan/df.shape[0]


# for col_name in real_features + discrete_features:
#     print(col_name)
#     print(nil_part(data, col_name))

# Task 2 histograms
# data[real_features].hist(bins=100, figsize=(20, 20))
# plt.show()
# data[discrete_features].hist(bins=100, figsize=(10, 10))
# plt.show()

# Task 3 pairplot
# bad_features = ["Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Product_Info_4"]
# normal_real_fetures = list(filter(lambda x: x not in bad_features, real_features))
# topirplot = data[normal_real_fetures]
# topirplot = topirplot.fillna(0)
# seaborn.pairplot(topirplot)
# plt.show()

# Task 4 correlation
cor = data[real_features].corr()
# Task 5
# Код 3. Постройте countplot
# fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharey=True)
# print(axes)
# features = ['Medical_Keyword_23', 'Medical_Keyword_39', 'Medical_Keyword_45']
# for i in range(len(features)):
#     seaborn.countplot(x=features[i], data=data, ax=axes[i], hue="Response")
# plt.show()
#
# from sklearn.utils import shuffle
# from sklearn.preprocessing import scale
# sdata = shuffle(data, random_state=321)
# subset_l  = 1000
# selected_features = real_features[:-4]
# objects_with_nan = sdata.index[np.any(np.isnan(sdata[selected_features].values), axis=1)]
# data_subset = scale(sdata[selected_features].drop(objects_with_nan, axis=0)[:subset_l])
# response_subset = sdata["Response"].drop(objects_with_nan, axis=0)[:subset_l]
# from sklearn.manifold import TSNE
# import matplotlib.cm as cm # импортируем цветовые схемы, чтобы рисовать графики.
# tnse = TSNE(random_state=321)
# tsne_representation = tnse.fit_transform(data_subset, response_subset)
from sklearn import svm
from sklearn.utils import shuffle
sdata = shuffle(data, random_state=321)
person_features = ["Ins_Age", "Ht", "Wt", "BMI"]
from itertools import product

person_features_pairs = list(product(person_features, person_features))
person_features_pairs_un = []
for i in range(len(person_features_pairs)):
    cur = person_features_pairs[i]
    rev = (cur[1], cur[0])
    if rev not in person_features_pairs[0:i] and cur[0] != cur[1]:
        person_features_pairs_un.append(cur)
svm_ = svm.OneClassSVM(gamma=10, nu=0.01)
svm_.fit(sdata[person_features])
labels = svm_.predict(sdata[person_features])

# plt.show()
# f, axarr = plt.subplots(2, 3, figsize=(12,8))
# colors = ["red" if x == -1 else "blue" for x in labels]
# for i in range(len(person_features_pairs_un)):
#     x = person_features_pairs_un[i][0]
#     y = person_features_pairs_un[i][1]
#     axarr[int(i / 3), int(i % 3)].scatter(sdata[x], sdata[y], c=colors, s=1)
# plt.show()
features = ["BMI", "Employment_Info_1", "Medical_History_32"]
for_displot = data[features].dropna()
for f in features:
    seaborn.distplot(for_displot[f], bins=50)
    plt.show()



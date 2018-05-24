import os
import pickle
# pip install tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

df = pd.DataFrame.from_csv("/Users/nikita/PycharmProjects/ML_Tasks/course6/week1/train_data_3users.csv")
X_toy, y_toy = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_sparse_toy = csr_matrix()
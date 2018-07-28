import os
import pickle
# pip install tqdm
import numpy as np
import pandas as pd
import json
from scipy.sparse import csr_matrix

#сошлось с типом
def to_sparce_df(df, site_dict):
    N = df.shape[0]  # кол во сессий
    M = len(site_dict)  # кол во уникальных сайтов
    siteMat = np.zeros((N, M), dtype=int)
    site_cols = ["site" + str(i) for i in range(1, df.shape[1] - 1)]
    for i in range(0, df.shape[0]):
        row = df.iloc[i][site_cols]
        for sid in row:
            # 0й сайт не учитываем, индексы сайтов в siteMat
            # начинаются с 0
            if sid != 0:
                siteMat[i][sid - 1] += 1
    session_col = df[["session_id.1"]]
    user_col = df[["user_id"]]
    resMat = np.concatenate((session_col, siteMat), axis=1)
    resMat = np.concatenate((resMat, user_col), axis=1)
    site_cols = ["site" + str(i) for i in range(1, M + 1)]
    columns = ["session_id"] + site_cols + ["user_id"]
    sparseDf = pd.DataFrame(resMat, columns=columns)
    return sparseDf, siteMat


path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week1/"
with open(path + "sites3.json") as f:
    jstr = f.read()
    sites3 = json.loads(jstr)
df3 = pd.DataFrame.from_csv("/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week1/train_data_3users.csv",
                            index_col="session_id")
spDf3, siteMat = to_sparce_df(df3, sites3)
sparceMat = csr_matrix(siteMat)

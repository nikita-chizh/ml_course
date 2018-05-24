import numpy as np
import pandas as pd
import os
import pickle
from scipy.sparse import csr_matrix
from collections import Counter

path = "/Users/nikita/PycharmProjects/ML_Tasks/course6/capstone_user_identification/"
uspath150 = path + "150users/"
uspath10 = path + "10users/"
uspath3 = path + "3users/"

# Сощлось с точностью до перестановки строк
# Код проверки
# X, y = create_sparce(uspath3)
# My
#  [0 2 1 0 2 0 0 0 0 0 0] [2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#  [0 1 0 0 1 0 0 0 0 0 0] [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#  [2 2 0 1 0 0 0 0 0 0 0] [0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 0]
#  [3 1 0 0 0 1 0 0 0 0 0] [0, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0]
#  [1 0 0 2 0 1 1 0 0 0 0] [1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1]
#  [1 1 0 2 0 0 0 0 0 0 0] [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0]
#  [0 1 0 0 0 0 0 0 0 0 0] -
#  [0 3 1 0 0 0 0 1 0 0 0] -
#  [1 1 0 0 0 1 0 1 1 0 0] -
#  [0 0 1 0 0 1 0 0 1 1 1] -
#  [3 0 1 0 0 0 0 0 0 0 1] -
#  [2 0 0 0 0 0 0 0 0 0 0] -
# X_toy_s5_w3.todense()
#
# matrix([[0, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0], 3
#         [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0], 8
#         [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], 9
#         [3, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], 7
#         [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  11
#         [0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 0], 2
#         [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], 1
#         [2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0], 0
#         [3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], 10
#         [1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1], 4
#         [1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0], 5
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) 6

def to_sparce_df(df, site_dict):
    N = df.shape[0]  # кол во сессий
    M = len(site_dict)  # кол во уникальных сайтов
    # индексы не 0х
    row_ind = []
    col_ind = []
    # значения в A(row_ind[i], col_ind[i])
    data = []
    site_cols = ["site" + str(i) for i in range(1, df.shape[1] - 1)]
    for i in range(0, df.shape[0]):
        row = df.iloc[i][site_cols]
        cnt = Counter(row)
        for k, v in cnt.items():
            if k != 0:
                row_ind.append(i)
                col_ind.append(k - 1)
                data.append(v)
    user_col = df[["user_id"]]
    X = csr_matrix((data, (row_ind, col_ind)), shape=(N, M), dtype=int)
    return X, user_col


def create_sparce(path, session_length=5, swindow=3):
    parser = SessionParser(path, session_length=session_length, swindow=swindow)
    parser.parse_dir()
    X, y = to_sparce_df(parser.res, parser.sites)
    X = csr_matrix(X)
    return X, y


class SessionParser:
    def __init__(self, path, session_length=10, swindow=10):
        self.path = path
        self.sites = {}
        self.session_length = session_length
        self.swindow = swindow
        self.max_siteid = 0
        columns = ["site" + str(i) for i in range(1, session_length + 1)]
        columns = ["session_id"] + columns + ["user_id"]
        self.columns = columns
        self.Nsessions_for_user = []
        self.res = None

    def process_site(self, site_name):
        site_p = self.sites.get(site_name)
        if not site_p:
            # 0-id 1-freq
            self.sites[site_name] = [0, 1]
        else:
            site_p[1] += 1

    # more drequent sites have less ids
    def set_site_ids(self):
        site_list = []
        for k, v in self.sites.items():
            site_list.append((k, v[1]))
        site_list = sorted(site_list, key=lambda x: x[1], reverse=True)
        for i in range(len(site_list)):
            cur = site_list[i]
            self.sites[cur[0]][0] = i + 1

    def getids(self, user_df, begin, end):
        sids = []
        if end >= user_df.shape[0]:
            end = user_df.shape[0]
        for i in range(begin, end):
            site = user_df.values[i][0]
            sid = self.sites.get(site)
            sid = sid[0]
            if not sid:
                raise Exception("suka blyat")
            sids.append(sid)
        return sids

    def process_session(self, user_df, row, sbegin):
        # extraction of all sites ids
        send = sbegin + self.session_length
        site_ids = self.getids(user_df, sbegin, send)
        # if session not full append zeros
        if len(site_ids) != self.session_length:
            for i in range(0, self.session_length - len(site_ids)):
                site_ids.append(0)

        for col in zip(range(1, len(site_ids) + 1), site_ids):
            site_pos = col[0]
            self.res.values[row][site_pos] = col[1]

    def one_file(self, u_data_id, start, s_pos):
        # how many sessions this user has
        N = self.Nsessions_for_user[s_pos]
        M = len(self.columns)
        # start and end indexes in self.res for this user sessions
        beg = start
        end = start + N
        sbegin = 0
        for i in range(beg, end):
            # session_id column
            self.res.values[i][0] = i + 1
            # user_id column
            self.res.values[i][M - 1] = u_data_id[1]
            # calculating begin and end for this session in userDataFrame
            self.process_session(u_data_id[0], i, sbegin)
            sbegin += self.swindow

    def parse_dir(self):
        user_dfs = []  #
        N = 0
        for filename in os.listdir(self.path):
            if filename.endswith(".csv"):
                user_df = pd.read_csv(self.path + filename)
                uid = filename.split(".")[0].split("r")[1]
                uid = int(uid)
                user_dfs.append((user_df[["site"]], uid))
                for i in range(user_df.shape[0]):
                    self.process_site(user_df.values[i][1])
                cur_user_sN = user_df.shape[0] // self.swindow
                if user_df.shape[0] % self.swindow != 0:
                    cur_user_sN += 1
                self.Nsessions_for_user.append(cur_user_sN)
                N += cur_user_sN

        self.set_site_ids()
        self.res = pd.DataFrame(np.zeros((N, len(self.columns)), dtype=int), columns=self.columns)
        start = 0
        for i in range(len(user_dfs)):
            self.one_file(user_dfs[i], start, i)
            start += self.Nsessions_for_user[i]

# X, y = create_sparce(uspath3)
# print((X.todense()))
#
PATH_TO_SAVE = "/Users/nikita/PycharmProjects/ML_Tasks/course6/week2/res/res10"
import itertools

data_lengths = []

for path in [uspath10]:
    for window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):
        if window_size <= session_length and (window_size, session_length) != (10, 10):
            X, y = create_sparce(path, session_length, window_size)
            namex = "X" + str(window_size) + "_" + str(session_length) + ".pkl"
            namey = "y" + str(window_size) + "_" + str(session_length) + ".pkl"
            with open(os.path.join(PATH_TO_SAVE, namex), 'wb') as handler:
                pickle.dump(X, handler, protocol=2)
            with open(os.path.join(PATH_TO_SAVE, namey), 'wb') as handler:
                pickle.dump(y, handler, protocol=2)
            data_lengths.append(X.shape[0])
            print(data_lengths)


# X, y = create_sparce(parser3)
# with open(os.path.join(PATH_TO_DATA, 'X.pkl'), 'wb') as X10_pkl:
#     pickle.dump(X, X10_pkl, protocol=2)
#
# with open(os.path.join(PATH_TO_DATA, 'y.pkl'), 'wb') as y_pkl:
#     pickle.dump(y, y_pkl, protocol=2)
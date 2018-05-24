# Реализуйте функцию *prepare_train_set*, которая принимает на вход путь к каталогу
# с csv-файлами *path_to_csv_files* и параметр *session_length* – длину сессии, а возвращает 2 объекта:
# - DataFrame, в котором строки соответствуют уникальным сессиям из *session_length* сайтов, *session_length* столбцов
# – индексам этих *session_length* сайтов и последний столбец – ID пользователя
#
# - частотный словарь сайтов вида {'site_string': [site_id, site_freq]}, например для недавнего игрушечного
# примера это будет {'vk.com': (1, 2), 'google.com': (2, 2), 'yandex.ru': (3, 3), 'facebook.com': (4, 1)}

from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings

warnings.filterwarnings('ignore')
from glob import glob
import os
import json
import pickle
# pip install tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def prepare_train_set(path_to_csv_files, session_length=10):
    pass

path = "/Users/nikita/PycharmProjects/ML_Tasks/course6/capstone_user_identification/"
uspath150 = path + "150users/"
uspath10 = path + "10users/"
uspath3 = path + "3users/"

# сошелся и работает достаточно быстро
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
            sbegin+=self.swindow


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


import sys

#
# parser10 = SessionParser(uspath10)
# parser10.parse_dir()
# s1 = sum(parser10.Nsessions_for_user)
# print(parser10.res.shape)
# sys.stdout.flush()

# # # UniqueSes10= 14061
# # # UniqueSites10= 4913
# #
# # #
parser150 = SessionParser(uspath150, 10, 5)
parser150.parse_dir()
s2 = sum(parser150.Nsessions_for_user)

# print(parser150.res.shape[0])
# sys.stdout.flush()

# # UniqueSes150= 137019
# # UniqueSites150= 27797
parser3 = SessionParser(uspath3)
parser3.parse_dir()
# #
# # site_list = []
# # for key, value in parser150.sites.items():
# #     site_list.append((key, value))
# #
# # site_list = sorted(site_list, key=lambda x: x[1][1], reverse=True)
# # print(site_list[:10])
# #www.linkedin.com
# res_path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week1/"
# # parser10.res.to_csv(res_path + "train_data_10users.csv", index_label='session_id', float_format='%d')
# # parser150.res.to_csv(res_path + "train_data_150users.csv", index_label='session_id', float_format='%d')
# #parser3.res.to_csv(res_path + "train_data_3users.csv", index_label='session_id', float_format='%d')
# def write_json(parser, fname):
#     json_res = json.dumps(parser.sites, ensure_ascii=False)
#     filename = res_path + fname + ".json"
#     f = open(filename, "w+")
#     f.write(json_res)
#
# print("Writing")
# write_json(parser10, "sites10")
# write_json(parser150, "sites150")

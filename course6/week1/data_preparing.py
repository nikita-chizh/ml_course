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
import pickle
# pip install tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def prepare_train_set(path_to_csv_files, session_length=10):
    pass


uspath = "/Users/nikita/PycharmProjects/ML_Tasks/course6/capstone_user_identification/3users/"


class SessionParser:
    def __init__(self, path, session_length):
        self.path = path
        self.sites = {}
        self.session_length = session_length
        self.cur_uid = 1
        self.cur_session = 1
        self.max_siteid = 0
        self.site_cols = []



        new_session = {
            "session_id": [0],
            "user_id": [0],
        }
        for i in range(1, session_length + 1):
            new_session["site" + str(i)] = 0
            self.site_cols.append("site" + str(i))
        self.res = pd.DataFrame(data=new_session)

    def getids(self, sites):
        sids = []
        for site_name in sites.values:
            sid = self.sites.get(site_name[0])
            if not sid:
                self.max_siteid += 1
                self.sites[site_name[0]] = [self.max_siteid, 1]
                sid = self.max_siteid
            else:
                self.sites[site_name[0]][1] += 1
                sid = self.sites[site_name[0]][0]
            sids.append(sid)
        return sids

    def process_session(self, session_df):
        # extraction of all sites ids
        site_ids = self.getids(session_df[["site"]])
        new_session = {
            "session_id": [self.cur_session],
            "user_id": [self.cur_uid],
        }
        if len(site_ids) != self.session_length:
            for i in range(0, self.session_length - len(site_ids)):
                site_ids.append(0)
        for col in zip(self.site_cols, site_ids):
            new_session[col[0]] = [col[1]]
        # incr session counter
        self.cur_session+=1
        new_session = pd.DataFrame(data=new_session)
        self.res = self.res.append(new_session, ignore_index=True)

    def one_file(self, filename):
        user_df = pd.read_csv(self.path + filename)
        rows = user_df.shape[0]
        #сколько сессий в документе
        sessions = rows // self.session_length
        for i in range(0, sessions):
            beg = i * self.session_length
            session_slice_df = user_df.iloc[beg: beg + self.session_length]
            # обработка одной полной сессии
            self.process_session(session_slice_df)
        # последняя, неполная сессия
        last_session = user_df.iloc[self.session_length * sessions:]
        if last_session.shape[0]!=0:
            self.process_session(last_session)

    def parse_dir(self):
        for filename in os.listdir(self.path):
            if filename.endswith(".csv"):
                uid = filename.split(".")[0].split("r")[1]
                uid = int(uid)
                self.cur_uid = uid
                self.one_file(filename)


parser = SessionParser(uspath, 5)
parser.parse_dir()
a = 1
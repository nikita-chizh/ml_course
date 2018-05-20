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


path = "/Users/nikita/PycharmProjects/ML_Tasks/course6/capstone_user_identification/3users"


class SessionParser:
    def __init__(self, path, session_length):
        self.path = path
        self.sites = {}
        self.session_length = session_length
        self.cur_uid = 0

    def one_file(self, filename):
        user_df = pd.read_csv(filename)
        rows = user_df.shape[0]
        sessions = rows // self.session_length
        last_s_length = rows % self.session_length
        for i in range(0, sessions):
            session_slice =

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        print("kek")
    else:
        continue

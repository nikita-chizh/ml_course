from time import time
import itertools
import os
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max.columns', 100)
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

PATH_TO_DATA = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/capstone_user_identification/kaggle_data"
save_path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/data"


train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                       index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
                             index_col='session_id')

train_df = train_df[pd.notnull(train_df['target'])]
train_df.reset_index(inplace=True, drop=True)


# PROCESS
def nan_tozero(data_frame):
    data_frame = data_frame.astype(float).fillna(0.0)
    data_frame = data_frame.astype(int)
    return data_frame

def time_to_unx(df):
    df_c = df.copy()
    def to_utime(time_str):
        try:
            time_str = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except TypeError:
            time_str = int(0)
        return int(time_str)

    for i in range(1, 11):
        c_name = "time" + str(i)
        df_c[c_name] = df_c[c_name].apply(to_utime)
    return df_c

def add_deltas(data_frame, train=True):
    delta_cols = ["delta" + str(i) for i in range(1, 10)]
    new_features = pd.DataFrame(np.zeros((data_frame.shape[0], len(delta_cols)), dtype=int), columns=delta_cols)
    for i in range(1, 10):
        str_i = str(i)
        time_c_name_next = "time" + str(i + 1)
        time_c_name = "time" + str_i
        delta_c_name = "delta" + str_i
        c2 = data_frame[time_c_name_next]
        c1 = data_frame[time_c_name]
        new_features[delta_c_name] = c2 - c1

        def for_z(x):
            if x < 0:
                return 0
            return x

        new_features[delta_c_name] = new_features[delta_c_name].apply(for_z)
    df_proc = pd.concat([data_frame, new_features], axis=1)
    df_proc = df_proc.drop(df_proc.index[[0]])
    return df_proc

def scale(df):
    df = nan_tozero(df)
    cols = list(df.columns.values)
    if "target" in cols:
        cols.remove('target')
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df



def process(df, fname):
    # 1
    df_unx = time_to_unx(df)
    df_unx = nan_tozero(df_unx)
    with open(os.path.join(save_path, fname + "_UNX"'.pkl'), 'wb') as f:
        pickle.dump(df_unx, f, protocol=2)

    # 2
    df_unx_scaled = scale(df_unx)
    with open(os.path.join(save_path, fname + "_UNX_SCALED"'.pkl'), 'wb') as f:
        pickle.dump(df_unx_scaled, f, protocol=2)

    # 3
    df_deltas = add_deltas(df_unx)
    with open(os.path.join(save_path, fname + "_DELTAS"'.pkl'), 'wb') as f:
        pickle.dump(df_deltas, f, protocol=2)

    # 4
    df_deltas_scaled = scale(df_deltas)
    with open(os.path.join(save_path, fname + "_DELTAS_SCALED"'.pkl'), 'wb') as f:
        pickle.dump(df_deltas_scaled, f, protocol=2)

process(test_df, "TEST")
process(train_df, "TRAIN")
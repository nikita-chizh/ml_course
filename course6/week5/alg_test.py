from matplotlib import pyplot as plt
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/"
fname = "BEST_LOG_CV.pkl"
resf_name = "BEST_LOG_CV_res.csv"
df = pd.read_pickle(path + "TEST_SCALED.pkl")



cls = pickle.load(open(path + fname, 'rb'))
predicted_labels = cls.predict_proba(df)
#predicted_labels = np.reshape(predicted_labels,(predicted_labels.shape[0], 1) )
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

write_to_submission_file(predicted_labels[:, 1], resf_name)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
import os
PATH_TO_DATA = "/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/capstone_user_identification/kaggle_data/"
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                       index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
                      index_col='session_id')

f = open(PATH_TO_DATA + "X_train_sparse.pkl", "rb")
X_train_sparse = pickle.load(f)
f = open(PATH_TO_DATA + "X_test_sparse.pkl", "rb")
X_test_sparse = pickle.load(f)
y = train_df['target']
train_share = int(.7 * X_train_sparse.shape[0])
X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]
X_valid, y_valid  = X_train_sparse[train_share:, :], y[train_share:]


from sklearn.model_selection import GridSearchCV
alpha_step = 0.0001 / 5
param_grid = {
    'alpha': [alpha_step + alpha_step * i for i in range(10)], # learning rate
    'n_iter': [1000], # number of epochs
    'loss': ['log'], # logistic regression,
    'penalty': ['l2'],
    'n_jobs': [-1]
}

# grid_search = GridSearchCV(SGDClassifier(), param_grid, cv=4, scoring="f1")
# grid_search.fit(X_train, y_train)
sgd_logit = pickle.load(open("/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/res_1_sgd_logit_TUNED.pkl", 'rb'))
logit_valid_pred_proba = sgd_logit.predict_proba(X_valid)
roc_auc_logit = roc_auc_score(y_valid, logit_valid_pred_proba[:, 1])
print('ROC AUC score: {:.3f}'.format(roc_auc_logit))
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)

write_to_submission_file(logit_valid_pred_proba[:, 1], 'res_1_sgd_logit_TUNED.csv')
pickle.dump(sgd_logit, open("/home/nikita/Desktop/some/help/bl/course/1/2.1/course6/week5/"
                             + "res_1_sgd_logit_TUNED" + ".pkl", 'wb'))
# {'n_jobs': -1, 'alpha': 2e-05, 'n_iter': 1000, 'penalty': 'l2', 'loss': 'log'}
# ROC AUC score: 0.954
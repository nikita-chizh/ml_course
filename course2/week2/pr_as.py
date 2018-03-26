import numpy as np
from matplotlib import pyplot as plt
import seaborn

path = "/home/nikita/Desktop/some/help/bl/course/1/2.1/2_course/week2/"

# рисует один scatter plot
def scatter(actual, predicted, T):
    plt.scatter(actual, predicted)
    plt.xlabel("Labels")
    plt.ylabel("Predicted probabilities")
    plt.plot([-0.2, 1.2], [T, T])


# рисует несколько scatter plot в таблице, имеющие размеры shape
def many_scatters(actuals, predicteds, Ts, titles, shape):
    plt.figure(figsize=(shape[1] * 5, shape[0] * 5))
    i = 1
    for actual, predicted, T, title in zip(actuals, predicteds, Ts, titles):
        ax = plt.subplot(shape[0], shape[1], i)
        ax.set_title(title)
        i += 1
        scatter(actual, predicted, T)


actual_0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
predicted_0 = np.array([0.19015288, 0.23872404, 0.42707312, 0.15308362, 0.2951875,
                        0.23475641, 0.17882447, 0.36320878, 0.33505476, 0.202608,
                        0.82044786, 0.69750253, 0.60272784, 0.9032949, 0.86949819,
                        0.97368264, 0.97289232, 0.75356512, 0.65189193, 0.95237033,
                        0.91529693, 0.8458463])
actual_1 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1.])
predicted_1 = np.array([0.41310733, 0.43739138, 0.22346525, 0.46746017, 0.58251177,
                        0.38989541, 0.43634826, 0.32329726, 0.01114812, 0.41623557,
                        0.54875741, 0.48526472, 0.21747683, 0.05069586, 0.16438548,
                        0.68721238, 0.72062154, 0.90268312, 0.46486043, 0.99656541,
                        0.59919345, 0.53818659, 0.8037637, 0.272277, 0.87428626,
                        0.79721372, 0.62506539, 0.63010277, 0.35276217, 0.56775664])
actual_2 = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
predicted_2 = np.array([0.07058193, 0.57877375, 0.42453249, 0.56562439, 0.13372737,
                        0.18696826, 0.09037209, 0.12609756, 0.14047683, 0.06210359,
                        0.36812596, 0.22277266, 0.79974381, 0.94843878, 0.4742684,
                        0.80825366, 0.83569563, 0.45621915, 0.79364286, 0.82181152,
                        0.44531285, 0.65245348, 0.69884206, 0.69455127])
# рискующий идеальный алгоитм
actual_0r = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
predicted_0r = np.array([0.23563765, 0.16685597, 0.13718058, 0.35905335, 0.18498365,
                         0.20730027, 0.14833803, 0.18841647, 0.01205882, 0.0101424,
                         0.10170538, 0.94552901, 0.72007506, 0.75186747, 0.85893269,
                         0.90517219, 0.97667347, 0.86346504, 0.72267683, 0.9130444,
                         0.8319242, 0.9578879, 0.89448939, 0.76379055])
# рискующий хороший алгоритм
actual_1r = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
predicted_1r = np.array([0.13832748, 0.0814398, 0.16136633, 0.11766141, 0.31784942,
                         0.14886991, 0.22664977, 0.07735617, 0.07071879, 0.92146468,
                         0.87579938, 0.97561838, 0.75638872, 0.89900957, 0.93760969,
                         0.92708013, 0.82003675, 0.85833438, 0.67371118, 0.82115125,
                         0.87560984, 0.77832734, 0.7593189, 0.81615662, 0.11906964,
                         0.18857729])

actual_10 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                      1., 1., 1.])
predicted_10 = np.array([0.29340574, 0.47340035, 0.1580356, 0.29996772, 0.24115457, 0.16177793,
                         0.35552878, 0.18867804, 0.38141962, 0.20367392, 0.26418924, 0.16289102,
                         0.27774892, 0.32013135, 0.13453541, 0.39478755, 0.96625033, 0.47683139,
                         0.51221325, 0.48938235, 0.57092593, 0.21856972, 0.62773859, 0.90454639, 0.19406537,
                         0.32063043, 0.4545493, 0.57574841, 0.55847795])
actual_11 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
predicted_11 = np.array([0.35929566, 0.61562123, 0.71974688, 0.24893298, 0.19056711, 0.89308488,
                         0.71155538, 0.00903258, 0.51950535, 0.72153302, 0.45936068, 0.20197229, 0.67092724,
                         0.81111343, 0.65359427, 0.70044585, 0.61983513, 0.84716577, 0.8512387,
                         0.86023125, 0.7659328, 0.70362246, 0.70127618, 0.8578749, 0.83641841,
                         0.62959491, 0.90445368])
from sklearn.metrics import precision_score, recall_score, accuracy_score

T = 0.5
print("Алгоритмы, разные по качеству:")
for actual, predicted, descr in zip([actual_0, actual_1, actual_2],
                                    [predicted_0 > T, predicted_1 > T, predicted_2 > T],
                                    ["Perfect:", "Typical:", "Awful:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
          recall_score(actual, predicted), ";", \
          "accuracy =", accuracy_score(actual, predicted))

print("Осторожный и рискующий алгоритмы:")
for actual, predicted, descr in zip([actual_1, actual_1r],
                                    [predicted_1 > T, predicted_1r > T],
                                    ["Typical careful:", "Typical risky:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
          recall_score(actual, predicted), ";", \
          "accuracy =", accuracy_score(actual, predicted))

print("Разные склонности алгоритмов к ошибкам FP и FN:")
for actual, predicted, descr in zip([actual_10, actual_11],
                                    [predicted_10 > T, predicted_11 > T],
                                    ["Avoids FP:", "Avoids FN:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
          recall_score(actual, predicted), ";", \
          "accuracy =", accuracy_score(actual, predicted))
plt.show()
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

precs = []
recs = []
threshs = []
labels = ["Typical", "Avoids FP", "Avoids FN"]
for actual, predicted in zip([actual_1, actual_10, actual_11],
                             [predicted_1, predicted_10, predicted_11]):
    prec, rec, thresh = precision_recall_curve(actual, predicted)
    precs.append(prec)
    recs.append(rec)
    threshs.append(thresh)

#################FIRST TASK
T = 0.65
pred = [1 if x > T else 0 for x in predicted_1]
precision_1 = precision_score(actual_1, pred)
recall_1 = recall_score(actual_1, pred)
# second
pred = [1 if x > T else 0 for x in predicted_10]
precision_10 = precision_score(actual_10, pred)
recall_10 = recall_score(actual_10, pred)
# third
pred = [1 if x > T else 0 for x in predicted_11]
precision_11 = precision_score(actual_11, pred)
recall_11 = recall_score(actual_11, pred)


#
def write_answer_1(precision_1, recall_1, precision_10, recall_10, precision_11, recall_11):
    answers = [precision_1, recall_1, precision_10, recall_10, precision_11, recall_11]
    with open(path + "pa_metrics_problem1.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))


write_answer_1(precision_1, recall_1, precision_10, recall_10, precision_11, recall_11)
#
from sklearn.metrics import f1_score


def opt_fscore(actual, predicted, T):
    pred = [1 if x > T else 0 for x in predicted]
    return f1_score(actual, pred)


t = np.linspace(0, 1)
f = np.array([opt_fscore(actual_1, predicted_1, x) for x in t])
plt.plot(t, f, 'r-', lw=5, alpha=0.6, color="red")
plt.ylabel('T')
plt.xlabel('F')
plt.show()
#2
t = np.linspace(0, 1)
f = np.array([opt_fscore(actual_10, predicted_10, x) for x in t])
plt.plot(t, f, 'r-', lw=5, alpha=0.6, color="blue")
plt.ylabel('T')
plt.xlabel('F')
plt.show()
#3
t = np.linspace(0, 1)
f = np.array([opt_fscore(actual_11, predicted_11, x) for x in t])
plt.plot(t, f, 'r-', lw=5, alpha=0.6, color="green")
plt.ylabel('T')
plt.xlabel('F')
plt.show()

#################SECOND TASK
from scipy.optimize import minimize_scalar
maxF = minimize_scalar(lambda x: 1 / opt_fscore(actual_1, predicted_1, x), (0, 1), (0,1), method='Bounded')
k_1 = maxF.x * 10
maxF = minimize_scalar(lambda x: 1 / opt_fscore(actual_10, predicted_10, x), (0, 1), (0,1), method='Bounded')
k_10 = maxF.x * 10
maxF = minimize_scalar(lambda x: 1 / opt_fscore(actual_11, predicted_11, x), (0, 1), (0,1), method='Bounded')
k_11 = maxF.x * 10

def write_answer_2(k_1, k_10, k_11):
    answers = [k_1, k_10, k_11]
    answers = [round(x, 3) for x in answers]
    with open(path + "pa_metrics_problem2.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))
write_answer_2(k_1, k_10, k_11)



#ALL WORKS BELOW
#
# #################THIRD TASK
# from math import log
# #weighted_log_loss(actual,predicted)=−1n∑ni=1(0.3⋅actuali⋅log(predictedi)+0.7⋅(1−actuali)⋅log(1−predictedi))
# def w_log_loss(act, pred):
#     n = act.shape[0]
#     res = 0
#     for i in range(0, n):
#         err = 0.3 * act[i] * log(pred[i]) + 0.7 * (1 - act[i]) * log(1 - pred[i])
#         res = res + err
#     return (-1/n) * res
#
# def write_answer_3(wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11):
#     answers = [wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11]
#     with open(path + "pa_metrics_problem3.txt", "w") as fout:
#         fout.write(" ".join([str(num) for num in answers]))
#
# wll_0 = w_log_loss(actual_0, predicted_0)
# wll_1 = w_log_loss(actual_1, predicted_1)
# wll_2 = w_log_loss(actual_2, predicted_2)
# wll_0r = w_log_loss(actual_0r, predicted_0r)
# wll_1r = w_log_loss(actual_1r, predicted_1r)
# wll_10 = w_log_loss(actual_10, predicted_10)
# wll_11 = w_log_loss(actual_11, predicted_11)
# write_answer_3(wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11)
# #################FOURTH TASK
# from sklearn.metrics import roc_curve, roc_auc_score
# import math
#
# def write_answer_4(T_0, T_1, T_2, T_0r, T_1r, T_10, T_11):
#     answers = [T_0, T_1, T_2, T_0r, T_1r, T_10, T_11]
#     with open(path + "pa_metrics_problem4.txt", "w") as fout:
#         fout.write(" ".join([str(num) for num in answers]))
#
# #from course
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# aucs = ""
# def find_opt_thr(fpr, tpr, thr):
#     n = fpr.shape[0]
#     dist = np.inf
#     res_i = 0
#     for i in range(0, n):
#         new_dist = math.hypot(0 - fpr[i], 1 - tpr[i])
#         if new_dist < dist:
#             res_i = i
#             dist = new_dist
#     return thr[res_i]
#
# thrs = []
# for actual, predicted, descr in zip([actual_0, actual_1, actual_2],
#                                     [predicted_0, predicted_1, predicted_2],
#                                     ["Perfect", "Typical", "Awful"]):
#     fpr, tpr, thr = roc_curve(actual, predicted)
#     #######
#     thrs.append(find_opt_thr(fpr, tpr, thr))
#     #######
#     plt.plot(fpr, tpr, label=descr)
#     aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
#
# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.legend(loc=4)
# plt.margins(0.1, 0.1)
# plt.subplot(1, 3, 2)
# for actual, predicted, descr in zip([actual_0, actual_0r, actual_1, actual_1r],
#                                     [predicted_0, predicted_0r, predicted_1, predicted_1r],
#                                     ["Ideal careful", "Ideal Risky", "Typical careful", "Typical risky"]):
#     fpr, tpr, thr = roc_curve(actual, predicted)
#     #######
#     thrs.append(find_opt_thr(fpr, tpr, thr))
#     #######
#     aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
#     plt.plot(fpr, tpr, label=descr)
# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.legend(loc=4)
# plt.margins(0.1, 0.1)
# plt.subplot(1, 3, 3)
# for actual, predicted, descr in zip([actual_1, actual_10, actual_11],
#                                     [predicted_1, predicted_10, predicted_11],
#                                     ["Typical", "Avoids FP", "Avoids FN"]):
#     fpr, tpr, thr = roc_curve(actual, predicted)
#     #######
#     thrs.append(find_opt_thr(fpr, tpr, thr))
#     #######
#     aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
#     plt.plot(fpr, tpr, label=descr)
# plt.xlabel("false positive rate")
# plt.ylabel("true positive rate")
# plt.legend(loc=4)
# plt.margins(0.1, 0.1)
# print("threshholds", thrs)
# T_0, T_1, T_2, T_0r, T_1r, T_10, T_11 = thrs[0], thrs[1], thrs[2], thrs[4], thrs[6], thrs[8], thrs[9]
# write_answer_4(T_0, T_1, T_2, T_0r, T_1r, T_10, T_11)
#
# plt.show()

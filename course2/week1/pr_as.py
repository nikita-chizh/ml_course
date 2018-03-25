import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt


def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


path = "/Users/nikita/PycharmProjects/ML_Tasks/course2/week1/"

df = pd.read_csv(path + "advertising.csv")
X = df[["TV", "Radio", "Newspaper"]].values
Y = df["Sales"].values


#
def scale(array):
    means, stds = np.mean(array), np.std(array)
    n = array.shape[0]
    for i in range(0, n):
        array[i] = (array[i] - means) / stds


scale(X[:, 0])
scale(X[:, 1])
scale(X[:, 2])
n = X.shape[0]
one_v = np.linspace(1, 1, n)
one_v = one_v[np.newaxis, :].T
X = np.concatenate((X, one_v), axis=1)


#
def mserror(y, y_pred):
    mse = ((y - y_pred) ** 2).mean()
    return mse


sales_mean = np.median(Y)
Y_mean = np.linspace(sales_mean, sales_mean, n)
answ1 = mserror(Y, Y_mean)
write_answer_to_file(answ1, path + '1.txt')


# 2
def normal_equation(X, y):
    pseudo_inv = np.linalg.pinv(X)
    return np.matmul(pseudo_inv, y)


W_theory = normal_equation(X, Y)
# mean answer
mean_X = np.array([np.mean(X[:, 0]), np.mean(X[:, 1])
                      , np.mean(X[:, 2]), 1])

answ2 = np.dot(mean_X, W_theory)
write_answer_to_file(answ2, path + '2.txt')


# 3
def linear_prediction(X, W):
    return np.matmul(X, W)


Y_norm_pred = linear_prediction(X, W_theory)
answ3 = mserror(Y, Y_norm_pred)
write_answer_to_file(answ3, path + '3.txt')


# 4
def stochastic_gradient_step(X, y, w, k_idx, eta=0.01):
    m = X.shape[1]
    n = X.shape[0]
    gradient = np.ones(m)
    cur_X = X[k_idx]
    for j in range(0, m):
        grad = cur_X[j] * (np.dot(cur_X, w) - y[k_idx])
        gradient[j] = grad * 2 / n
    return w - eta * gradient


def stochastic_gradient_descent(X, y, w_init, max_iter=1e5, min_weight_dist=1e-9, seed=42):
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    np.random.seed(seed)
    n = X.shape[0]
    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # step
        random_ind = np.random.randint(n)
        w_new = stochastic_gradient_step(X, y, w, random_ind)
        weight_dist = np.linalg.norm(w - w_new)
        #update
        w = w_new
        iter_num = iter_num + 1
        # calc error
        pred = linear_prediction(X[random_ind], w)
        errors.append(mserror(y[random_ind], pred))
    return w, errors


w_init = np.array([0] * 4)
W_grad, errors = stochastic_gradient_descent(X, Y, w_init)
print(W_grad, W_theory)
Y_pred = linear_prediction(X, W_grad)
answ4 = mserror(Y, Y_pred)
write_answer_to_file(answ4, path + '4.txt')


plt.plot(range(50), errors[:50])
plt.show()

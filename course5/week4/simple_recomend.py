train = open("/home/nikita/Desktop/some/help/bl/course/1/2.1/course5/week4/coursera_sessions_train.txt", "r").read()
test = open("/home/nikita/Desktop/some/help/bl/course/1/2.1/course5/week4/coursera_sessions_test.txt", "r").read()

# Выгрузка данных
def extract_data(file):
    file = file.split("\n")
    views_all = []
    purchases_all = []
    for line in file:
        split = line.split(";")
        views = split[0].split(",")
        # избавляемся от повторов в просмотрах
        views = list(set(views))
        views = [int(v) for v in views]
        purchases = []
        if split[1]:
            purchases = split[1].split(",")
            purchases = [int(p) for p in purchases]
        views_all += views
        purchases_all += purchases
    return views_all, purchases_all

train_views, train_purchases = extract_data(train)
test_views, test_purchases = extract_data(test)
# Расчет частот
from collections import Counter
# На обучении постройте частоты появления id в просмотренных и в купленных (id может несколько раз появляться
# в просмотренных, все появления надо учитывать)
c1=Counter(train_views)
c2=Counter(train_purchases)
train_views_freq = []
for key in c1:
    train_views_freq.append((key, c1[key]))
train_purch_freq=[]
for key in c2:
    train_purch_freq.append((key, c2[key]))

train_purch_freq = sorted(train_purch_freq, key=lambda x: x[1])
train_views_freq = sorted(train_views_freq, key=lambda x: x[1])
import json

path = "/Users/nikita/PycharmProjects/ML_Tasks/course3/week4/"


def save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs):
    with open(path + "cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs]]))


with open(path + "recipes.json", "r", encoding="utf-8") as f:
    content = f.read()
    recipes = json.loads(content)
from gensim import corpora, models
import numpy as np
np.random.seed(76543)
texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts)  # составляем словарь
token_ids = dictionary.token2id
corpus = [dictionary.doc2bow(text) for text in texts]  # составляем корпус документов
targets = ["salt", "sugar", "water", "mushrooms", "chicken", "eggs"]
trgt_ids = {}
for target in targets:
    tid = token_ids[target]
    trgt_ids[target] = str(tid)

#
lda = models.LdaMulticore(corpus, id2word=dictionary, num_topics=40, passes=5, workers=4)
topics = lda.show_topics(num_topics=40, formatted=False)

# Task 1
def calc_tpc(tcs, id):
    num = 0
    for topic in tcs:
        for them in topic[1]:
            if id == them[0]:
                num = num + 1
    return num
c_salt = calc_tpc(topics, trgt_ids["salt"])
c_sugar = calc_tpc(topics, trgt_ids["sugar"])
c_water = calc_tpc(topics, trgt_ids["water"])
c_mushrooms = calc_tpc(topics, trgt_ids["mushrooms"])
c_chicken = calc_tpc(topics, trgt_ids["chicken"])
c_eggs = calc_tpc(topics, trgt_ids["eggs"])
save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs)

# Task 2
def save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after):
    with open(path + "cooking_LDA_pa_task2.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [dict_size_before, dict_size_after, corpus_size_before, corpus_size_after]]))
freq_ids = []
for id, num in  dictionary.dfs.items():
    if num > 4000:
        freq_ids.append(id)

def corpsize(crps):
    unids = 0
    for c in crps:
        unids = unids + len(c)
    return unids

dict_size_before = len(dictionary.dfs)
corpus_size_before = corpsize(corpus)
#after
import copy
dictionary2 = copy.deepcopy(dictionary)
dictionary2.filter_tokens(freq_ids)
dict_size_after = len(dictionary2.dfs)
corpus2 = [dictionary2.doc2bow(text) for text in texts]
corpus_size_after = corpsize(corpus2)
save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after)

# Task 3
def save_answers3(coherence, coherence2):
    with open(path + "cooking_LDA_pa_task3.txt", "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))

lda2 = models.LdaMulticore(corpus2, id2word=dictionary2, num_topics=40, passes=5, workers=4)
ttpcs = lda.top_topics(corpus)
coherence = np.mean( [coh[1] for coh in ttpcs] )
ttpcs2 = lda2.top_topics(corpus2)
coherence2 = np.mean( [coh[1] for coh in ttpcs2] )
save_answers3(coherence, coherence2)

#Task 4
def save_answers4(count_model2, count_model3):
    with open(path + "cooking_LDA_pa_task4.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [count_model2, count_model3]]))

np.random.seed(76543)
# здесь код для построения модели:
# обучение модели
lda3 = models.LdaMulticore(corpus2, id2word=dictionary2, alpha=1, num_topics=40, passes=5, workers=1)
count_model2 = 0
count_model3 = 0
for doc in corpus2:
    count_model2 += len(lda2.get_document_topics(doc, minimum_probability=0.01))
    count_model3 += len(lda2.get_document_topics(doc, minimum_probability=0.01))
save_answers4(count_model2, count_model3)

# Task 5
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
def save_answers5(accuracy):
    with open(path + "cooking_LDA_pa_task5.txt", "w") as fout:
        fout.write(str(accuracy))
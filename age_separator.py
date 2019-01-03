from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from transliterate import translit
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

labels = []
data_dict = {}
data = []
label = []
vectorizer = CountVectorizer()
with open('spisok', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels.append(line_no_spaces[:-1])
            continue

        for i in range(len(line)):
            if line[i] == ' ':
                continue
            if line[i] == '.':
                line = line[i+2:]
                break

        if len(labels) == 2 or len(labels) == 7:
            data_dict[translit(u"{}".format(line[:-1]), "ru", reversed=True)] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            label.append(0)
            continue
        data_dict[line[:-1]] = len(labels)
        data.append(line[:-1])
        label.append(0)

for dir in labels:
    if dir in os.listdir('truck-link/WWII'):
        for text in os.listdir('truck-link/WWII/{0}'.format(dir)):
            data.append(text)
            label.append(0)

with open('modern_tech', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels.append(line_no_spaces[:-1])
            continue

        for i in range(len(line)):
            if line[i] == ' ':
                continue
            if line[i] == '.':
                line = line[i+2:]
                break


        data_dict[translit(u"{}".format(line[:-1]), "ru", reversed=True)] = len(labels)
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
        label.append(1)

for dir in labels:
    if dir in os.listdir('truck-link/Modern'):
        for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
            data.append(text)
            label.append(1)

data_vectorised = vectorizer.fit_transform(data)
print(labels)
n_label = np.array(label)


trainX, testX, trainY, testY = train_test_split(data_vectorised, n_label, test_size = 0.1, random_state = 42)
nb = Pipeline([

                ('clf', MultinomialNB(alpha=0.2)),
                #('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=3, tol=None)),
              ])
nb.fit(trainX, trainY)
y_pred = nb.predict(testX)
a = nb.predict(vectorizer.transform(['MAZ-537 G']))
print(a)

print('accuracy %s' % metrics.accuracy_score(y_pred, testY))


print(classification_report(testY, nb.predict(testX), target_names=['modern', 'old']))
#pickle.dump(nb, open('age_model', 'wb'))
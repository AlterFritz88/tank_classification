from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from transliterate import translit
import pickle
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

labels = []
data_dict = {}
data = []
label = []
vectorizer = TfidfVectorizer()
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

        data_dict[line[:-1]] = len(labels)
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True).lower())
        label.append(len(labels) - 1)

data_vectorised = vectorizer.fit_transform(data)
print(labels)
n_label = np.array(label)

# добавляем подкрепление

for dir in labels:
    if dir in os.listdir('truck-link/Modern'):
        for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
            data.append(text.lower())
            label.append(labels.index(dir) + 1)




trainX, testX, trainY, testY = train_test_split(data_vectorised, n_label, test_size = 0.3, random_state = 42, stratify=n_label)

nb_classifier = MultinomialNB(alpha=0.3)
nb_classifier.fit(trainX, trainY)
pred = nb_classifier.predict(testX)
score = metrics.accuracy_score(testY, pred)
print(score)
a = nb_classifier.predict(vectorizer.transform(['Mk II']))
print(a)
print()

from sklearn.linear_model import SGDClassifier

nb = Pipeline([
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=3, tol=None)),
              ])
nb.fit(trainX, trainY)
y_pred = nb.predict(testX)
a = nb_classifier.predict(vectorizer.transform(['AVGP Cougar']))
print(a)

print('accuracy %s' % metrics.accuracy_score(y_pred, testY))


print(classification_report(testY, nb.predict(testX), target_names=labels))

#filename = 'wwii'
#pickle.dump(nb, open(filename, 'wb'))

model = RandomForestClassifier(criterion='gini', max_depth=18, max_features='auto', n_estimators=100)
model.fit(trainX, trainY)
y_pred = nb.predict(testX)
print('accuracy RR %s' % metrics.accuracy_score(y_pred, testY))

param_grid = {'n_estimators': [1, 100], 'max_features': ['auto', 'log2'], 'max_depth': [4, 18], 'criterion': ['gini', 'entropy']}

# Define the model to use
model = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# Fit the model to our training data and obtain best parameters
CV_model.fit(trainX, trainY)
print(CV_model.best_params_)
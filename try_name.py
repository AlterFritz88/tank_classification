from keras.models import load_model
from sklearn.feature_extraction.text import  TfidfVectorizer
import os
from transliterate import translit

model = load_model('modern_names')

labels_modern = []
with open('modern_tech', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_modern.append(line_no_spaces[:-1])
            continue


data = []
vectorizer = TfidfVectorizer()
with open('modern_tech', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            continue

        for i in range(len(line)):
            if line[i] == ' ':
                continue
            if line[i] == '.':
                line = line[i+2:]
                break
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
# добавляем подкрепление
for dir in labels_modern:
    if dir in os.listdir('truck-link/Modern'):
        for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
            data.append(text)

data_vectorised = vectorizer.fit_transform(data)
t_label = tuple(labels_modern)
t_label = model.predict(vectorizer.transform(['BMP-2']))[0]
print(labels_modern)
print(labels_modern[model.predict(vectorizer.transform(['BMP-2'])).argmax(axis=1)[0] - 1])
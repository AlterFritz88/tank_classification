import requests
from lxml.html import fromstring
import os
import random as rd
from names_models import *



#model_age = pickle.load(open('age_model', 'rb'))
#model_WWII = pickle.load(open('wwii', 'rb'))
#model_modern = pickle.load(open('modern', 'rb'))

model_WWII, vectorizer_wwii = fitter_WWII()
model_modern, vectorizer_modern = fitter_modern()
model_age, vectorizer_age = fitter_age()


labels_wwII = []
with open('spisok', 'r') as file:
    for line in file:
        if len(line) < 3:
            continue
        line_no_spaces = line.replace(' ', '')
        try:
            start = int(line_no_spaces[0])
        except:
            labels_wwII.append(line_no_spaces[:-1])
            continue

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






for model_page in range(8250, 11500, 1):
    print(model_page)

    url = "https://www.track-link.com/gallery/{}".format(model_page)
    r = requests.get(url)
    tree = fromstring(r.content)
    path = ' '.join(tree.findtext('.//title').split(' ')[4:]).replace(r'/', ' ')
    print(path)
    if len(path) < 3:      # если в шапке пусто, то скипаем, чтобы не мусорить
        continue

    what_age = model_age.predict(vectorizer_age.transform([path])).argmax(axis=1)[0]
    if what_age == 0:
        age = 'WWII'
        nation = labels_wwII[model_WWII.predict(vectorizer_wwii.transform([path])).argmax(axis=1)[0] - 1]
    else:
        age = 'Modern'
        nation = labels_modern[model_modern.predict(vectorizer_modern.transform([path])).argmax(axis=1)[0] - 1]
    print(age, nation)
    print()




    dirName = 'truck-link/{0}/{1}/{2}'.format(age, nation, path)

    if not os.path.exists('truck-link/{0}/{1}'.format(age, nation)):
        os.mkdir('truck-link/{0}/{1}'.format(age, nation))


    if not os.path.exists(dirName):
        os.mkdir(dirName)

    try:
        for i in range(20):
            image_url = "https://www.track-link.com/gallery/images/b_{0}_{1}.jpg".format(model_page, i)
            r = requests.get(image_url)
            filename = 'truck-link/{0}/{1}/{2}/{3}.jpeg'.format(age, nation, path, i + rd.randint(10, 100000))

            if os.path.isfile(filename):
                filename = 'truck-link/{0}/{1}/{2}/{3}.jpeg'.format(age, nation, path, i + rd.randint(10, 100000))
                with open(filename, 'wb') as f:
                    f.write(r.content)
            else:

                with open(filename, 'wb') as f:
                    f.write(r.content)
    except:
        continue

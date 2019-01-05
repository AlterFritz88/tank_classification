import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import  TfidfVectorizer
import random as rd
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

'''
кластеризует собранные данные по папкам и удаляет пустные
важно знать на сколько видов делить!!!!
работать с бэкапом данных!!!!
'''
model = load_model('models/tankNotTank_test.h5f')
li_dir = os.listdir(path="truck-link/sample")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(li_dir)
kmeans = KMeans(n_clusters=2).fit(X)
print(kmeans.labels_)

dick_paths = {}
for i in range(len(li_dir)):
    if kmeans.labels_[i] not in dick_paths.keys():
        dick_paths[kmeans.labels_[i]] = []
        dick_paths[kmeans.labels_[i]].append(li_dir[i])
    else:
        dick_paths[kmeans.labels_[i]].append(li_dir[i])
print(dick_paths)

for key, value in dick_paths.items():

    for i in range(len(value)):
        if i == 0:
            temp_path = 'truck-link/sample/{0}/'.format(value[0])
            print('into', temp_path)

        else:
            for item in os.listdir('truck-link/sample/{0}'.format(value[i])):
                image = cv2.imread('truck-link/sample/{0}/{1}'.format(value[i], item))
                image = cv2.resize(image, (64, 64))
                image = image.astype("float") / 255.0
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                list_of_pred = model.predict(image)[0]

                sourse = 'truck-link/sample/' + value[i] + '/' + item
                destin = temp_path + str(rd.randint(10,10000000)) + '.jpeg'
                if os.stat(sourse).st_size < 25000 or list_of_pred[0] > 0.6:
                    os.remove(sourse)
                else:
                    os.rename(sourse,  destin)
            os.rmdir('truck-link/sample/{0}'.format(value[i]))
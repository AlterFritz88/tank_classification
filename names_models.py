from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.pipeline import Pipeline
from transliterate import translit
import pickle
from keras.utils import to_categorical
from sklearn.feature_extraction.text import  TfidfVectorizer
import os
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


def fitter_modern():
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
                    line = line[i + 2:]
                    break

            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            label.append(len(labels))
    # добавляем подкрепление
    for dir in labels:
        if dir in os.listdir('truck-link/Modern'):
            for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
                data.append(text)
                label.append(labels.index(dir) + 1)

    data_vectorised = vectorizer.fit_transform(data)
    print(data_vectorised.shape)
    print(data_vectorised)
    n_label = np.array(label)
    print(n_label.shape)

    trainX, testX, trainY, testY = train_test_split(data_vectorised.toarray(), n_label, test_size=0.1, random_state=42)

    trainY = to_categorical(trainY, num_classes=len(labels) + 1)
    testY = to_categorical(testY, num_classes=len(labels) + 1)

    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units=164, activation='relu'))
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(len(labels) + 1))
    model.add(Activation("softmax"))

    checpoint = ModelCheckpoint('models/modern_names.h5f', monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [checpoint]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY), callbacks=callbacks)

    return load_model('models/modern_names.h5f'), vectorizer


def fitter_WWII():
    labels = []
    data_dict = {}
    data = []
    label = []
    vectorizer = TfidfVectorizer()
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
                    line = line[i + 2:]
                    break

            data_dict[line[:-1]] = len(labels)
            data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
            label.append(len(labels))
    # добавляем подкрепление
    for dir in labels:
        if dir in os.listdir('truck-link/WWII'):
            for text in os.listdir('truck-link/WWII/{0}'.format(dir)):
                data.append(text)
                label.append(labels.index(dir) + 1)

    data_vectorised = vectorizer.fit_transform(data)
    print(data_vectorised.shape)
    print(data_vectorised)
    n_label = np.array(label)
    print(n_label.shape)

    trainX, testX, trainY, testY = train_test_split(data_vectorised.toarray(), n_label, test_size=0.1, random_state=42)

    trainY = to_categorical(trainY, num_classes=len(labels) + 1)
    testY = to_categorical(testY, num_classes=len(labels) + 1)

    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units=164, activation='relu'))
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(len(labels) + 1))
    model.add(Activation("softmax"))


    checpoint = ModelCheckpoint('models/WWII_names.h5f', monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [checpoint]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=12, validation_data=(testX, testY), callbacks=callbacks)

    return load_model('models/WWII_names.h5f'), vectorizer

def fitter_age():
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
                    line = line[i + 2:]
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
                    line = line[i + 2:]
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
    print(data_vectorised.shape)
    print(data_vectorised)
    n_label = np.array(label)
    print(n_label.shape)

    trainX, testX, trainY, testY = train_test_split(data_vectorised.toarray(), n_label, test_size=0.1,
                                                    random_state=42)

    trainY = to_categorical(trainY, num_classes=len(labels) + 1)
    testY = to_categorical(testY, num_classes=len(labels) + 1)

    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(units=164, activation='relu'))
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(len(labels) + 1))
    model.add(Activation("softmax"))

    checpoint = ModelCheckpoint('models/age_names.h5f', monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [checpoint]

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=6, validation_data=(testX, testY), callbacks=callbacks)

    return load_model('models/age_names.h5f'), vectorizer
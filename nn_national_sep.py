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
        data.append(translit(u"{}".format(line[:-1]), "ru", reversed=True))
        label.append(len(labels))
# добавляем подкрепление
for dir in labels:
    if dir in os.listdir('truck-link/Modern'):
        for text in os.listdir('truck-link/Modern/{0}'.format(dir)):
            data.append(text)
            label.append(labels.index(dir) + 1)

data_vectorised = vectorizer.fit_transform(data)
n_label = np.array(label)


trainX, testX, trainY, testY = train_test_split(data_vectorised.toarray(), n_label, test_size = 0.3, random_state = 42)

trainY = to_categorical(trainY, num_classes=len(labels)+1)
testY = to_categorical(testY, num_classes=len(labels)+1)



from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
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
model.add(Dense(len(labels)+1))
model.add(Activation("softmax"))

epochs = 20
checpoint = ModelCheckpoint('models/modern_names.h5f', monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [checpoint]
opt = SGD(lr=10, decay=1/epochs, momentum=0.9, nesterov=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=epochs, validation_data=(testX, testY), callbacks=callbacks, verbose=1)

model = load_model('models/modern_names.h5f')
score, acc = model.evaluate(testX, testY)
print(score, acc)


prediction = model.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction))

model.save('modern_names')
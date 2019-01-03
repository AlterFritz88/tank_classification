from imutils import paths
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)




data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("dataset_for_models")))
w, h = 40, 40
preproc = SimplePreprocessor(w,h)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = preproc.preprocess(image)
    print(imagePath)
    data.append(image)


    # extract the class label from the image path and update the
	# labels list
    label = imagePath.split(os.path.sep)[-2]
    if label == "model":
        label = 1
    else:
        label = 0
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
data = data.reshape(data.shape[0], h * w * 3)
labels = np.array(labels)
print(data.shape)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size = 0.2, random_state = 42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

neib = range(1, 10)
for n in neib:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(trainX, trainY)

    pred = model.predict(testX)
    score = metrics.accuracy_score(testY, pred)
    print(n, score)
    print()



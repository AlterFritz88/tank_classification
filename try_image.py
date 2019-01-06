from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
dataset_path = 'tankNotTank'
image = cv2.imread("st.jpeg")
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model('models/{0}_test.h5f'.format(dataset_path))


labels_dict = {}
with open('models/{0}_dict_labels'.format(dataset_path), 'r') as file:
    for line in file:
        if len(line) > 3:
            items = line[:-1].split(' ')
            labels_dict[int(items[1])] = items[0]

list_of_pred = model.predict(image)[0]
print(list_of_pred)
top = sorted(range(len(list_of_pred)), key=lambda i: list_of_pred[i], reverse=True)[:5]
for i in top:
    print(labels_dict[i], list_of_pred[i])
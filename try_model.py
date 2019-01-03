from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

image = cv2.imread("st.jpeg")
image = cv2.resize(image, (48, 48))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model('model_for_models')

(not_model, model) = model.predict(image)[0]

print('')
print("not model", not_model)
print('model', model)

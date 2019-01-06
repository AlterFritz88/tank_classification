from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import time
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
from color_detec import get_color
import random as rd

# construct the argument parser and parse the arguments

model = load_model('model_tanks')

labels_dict = {}
with open('dict_labels', 'r') as file:
    for line in file:
        if len(line) > 3:
            items = line[:-1].split(' ')
            labels_dict[int(items[1])] = items[0]

# load the image and define the window width and height
image = cv2.imread('st.jpeg')
image_result = image.copy()
color_list = []
detected_list = []

(winW, winH) = (min(image.shape[:2]), min(image.shape[:2]))
m = 0
model_frame_size = 64
for r in range(0, 20):
    (winW, winH) = (winW - 50, winH - 50)
    if winH < model_frame_size / 2:
        break
    # for resized in pyramid(image, scale=1.01):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        if get_color(window) > 35:
            continue
        im = cv2.resize(window, (model_frame_size, model_frame_size))

        im = im.astype("float") / 255.0
        im = img_to_array(im)
        im = np.expand_dims(im, axis=0)

        list_of_pred = model.predict(im)[0]
        top = sorted(range(len(list_of_pred)), key=lambda i: list_of_pred[i], reverse=True)[:2]
        print(labels_dict[top[0]], list_of_pred[top[1]])

        if list_of_pred[top[1]] > 0.43:
            print(labels_dict[top[0]], list_of_pred[top[1]])

            cv2.imwrite('sample{0}_{1}.jpeg'.format(m, labels_dict[top[0]]), window)
            cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 255, 0), cv2.FILLED)
            color = (rd.randint(1, 255), rd.randint(1, 255), rd.randint(1, 255))
            color_list.append(color)
            detected_list.append((labels_dict[top[0]], list_of_pred[top[1]]))
            cv2.rectangle(image_result, (x, y), (x + winW, y + winH), color, 2)
            m = m + 1

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW

        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2, cv2.FILLED)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)

legend = np.zeros((image.shape[0], 300, 3), dtype="uint8")
legend[:] = (255, 255, 255)
for i in range(len(color_list)):
    cv2.putText(legend, detected_list[i][0] + ' ' +  str(detected_list[i][1])[:6], (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
    cv2.rectangle(legend, (200, (i * 25)), (300, (i * 25) + 25),
                  tuple(color_list[i]), -1)
cv2.imwrite('legend.jpeg', legend)
image_result = np.concatenate((image_result, legend), axis=1)
cv2.imwrite('result_image.jpeg', image_result)

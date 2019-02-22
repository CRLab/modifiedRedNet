from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

model = load_model('./models/rgbd.02-0.72.hdf5')

model.compile(loss = "categorical_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

for image in os.listdir('data/classifier_data/train/depth/forward/'):

	depth_img = cv2.imread('data/classifier_data/train/depth/forward/' + image)
	depth_img = cv2.resize(depth_img,(256,256))
	depth_img = np.reshape(depth_img,[1,256,256,3])

	rgb_img = cv2.imread('data/classifier_data/train/rgb/forward/' + image.replace("depth", "rgb"))
	rgb_img = cv2.resize(rgb_img,(256,256))
	rgb_img = np.reshape(rgb_img,[1,256,256,3])

	classes = model.predict([rgb_img, depth_img])
	label = ["forward", "left", "right"][np.argmax(classes[0])]

	plt.close()

	fig=plt.figure()
	ax = fig.add_subplot(1,2,1)
	plt.imshow(cv2.imread('data/classifier_data/train/depth/forward/' + image))
	ax.set_title(label)
	ax = fig.add_subplot(1,2,2)
	ax.set_title(label)
	plt.imshow(cv2.imread('data/classifier_data/train/rgb/forward/' + image.replace("depth", "rgb")))
	
	plt.pause(0.5)

plt.show()

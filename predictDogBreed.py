import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show

import cv2
import pickle


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3


### -- ATTRIBUTES
img_width, img_height = 224, 224
model_inception = InceptionV3(include_top=False, weights='imagenet', input_shape = (img_width, img_height, 3))


def load_model() :
	print("[INFO] loading dog prediction model ...")  
	model = model_from_json(open('./saved_model/top_model_architecture.json').read())
	model.load_weights('./saved_model/top_model_weights.h5')
	return model


def load_BreedIndexValues() :
	print("[INFO] loading breed names dictionary ...")  
	with open('./saved_model/breedNamesIndex.pkl', 'rb') as f:
		breedNamesIndex = pickle.load(f)
	return breedNamesIndex

def display_image(image_path) :
	orig_img = cv2.imread(image_path)
	imshow(cv2.cvtColor(orig_img, cv2.CV_32S))
	plt.show()


def load_process_image(image_path) :	
	print("[INFO] loading and preprocessing image...")  
	image = load_img(image_path, target_size=(img_width, img_height))  
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0) 
	image = inception_v3.preprocess_input(image)
	return image

def predict_dog_class(image, model) :
	bottleneck_prediction = model_inception.predict(image)
	class_predicted = model.predict_classes(bottleneck_prediction)
	return class_predicted[0]


## -- MAIN --

if __name__=="__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--image", help="path to image")
	args = a.parse_args()
	if args.image is None:
		a.print_help()
		sys.exit(1)


breedIndexNamesDict = load_BreedIndexValues()
dog_model = load_model()
image = load_process_image(args.image)
dog_class = predict_dog_class(image, dog_model)
print("------------------------------------------------")
print("[RESULT] Le ti matou trouvé est : ", breedIndexNamesDict[dog_class])
display_image(args.image)
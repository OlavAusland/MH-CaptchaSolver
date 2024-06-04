import os

import cv2
import keras.models
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

INPUT_SHAPE = (96, 71, 3)
CHAR_SIZE = (20, 30)

bbox_model: keras.models.Model = keras.models.load_model('./bbox_model.h5')
char_model: keras.models.Model = keras.models.load_model('./character_model.h5')


parser = ArgumentParser()
parser.add_argument('--path', type=str, help='Path to image')
parser.add_argument('--verbose', action='store_true', help='Extra information about prediction time & etc')
args = parser.parse_args()

def predict(path: str):
	original_image = cv2.imread(path) / 255.0
	original_image = original_image[12:71 + 12, 7:96 + 7]

	image = original_image.copy()

	predictions = bbox_model.predict([image.reshape(-1, 96, 71, 3)], verbose=args.verbose)

	result: str = ""

	for prediction in predictions:
		x_min: int = int(prediction[0][0] * INPUT_SHAPE[0])
		y_min: int = int(prediction[0][1] * INPUT_SHAPE[1])
		x_max: int = int(prediction[0][2] * INPUT_SHAPE[0])
		y_max: int = int(prediction[0][3] * INPUT_SHAPE[1])

		character = original_image[y_min+1:y_max, x_min+1:x_max]
		character = cv2.resize(character, dsize=CHAR_SIZE, interpolation=cv2.INTER_AREA)
		character_prediction = char_model.predict([character.reshape(-1, 20, 30, 3)], verbose=args.verbose)

		result += chr(character_prediction.argmax() + 65)
	
	return result


def main():
	if args.path is not None:
		print(predict(args.path))

if __name__ == '__main__':
	main()


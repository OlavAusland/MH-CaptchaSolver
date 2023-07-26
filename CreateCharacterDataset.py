from tensorflow import keras
import numpy as np
import cv2
import os

INPUT_SHAPE = (96, 71, 3)
IMAGE_SIZE = ()


def create_dataset():
    model: keras.Model = keras.models.load_model('./bbox_model.h5')

    for i, file in enumerate(os.listdir('./annotated/')):
        image: np.ndarray = cv2.imread(f'./annotated/{file}')[12:71 + 12, 7:96 + 7]

        predictions = model.predict([(image/255.0).reshape(-1, 96, 71, 3)])

        for j, prediction in enumerate(predictions):
            x_min = int(prediction[0][0] * INPUT_SHAPE[0])
            y_min = int(prediction[0][1] * INPUT_SHAPE[1])
            x_max = int(prediction[0][2] * INPUT_SHAPE[0])
            y_max = int(prediction[0][3] * INPUT_SHAPE[1])

            cv2.imwrite(f'./characters/{file[j]} {i}_{j}.png', image[y_min:y_max, x_min:x_max])


def main():
    create_dataset()


if __name__ == '__main__':
    main()

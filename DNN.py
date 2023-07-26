from tensorflow.python import keras
from tensorflow.python.keras import layers, Model
import json
import os
import cv2
import numpy as np
import random

INPUT_SHAPE = (96, 71, 3)
CHAR_SHAPE = (20, 30, 3)
EPOCH_SIZE = 14
BATCH_SIZE = 8


def load_bbox_dataset():
    first_character = []
    middle_character = []
    last_character = []

    data_x = []

    with open('bbox.json', 'r+') as f:
        data = json.load(f)

        for key, val in data.items():
            image = cv2.imread(f'./annotated/{key}') / 255.0
            image = image[12:71 + 12, 7:96 + 7]

            first_character.append(np.array([val[0]['x_min'] / float(INPUT_SHAPE[0]),
                      val[0]['y_min'] / float(INPUT_SHAPE[1]),
                      val[0]['x_max'] / float(INPUT_SHAPE[0]),
                      val[0]['y_max'] / float(INPUT_SHAPE[1])]))

            middle_character.append(np.array([val[1]['x_min'] / float(INPUT_SHAPE[0]),
                      val[1]['y_min'] / float(INPUT_SHAPE[1]),
                      val[1]['x_max'] / float(INPUT_SHAPE[0]),
                      val[1]['y_max'] / float(INPUT_SHAPE[1])]))

            last_character.append(np.array([val[2]['x_min'] / float(INPUT_SHAPE[0]),
                      val[2]['y_min'] / float(INPUT_SHAPE[1]),
                      val[2]['x_max'] / float(INPUT_SHAPE[0]),
                      val[2]['y_max'] / float(INPUT_SHAPE[1])]))

            data_x.append(image)
    print(np.asarray(first_character))
    return np.asarray(data_x), [np.asarray(first_character), np.asarray(middle_character), np.asarray(last_character)]


def create_bbox_model():
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    x = layers.Flatten()(x)

    # leftmost character
    l_bbox = layers.Dense(2048, activation='relu')(x)
    l_bbox = layers.Dropout(0.2)(l_bbox)
    l_bbox = layers.Dense(2048, activation='relu')(l_bbox)
    l_bbox = layers.Dense(4, activation='sigmoid')(l_bbox)

    # middle character
    m_bbox = layers.Dense(2048, activation='relu')(x)
    m_bbox = layers.Dropout(0.2)(m_bbox)
    m_bbox = layers.Dense(2048, activation='relu')(m_bbox)
    m_bbox = layers.Dense(4, activation='sigmoid')(m_bbox)

    # rightmost character
    r_bbox = layers.Dense(2048, activation='relu')(x)
    r_bbox = layers.Dropout(0.2)(r_bbox)
    r_bbox = layers.Dense(2048, activation='relu')(r_bbox)
    r_bbox = layers.Dense(4, activation='sigmoid')(r_bbox)

    model = Model(inputs=inputs, outputs=[l_bbox, m_bbox, r_bbox])

    return model


def train_bbox_model(model: keras.Model, data_x, data_y):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x=data_x, y=data_y, epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, validation_split=0.1)

    keras.models.save_model(model, 'bbox_model.h5')


def load_character_dataset():
    data_x = []
    data_y = []

    files = os.listdir('./characters/')
    random.shuffle(files)

    for file in files:
        image = cv2.imread(f'./characters/{file}') / 255.0
        data_x.append(cv2.resize(image, dsize=(CHAR_SHAPE[0], CHAR_SHAPE[1]), interpolation=cv2.INTER_AREA))
        cv2.imwrite(f'./input/{file}', cv2.resize(image, dsize=(CHAR_SHAPE[0], CHAR_SHAPE[1]), interpolation=cv2.INTER_AREA) * 255.0)
        label = np.zeros(26, dtype=int)
        label[ord(file[0]) - 97] = 1
        data_y.append(label)
    return np.asarray(data_x), np.asarray(data_y)


def create_character_model():
    model = keras.Sequential([
        layers.InputLayer(input_shape=CHAR_SHAPE),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        layers.Dropout(0.1),
        layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        layers.Flatten(),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(2048, activation='relu'),
        layers.Dense(26, activation='softmax')
    ])

    return model


def train_character_model(model: keras.Model, data_x, data_y):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x=data_x, y=data_y, epochs=EPOCH_SIZE, batch_size=BATCH_SIZE, validation_split=0.1)

    keras.models.save_model(model, './character_model.h5')


def main():
    # load datasets
    bbox_data_x, bbox_data_y = load_bbox_dataset()
    character_data_x, character_data_y = load_character_dataset()

    # load models
    bbox_model = create_bbox_model()
    character_model = create_character_model()

    # train models
    train_character_model(bbox_model, character_data_x.reshape(-1, 20, 30, 3), character_data_y)
    train_bbox_model(character_model, bbox_data_x.reshape(-1, 96, 71, 3), bbox_data_y)


if __name__ == '__main__':
    main()

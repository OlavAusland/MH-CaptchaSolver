import os
import cv2
import keras.models

INPUT_SHAPE = (96, 71, 3)
CHAR_SIZE = (20, 30)
cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)

bbox_model: keras.models.Model = keras.models.load_model('./bbox_model.h5')
char_model: keras.models.Model = keras.models.load_model('./character_model.h5')


class CaptchaPrediction:
    def __init__(self, bbox: tuple[int, int, int, int], label: str):
        self.bbox = bbox
        self.label = label

    def __repr__(self):
        return (f'Predicted Label: {self.label}\n'
                f'Predicted bounding box: {self.bbox}')


def predict_image(path: str) -> CaptchaPrediction:
    original_image = cv2.imread(path) / 255.0
    original_image = original_image[12:71 + 12, 7:96 + 7]
    image = original_image.copy()

    label = ''
    x_min, y_min, x_max, y_max = -1, -1, -1, -1

    predictions = bbox_model.predict([image.reshape(-1, 96, 71, 3)])

    for prediction in predictions:
        x_min: int = int(prediction[0][0] * INPUT_SHAPE[0])
        y_min: int = int(prediction[0][1] * INPUT_SHAPE[1])
        x_max: int = int(prediction[0][2] * INPUT_SHAPE[0])
        y_max: int = int(prediction[0][3] * INPUT_SHAPE[1])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

        character = original_image[y_min + 1:y_max, x_min + 1:x_max]

        character = cv2.resize(character, dsize=(20, 30), interpolation=cv2.INTER_AREA)
        character_prediction = char_model.predict([character.reshape(-1, 20, 30, 3)])

        label += chr(character_prediction.argmax() + 65)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_min - 10), (255, 255, 255), -1)
        cv2.putText(image, label[-1], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

    while True:
        cv2.imshow('prediction', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return CaptchaPrediction((x_min, y_min, x_max, y_max), label)


def main():
    for file in os.listdir('./captcha'):
        print(predict_image(f'./captcha/{file}'), end='\n\n')


if __name__ == '__main__':
    main()

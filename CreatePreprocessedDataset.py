import os
import json
import requests
from PIL import Image
import cv2
import numpy as np
from operator import itemgetter
from functools import partial
from cvlib.object_detection import draw_bbox
import cvlib

DEBUG = True


def download_captcha_image(path: str, name: str):
    url = 'https://www.mafiaenshevn.com/antibot.php'

    headers = {
        'cookie': 'mhkey=%242y%2410%24T.C0OcXO39jISlprW.d7EuVsFJ4rs673%2FQniFHu7LC4flJPeXpPuS;'
                  'mhuser=TereseLover;'
    }

    response = requests.get(url, headers=headers, stream=True)
    image = Image.open(response.raw)
    image.save(f'{path}/{name}.png')

def preprocess_image_demo(path: str):
    h_l, s_l, v_l = 0, 0, 0
    h_h, s_h, v_h = 255, 255, 255

    window = 'preprocessing'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.createTrackbar('h_l', window, 0, 255, lambda x: x)
    cv2.createTrackbar('s_l', window, 0, 255, lambda x: x)
    cv2.createTrackbar('v_l', window, 0, 255, lambda x: x)

    cv2.createTrackbar('h_h', window, 0, 255, lambda x: x)
    cv2.createTrackbar('s_h', window, 0, 255, lambda x: x)
    cv2.createTrackbar('v_h', window, 0, 255, lambda x: x)

    image = cv2.imread(path)
    original_image = image[12:71 + 12, 7:96 + 7]

    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # 55 70 42 - 151 255 255
    # 00 00 00 - 00 255 42
    while True:
        image = original_image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_l = cv2.getTrackbarPos('h_l', window)
        s_l = cv2.getTrackbarPos('s_l', window)
        v_l = cv2.getTrackbarPos('v_l', window)
        h_h = cv2.getTrackbarPos('h_h', window)
        s_h = cv2.getTrackbarPos('s_h', window)
        v_h = cv2.getTrackbarPos('v_h', window)

        low = np.array([h_l, s_l, v_l])
        high = np.array([h_h, s_h, v_h])

        mask = cv2.inRange(hsv, low, high)

        image[mask > 0] = (0, 0, 0)

        cv2.imshow(window, image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def preprocess_image(path: str, out_path = None):
    original_image = cv2.imread(path)
    original_image = original_image[12:71 + 12, 7:96 + 7]

    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    background_mask = cv2.inRange(hsv, np.array([55, 70, 42]), np.array([151, 255, 75]))
    logo_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([0, 255, 42]))

    image = original_image.copy()
    image[(background_mask | logo_mask) > 0] = (0, 0, 0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.filter2D(image, -1, np.ones((3, 3), np.float32)/25)

    # with filter = 10 without = 55
    image[image < 15] = 0
    # image[image < 85] = 0
    image[image > 0] *= 5

    # image = cv2.blur(image, (5, 5))

    if(out_path):
        cv2.imwrite(out_path, image)

    return image

cv2.namedWindow("bbox image", cv2.WINDOW_NORMAL)
def get_bounding_box(file: str):
    original_image = cv2.imread(f'./data/preprocessed/{file}', cv2.IMREAD_GRAYSCALE)
    preview_image = original_image.copy()

    contours, hierarchy = cv2.findContours(original_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []
    os.system('clear')
    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1 and cv2.contourArea(c) > 100:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Draw the bounding boxes on the (copied) input image:
    if(len(boundRect) < 3): return
    print(boundRect)
    boundRect = sorted(boundRect, key=itemgetter(0))

    for i in range(min(len(boundRect), 3)):
        color = (255, 255, 0)
        cv2.rectangle(preview_image, (int(boundRect[i][0]), int(boundRect[i][1])), \
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 1)

    cv2.imwrite(f'./data/bbox/{file}', original_image)
    with open('./data/bbox/bbox.json', 'r+') as f:
        data = json.load(f)
        data[file] = []
        for i in range(min(len(boundRect), 3)):
            data[file].append({
                "x_min": int(boundRect[i][0]),
                "y_min": int(boundRect[i][1]),
                "x_max": int(boundRect[i][0] + boundRect[i][2]),
                "y_max": int(boundRect[i][1] + boundRect[i][3])
            })
        f.seek(0)
        json.dump(data, f, indent=4)


def extract_characters():
    index: int = 0
    with open('./data/bbox/bbox.json', 'r+') as f:
        data = json.load(f)

        for entry in [key for key in data]:
            image = cv2.imread(f'./data/bbox/{entry}', cv2.IMREAD_GRAYSCALE)

            for i, bbox in enumerate(data[entry]):
                if DEBUG:
                    while True:
                        cv2.imshow('bbox image', image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']])
                        print(entry, entry[i])
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                else:
                    cv2.imwrite(f'./data/characters/{entry[i]} ({index}).png', image[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']])
                index += 1

def main():
    extract_characters()
    #for file in os.listdir('./data/preprocessed'):
    #    get_bounding_box(file)
    #    display_bbox(file)
    # for file in os.listdir('./data/annotated'):
    #    preprocess_image(f'./data/annotated/{file}', f'./data/preprocessed/{file}')

    # image = './data/annotated/tey.png'
    # preprocess_image_demo(image)
    # preprocess_image(image)


if __name__ == '__main__':
    main()


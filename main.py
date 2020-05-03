import modules

import os
import numpy as np
import cv2


def load_models(max_size=500, *file_names):
    """Load models.

    Загружает изображения(grayscale) из папки models с названиями file_names,
    при загрузке каждое изображение масштабируется, чтобы его большая сторона стала равна max_size

    Если не указаны file_names, будут загружены все файлы

    Возвращает словарь {file_name: [image, key_points, description]}
    """

    models = {}
    orb = cv2.ORB_create()

    if not file_names:
        file_names = os.listdir('models')

    for file_name in file_names:
        image = cv2.imread('models/' + file_name, 0)
        image = modules.rescale(image, max_size)
        key_points, description = orb.detectAndCompute(image, None)
        models[file_name] = [image, key_points, description]

    return models


def main():
    models = load_models(500, 'card.jpg')
    photo_bgr = modules.rescale(cv2.imread('photo.jpg'), 500)
    photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp_photo, des_photo = orb.detectAndCompute(photo, None)
    model, kp_model, des_model = list(models.values())[0]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_model, des_photo)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.expand_dims(np.float32([kp_model[x.queryIdx].pt for x in matches]), 1)
    dst_pts = np.expand_dims(np.float32([kp_photo[x.trainIdx].pt for x in matches]), 1)

    transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = model.shape
    pts = np.expand_dims(np.float32([[0, 0], [0, h], [w, h], [w, 0]]), 1)

    dst = cv2.perspectiveTransform(pts, transform_matrix)

    cv2.polylines(photo_bgr, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', photo_bgr)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

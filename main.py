import modules

import os
import numpy as np
import cv2
import sys


def alpha_blending(background, foreground, alpha=None):
    """Alpha blending.

    Наложение foreground на background.
    Если alpha = None, то наложение происходит с ипсользованием альфа-канала foreground

    Возвращает image - результат наложения
    """
    if alpha is None and foreground.shape[2] == 4:
        alpha = foreground[:, :, 3] // 255

    image = np.copy(background)
    image[:, :, 0] = image[:, :, 0] * (1 - alpha) + foreground[:, :, 0] * alpha
    image[:, :, 1] = image[:, :, 1] * (1 - alpha) + foreground[:, :, 1] * alpha
    image[:, :, 2] = image[:, :, 2] * (1 - alpha) + foreground[:, :, 2] * alpha
    return image


def read_input(input_type, source):
    """Read input

    Считать входное изображение

    Аргументы:
        input_type: '1'/'2' - изображение/видео
        source: для файла указать путь: 'image.jpg' или 'video.mp4.
            для камеры: 0-9 - номер камеры или '192.168.0.xxx:port' - для ip камеры

    Возвращает image - считанное изображение
    """

    flag, photo_bgr = None, None
    if input_type == '1':
        photo_bgr = source
    if input_type == '2':
        flag, photo_bgr = source.read()
    return modules.rescale(photo_bgr, 500)


def main():
    input_type, source = modules.get_input()
    photo_bgr = read_input(input_type, source)

    img_paste = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
    h, w = img_paste.shape[:2]
    points_paste = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Стартова модель - квадрат - описывающий acuro маркер в пространстве
    model_points = np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]], dtype=np.float32)
    # Точки которые должны быть перемещены в новую позицую
    points3d = np.array([(0.0, 50.0, -50.0), (0.0, 50.0, -100.0), (100.0, 50.0, -100.0), (100.0, 50.0, -50.0)])

    camera_matrix = np.array([[photo_bgr.shape[1], 0, photo_bgr.shape[1] / 2],
                              [0, photo_bgr.shape[1], photo_bgr.shape[0] / 2],
                              [0, 0, 1]], dtype=np.float32)

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    while cv2.waitKey(1) != 27:  # пока не нажат #ESC
        photo_bgr = read_input(input_type, source)

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(photo_bgr, dictionary,
                                                                               parameters=parameters)

        for corners in markerCorners:
            matrix = cv2.getPerspectiveTransform(points_paste, corners)
            result = cv2.warpPerspective(img_paste, matrix, photo_bgr.shape[:2][::-1])

            photo_bgr = alpha_blending(photo_bgr, result)

            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            _, rotation_vector, translation_vector = cv2.solvePnP(model_points, corners, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            points2D, _ = cv2.projectPoints(points3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            ap = (int(points2D[3][0][0]), int(points2D[3][0][1]))
            bp = (int(points2D[0][0][0]), int(points2D[0][0][1]))
            cv2.line(photo_bgr, ap, bp, (255, 255, 255), 5)

            ap = (int(points2D[0][0][0]), int(points2D[0][0][1]))
            bp = (int(points2D[1][0][0]), int(points2D[1][0][1]))
            cv2.line(photo_bgr, ap, bp, (255, 255, 255), 5)

            ap = (int(points2D[1][0][0]), int(points2D[1][0][1]))
            bp = (int(points2D[2][0][0]), int(points2D[2][0][1]))
            cv2.line(photo_bgr, ap, bp, (255, 255, 255), 5)

            ap = (int(points2D[2][0][0]), int(points2D[2][0][1]))
            bp = (int(points2D[3][0][0]), int(points2D[3][0][1]))
            cv2.line(photo_bgr, ap, bp, (255, 255, 255), 5)

        cv2.imshow('frame', photo_bgr)

        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()

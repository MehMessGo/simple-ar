from modules import ar
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
        alpha = foreground[:, :, 3] / 255

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

    ar_tool = ar.ArucoAR()

    ar_tool.add_paste_image(cv2.imread('input.png', cv2.IMREAD_UNCHANGED), 0)
    ar_tool.add_paste_image(cv2.imread('input2.png', cv2.IMREAD_UNCHANGED), 1)

    # Точки которые должны быть перемещены в новую позицую
    ar_tool.points3D = np.array([(-50.0, -50.0, -80.0), (50.0, -50.0, -80.0), (50.0, 50.0, 0.0), (-50.0, 50.0, 0.0)])

    ar_tool.camera_matrix = np.array([[photo_bgr.shape[1], 0, photo_bgr.shape[1] / 2],
                                      [0, photo_bgr.shape[1], photo_bgr.shape[0] / 2],
                                      [0, 0, 1]], dtype=np.float32)

    ar_tool.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    while cv2.waitKey(1) != 27:  # пока не нажат #ESC
        photo_bgr = read_input(input_type, source)

        markers = ar_tool.coordinate_transformation(photo_bgr)
        for marker in markers:
            marker_id = marker[0]
            point2D_array = marker[1]

            matrix = cv2.getPerspectiveTransform(ar_tool.img_paste_dictionary[marker_id][1], point2D_array)
            result = cv2.warpPerspective(ar_tool.img_paste_dictionary[marker_id][0], matrix, photo_bgr.shape[:2][::-1])

            photo_bgr = alpha_blending(photo_bgr, result)

        cv2.imshow('frame', photo_bgr)

        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()

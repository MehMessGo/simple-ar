"""
:authors: Michael Pakhmurin
:license: MIT License, see LICENSE file

:copyright: (c) 2020 Michael Pakhmurin
"""

import simple_ar

import numpy as np
import cv2
import sys


def get_input():
    """Get input.

    Получить данные из sys.argv, если таких нет, то
    запросить ввод данных:

    первый аргумент - тип входных данных:
        '1' - изображение.
        '2' - видео.
    второй аргумент - путь к источнику:
        для файла просто указать путь: 'image.jpg' или 'video.mp4.
        для камеры: 0-9 - номер камеры или '192.168.0.xxx:port' - для ip камеры

    Вовзравщает:
        input_type - '1'/'2' - изобажение/видео
        source - изображение или cv2.VideoCapture для видео
    """

    source = None
    if len(sys.argv) != 3:
        print("Выберите источник входных данных:")
        print(" \t* 1-изображение")
        print(" \t* 2-видео")
        input_type = input('> ')

        print("Путь к источнику:")
        input_path = input('> ')
    else:
        *_, input_type, input_path = sys.argv

    if input_type == '1':  # открыть файл - фотографию
        source = simple_ar.rescale(cv2.imread(input_path), 500)
    elif input_type == '2':  # открыть видео
        if input_path == '':  # открыть стандрартную камеру
            input_path = 0
        elif len(input_path) == 1:  # если путь слишком короткий, открыть камеру с номером
            input_path = int(input_path)
        source = cv2.VideoCapture(input_path)

    return input_type, source


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
    return simple_ar.rescale(photo_bgr, 500)


def example():
    input_type, source = get_input()
    photo_bgr = read_input(input_type, source)

    ar_tool = simple_ar.ar.ArucoAR(cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250))

    ar_tool.camera_matrix = np.array([[photo_bgr.shape[1], 0, photo_bgr.shape[1] / 2],
                                      [0, photo_bgr.shape[1], photo_bgr.shape[0] / 2],
                                      [0, 0, 1]], dtype=np.float32)

    ar_tool.add_image_paste(cv2.imread('input.png', cv2.IMREAD_UNCHANGED), 0, np.array(
        [(-50.0, 0.0, -80.0), (50.0, 0.0, -80.0), (50.0, 0.0, 0.0), (-50.0, 0.0, 0.0)]))

    while cv2.waitKey(1) != 27:  # пока не нажат #ESC
        photo_bgr = read_input(input_type, source)

        photo_bgr = ar_tool.draw_on_markers(photo_bgr)

        cv2.imshow('frame', photo_bgr)

        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    example()

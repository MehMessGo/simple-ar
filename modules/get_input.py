from .rescale import rescale

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
        source = rescale(cv2.imread(input_path), 500)
    elif input_type == '2':  # открыть видео
        if input_path == '':  # открыть стандрартную камеру
            input_path = 0
        elif len(input_path) == 1:  # если путь слишком короткий, открыть камеру с номером
            input_path = int(input_path)
        source = cv2.VideoCapture(input_path)

    return input_type, source

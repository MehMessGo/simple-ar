import modules

import os
import numpy as np
import cv2
import sys


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
        source = modules.rescale(cv2.imread(input_path), 500)
    elif input_type == '2':  # открыть видео
        if input_path == '':  # открыть стандрартную камеру
            input_path = 0
        elif len(input_path) == 1:  # если путь слишком короткий, открыть камеру с номером
            input_path = int(input_path)
        source = cv2.VideoCapture(input_path)

    return input_type, source


def main():
    input_type, source = get_input()
    models = load_models(500, 'card3.jpg')
    photo_bgr = None

    while True:
        if input_type == '1':
            photo_bgr = modules.rescale(source, 500)
        if input_type == '2':
            flag, photo_bgr = source.read()
            if cv2.waitKey(1) == 27:  # ESC
                break
        photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)


        orb = cv2.ORB_create()
        kp_photo, des_photo = orb.detectAndCompute(photo, None)
        model, kp_model, des_model = list(models.values())[0]

        if (des_photo is not None) and (des_model is not None):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_model, des_photo)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.expand_dims(np.float32([kp_model[x.queryIdx].pt for x in matches]), 1)
            dst_pts = np.expand_dims(np.float32([kp_photo[x.trainIdx].pt for x in matches]), 1)

            transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if transform_matrix is not None:
                h, w = model.shape
                pts = np.expand_dims(np.float32([[0, 0], [0, h], [w, h], [w, 0]]), 1)

                dst = cv2.perspectiveTransform(pts, transform_matrix)
                cv2.polylines(photo_bgr, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('frame', photo_bgr)
        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()

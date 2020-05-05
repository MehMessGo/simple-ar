from .rescale import rescale

import cv2
import os


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
        file_names = os.listdir('../models')

    for file_name in file_names:
        image = cv2.imread('models/' + file_name, 0)
        image = rescale(image, max_size)
        key_points, description = orb.detectAndCompute(image, None)
        models[file_name] = [image, key_points, description]

    return models

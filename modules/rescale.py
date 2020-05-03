import cv2


def rescale(image, max_size):
    """Rescale image by max size.

    Масштабирование изображения(image),
    где большая сторона становится равана max_size

    Возвращает масштабированное изображение
    """
    h, w = image.shape[:2]

    coef = max(h, w) / max_size
    w = int(w / coef)
    h = int(h / coef)

    return cv2.resize(image, (w, h))

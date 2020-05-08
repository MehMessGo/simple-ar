"""
:authors: Michael Pakhmurin
:license: MIT License, see LICENSE file

:copyright: (c) 2020 Michael Pakhmurin
"""

import numpy as np
import cv2


class AR(object):
    def __init__(self):
        """AR

        Атрибуты:
            start_plane:  набор 3d точек описывающих маркер в пространстве.
            camera_matrix: матрица камеры/
            dist_coeffs: Дисторсия камеры.
            img_paste_dictionary: параметры изображений для рисования на маркере.
            type-detection: Метод детектирования объектов:
                                aruco: аруко-маркеры.
        """

        self.start_plane = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], dtype=np.float32)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        self.type_detection = 'aruco'

        self.img_paste_dictionary = {}

    @staticmethod
    def alpha_blending(background, foreground, alpha=None):
        """Alpha blending.

        Наложение foreground на background.
        Если alpha = None, то наложение происходит с ипсользованием альфа-канала foreground.

        Возвращает image - результат наложения.
        """
        if alpha is None and foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255

        image = np.copy(background)
        image[:, :, 0] = image[:, :, 0] * (1 - alpha) + foreground[:, :, 0] * alpha
        image[:, :, 1] = image[:, :, 1] * (1 - alpha) + foreground[:, :, 1] * alpha
        image[:, :, 2] = image[:, :, 2] * (1 - alpha) + foreground[:, :, 2] * alpha
        return image


class ArucoAR(AR):
    def __init__(self, aruco_dictionary):
        """AR

        Атрибуты:
            parameters: начальный набор параметров aruco детектора.
            aruco_dictionary: словарь аруко маркеров (cv2.aruco.DICT_NXN_NNN).
        """
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_dictionary = aruco_dictionary
        super().__init__()

    def add_image_paste(self, image, aruco_id, points_3d):
        """Add image for paste

        Добавление параметров 3-d изображения в self.aruco_dictionary,
        которое будет отрисовано возле маркера

        Аргументы:
            image: Изображение(BGRA), для данного маркера
            aruco_id: Номер(id) данного маркера
            points_3d: координаты пространства, в которых будет отрисовано изображение
                центром координат пространства является центр маркера
        """

        h, w = image.shape[:2]
        points_paste = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if aruco_id not in self.img_paste_dictionary:
            self.img_paste_dictionary[aruco_id] = []
        self.img_paste_dictionary[aruco_id].append([image, points_paste, points_3d])

    def coordinate_transformation(self, points3D, marker_corners, marker_id):
        """Сoordinate transformation

        Нахождение координаты маркера в пространстве,
        перенос заданых координат(points_3d) к маркеру
        и преобразование их в координаты на изображение

        Аргументы:
            points_3d: координаты точек пространства для переноса.
            marker_corners: координаты марекра на изображение.
            marker_ids: номер(id) маркера.

        Возвращает:
            point2D_array - список координат точек на изображение
        """
        point2D_array = [[]]

        _, rotation_vector, translation_vector = cv2.solvePnP(self.start_plane, marker_corners,
                                                              self.camera_matrix,
                                                              self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        points2D, _ = cv2.projectPoints(points3D, rotation_vector, translation_vector,
                                        self.camera_matrix,
                                        self.dist_coeffs)

        for point2D in points2D:
            point2D_array[0].append([point2D[0][0], point2D[0][1]])

        point2D_array = np.array(point2D_array, dtype=np.float32)

        return point2D_array

    def draw_on_markers(self, image):
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image,
                                                                self.aruco_dictionary,
                                                                parameters=self.parameters)
        for i in range(len(marker_corners)):
            if marker_ids[i][0] in self.img_paste_dictionary:
                for marker_properties in self.img_paste_dictionary[marker_ids[i][0]]:
                    point2D_array = self.coordinate_transformation(marker_properties[2],
                                                                   marker_corners[i],
                                                                   marker_ids[i][0])

                    matrix = cv2.getPerspectiveTransform(marker_properties[1], point2D_array)
                    result = cv2.warpPerspective(marker_properties[0], matrix, image.shape[:2][::-1])
                    image = self.alpha_blending(image, result)

        return image

    def aruco_generator(self, marker_id, size):
        marker_image = np.zeros((size, size), dtype=np.uint8)
        marker_image = cv2.aruco.drawMarker(self.aruco_dictionary, marker_id, size, marker_image, 1)
        return marker_image

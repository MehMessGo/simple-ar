import numpy as np
import cv2


class AR(object):
    def __init__(self):
        # Стартова модель - квадрат - описывающий acuro маркер в пространстве
        self.start_plane = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], dtype=np.float32)

        # Точки которые должны быть перемещены в новую позицую
        self.points3D = None

        self.camera_matrix = None
        self.type_detection = 'aruco'

        self.img_paste_dictionary = {}


class ArucoAR(AR):
    def __init__(self):
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.dictionary = None
        super().__init__()

    def add_paste_image(self, image, aruco_id):
        h, w = image.shape[:2]
        points_paste = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        self.img_paste_dictionary[aruco_id] = [image, points_paste]

    def coordinate_transformation(self, image):
        marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(image, self.dictionary,
                                                                                  parameters=self.parameters)
        markers = []
        for _id in range(len(marker_corners)):
            point2D_array = [[]]
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            _, rotation_vector, translation_vector = cv2.solvePnP(self.start_plane, marker_corners[_id],
                                                                  self.camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            points2D, _ = cv2.projectPoints(self.points3D, rotation_vector, translation_vector,
                                            self.camera_matrix,
                                            dist_coeffs)

            for point2D in points2D:
                point2D_array[0].append([point2D[0][0], point2D[0][1]])

            point2D_array = np.array(point2D_array, dtype=np.float32)
            markers.append([marker_ids[_id][0], point2D_array])

        return markers

from modules import ar
import modules

import os
import numpy as np
import cv2
import sys


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

    ar_tool.camera_matrix = np.array([[photo_bgr.shape[1], 0, photo_bgr.shape[1] / 2],
                                      [0, photo_bgr.shape[1], photo_bgr.shape[0] / 2],
                                      [0, 0, 1]], dtype=np.float32)

    ar_tool.aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    ar_tool.add_image_paste(cv2.imread('input.png', cv2.IMREAD_UNCHANGED), 0, np.array(
        [(-50.0, -50.0, -80.0), (50.0, -50.0, -80.0), (50.0, 50.0, 0.0), (-50.0, 50.0, 0.0)]))
    ar_tool.add_image_paste(cv2.imread('input2.png', cv2.IMREAD_UNCHANGED), 0, np.array(
        [(-50.0, -50.0, -80.0), (50.0, -50.0, -80.0), (50.0, 50.0, 0.0), (-50.0, 50.0, 0.0)]))

    while cv2.waitKey(1) != 27:  # пока не нажат #ESC
        photo_bgr = read_input(input_type, source)

        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(photo_bgr,
                                                                ar_tool.aruco_dictionary,
                                                                parameters=ar_tool.parameters)
        for i in range(len(marker_corners)):
            if marker_ids[i][0] in ar_tool.img_paste_dictionary:
                for marker_properties in ar_tool.img_paste_dictionary[marker_ids[i][0]]:
                    point2D_array = ar_tool.coordinate_transformation(marker_properties[2],
                                                                      marker_corners[i],
                                                                      marker_ids[i][0])

                    matrix = cv2.getPerspectiveTransform(marker_properties[1], point2D_array)
                    result = cv2.warpPerspective(marker_properties[0], matrix, photo_bgr.shape[:2][::-1])
                    photo_bgr = ar_tool.alpha_blending(photo_bgr, result)

        cv2.imshow('frame', photo_bgr)

        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()

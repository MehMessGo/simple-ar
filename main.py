import modules

import os
import numpy as np
import cv2
import sys


def aruco_get_markers(image):
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, 23, 200, markerImage, 1)

    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    return markerCorners, markerIds, rejectedCandidates


def main():
    input_type, source = modules.get_input()
    photo_bgr = None

    img_paste = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
    while True:
        if input_type == '1':
            photo_bgr = modules.rescale(source, 500)
        if input_type == '2':
            flag, photo_bgr = source.read()
            if cv2.waitKey(1) == 27:  # ESC
                break

        markerCorners, markerIds, rejectedCandidates = aruco_get_markers(photo_bgr)
        cv2.aruco.drawDetectedMarkers(photo_bgr, markerCorners, markerIds)

        if len(markerCorners) > 0:
            h, w = img_paste.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            for corners in markerCorners:
                pts2 = corners
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(img_paste, matrix, photo_bgr.shape[:2][::-1])

                alpha = result[:, :, 3]//255
                photo_bgr[:, :, 0] = photo_bgr[:, :, 0] * (1 - alpha) + result[:, :, 0] * alpha
                photo_bgr[:, :, 1] = photo_bgr[:, :, 1] * (1 - alpha) + result[:, :, 1] * alpha
                photo_bgr[:, :, 2] = photo_bgr[:, :, 2] * (1 - alpha) + result[:, :, 2] * alpha

        cv2.imshow('frame', photo_bgr)

        if input_type == '1':
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    main()

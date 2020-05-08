# Not working yet


# import cv2
# import numpy as np
#
#
# def draw_simple_markers(image, models):
#     photo = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     orb = cv2.ORB_create()
#     kp_photo, des_photo = orb.detectAndCompute(photo, None)
#     model, kp_model, des_model = list(models.values())[0]
#
#     if (des_photo is not None) and (des_model is not None):
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#         matches = bf.match(des_model, des_photo)
#         matches = sorted(matches, key=lambda x: x.distance)
#
#         src_pts = np.expand_dims(np.float32([kp_model[x.queryIdx].pt for x in matches]), 1)
#         dst_pts = np.expand_dims(np.float32([kp_photo[x.trainIdx].pt for x in matches]), 1)
#
#         transform_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         if transform_matrix is not None:
#             h, w = model.shape
#             pts = np.expand_dims(np.float32([[0, 0], [0, h], [w, h], [w, 0]]), 1)
#
#             dst = cv2.perspectiveTransform(pts, transform_matrix)
#             cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

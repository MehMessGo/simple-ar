import modules

import cv2

if __name__ == '__main__':
    show_matches_count = 15
    photo = modules.rescale(cv2.imread('photo2.jpg', 0), 500)
    model = modules.rescale(cv2.imread('card2.jpg', 0), 500)

    orb = cv2.ORB_create()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_model, des_model = orb.detectAndCompute(model, None)
    kp_photo, des_photo = orb.detectAndCompute(photo, None)

    matches = bf.match(des_model, des_photo)

    matches = sorted(matches, key=lambda x: x.distance)

    img = cv2.drawMatches(model, kp_model, photo, kp_photo, matches[:show_matches_count], 0, flags=2)
    cv2.imshow('frame', img)
    cv2.waitKey(0)

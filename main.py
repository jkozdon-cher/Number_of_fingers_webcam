import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    try:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # rectangle with test_area to reading the hand
        x1, y1 = 100, 400
        x2, y2 = 300, 600
        cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 0, 0), 2)
        test_area = frame[x1:x2, y1:y2]
        image = cv2.resize(test_area, (int(test_area.shape[1] * 2), int(test_area.shape[0] * 2)))
        contours_image = np.zeros(image.shape, np.uint8)

        blur_kernel = (3, 3)
        dilate_kernel = np.ones((3, 3), np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # value rangers for skin colors
        lower_skin1 = np.array([0, 30, 53], dtype=np.uint8)
        upper_skin1 = np.array([60, 255, 255], dtype=np.uint8)

        lower_skin2 = np.array([172, 30, 53], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

        # threshold the hsv image to get only skin colors
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

        mask1 = cv2.GaussianBlur(mask1, blur_kernel, 0)
        mask2 = cv2.GaussianBlur(mask2, blur_kernel, 0)

        skin = cv2.bitwise_or(mask1, mask2)
        dilate = cv2.dilate(skin, dilate_kernel, iterations=4)
        cv2.imshow('dilate', dilate)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_i = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_i = i

        cnt = contours[max_i]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
            center = (cx, cy)
            cv2.circle(contours_image, center, 5, [0, 0, 255], 2)

        cv2.drawContours(contours_image, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(contours_image, [hull], 0, (0, 255, 0), 2)

        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)

        defects = cv2.convexityDefects(cnt, hull)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt, center, True)
            cv2.line(contours_image, start, end, [0, 0, 255], 5)
            cv2.circle(contours_image, far, 10, [0, 0, 255], -1)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Number of Fingers: " + str(defects.shape[0] - 1), (50, 50), font, 1, (255, 0, 0), 2)

        cv2.imshow('contours', contours_image)
        cv2.imshow('frame', frame)

    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

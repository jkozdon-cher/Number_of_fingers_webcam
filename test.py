import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    try:
        _, img = cap.read()
        img = cv2.flip(img, 1)
        # img = cv2.resize(img, (int(img.shape[1] / 4), int(img.shape[0] / 4)))

        test_area = img[100:300, 400:600]  # x1:x2, y1:y2 - 0:0 to lewy górny róg
        cv2.rectangle(img, (400, 100), (600, 300), (0, 255, 0), 2)  # (y1, x1), (y2, x2) - 0:0 to lewy górny róg
        '''
        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(test_area, cv2.COLOR_BGR2HSV)
        # gray = cv2.cvtColor(test_area, cv2.COLOR_BGR2GRAY)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        mask = cv2.dilate(mask, kernel, iterations=5)
        cv2.imshow('contours', mask)
        '''
        gray = cv2.cvtColor(test_area, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret, thresh1 = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh1 = cv2.adaptiveThreshold(gray,
                                        maxValue=255,
                                        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                        thresholdType=cv2.THRESH_BINARY,
                                        blockSize=9,
                                        C=8)
        cv2.imshow('contours', thresh1)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros(test_area.shape, np.uint8)

        max_area = 0

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area > max_area):
                max_area = area
                ci = i
        cnt = contours[ci]
        hull = cv2.convexHull(cnt)
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00

        centr = (cx, cy)
        cv2.circle(img, centr, 5, [0, 0, 255], 2)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)
        '''
        cnt = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        
        defects = cv2.convexityDefects(cnt, hull)
        mind = 0
        maxd = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            dist = cv2.pointPolygonTest(cnt, centr, True)
            cv2.line(drawing, start, end, [0, 0, 255], 5)

            cv2.circle(drawing, far, 10, [0, 0, 255], -1)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "Number of Fingers: " + str(i), (26, 106), font, 1, (0, 0, 255), 2)
        i = 0
        '''
        # cv2.imshow('contours', drawing)
        cv2.imshow('frame', img)

    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

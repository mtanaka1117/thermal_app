import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import sys
import datetime as dt

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    sys.exit()

while True:
    ret, img = cap.read()
    cv2.imshow('test', img)
    if cv2.waitKey(1) == 27:
        # now = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        # cv2.imwrite('./test.jpg', img)
        break

cv2.destroyAllWindows()
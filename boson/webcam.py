import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import sys
import datetime as dt

cap = cv2.VideoCapture(1)
_, img = cap.read()
now = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
with open('./thermal/{}_V.dat'.format(now), 'wb') as f:
    f.write(img)
cv2.imwrite('./thermal/test.jpg', img)

# if not cap.isOpened():
#     sys.exit()

# while True:
#     ret, img = cap.read()
#     cv2.imshow('test', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
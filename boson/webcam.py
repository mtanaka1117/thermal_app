import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import sys

cap = cv2.VideoCapture(1)
ret, img = cap.read()
cv2.imwrite('test.jpg',img)

# if not cap.isOpened():
#     sys.exit()

# while True:
#     ret, img = cap.read()
#     cv2.imshow('test', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
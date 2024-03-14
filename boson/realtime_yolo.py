import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import sys
from ultralytics import YOLO

cap = cv2.VideoCapture(1)
model = YOLO("yolov8n.pt")

# if not cap.isOpened():
#     sys.exit()

# while True:
#     ret, img = cap.read()
#     pred = model.predict(img, classes=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79])
#     frame = pred[0].plot()
#     cv2.imshow('test', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

results = model(1, show=True)

# cv2.destroyWindow()
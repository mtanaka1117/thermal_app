from ultralytics import YOLO
import cv2

img = cv2.imread('test.jpg')
model = YOLO("yolov8x.pt")

results = model.predict(img)
ret = results[0].plot()
cv2.imshow('test', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()

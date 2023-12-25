from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8x.pt")

path = r"C:\Users\tnkmo\Downloads\items4\items4\20230807_1348\20230807_134859802_V.jpg"

image = cv2.imread(path)
pred = model.predict(image)
bboxes = pred[0].boxes.xyxy.cpu().numpy()
classes = pred[0].boxes.cls.cpu().numpy()

# crops = []
for box, cls in zip(bboxes, classes):
    xmin, ymin, xmax, ymax = map(int, box[:4])
    crop = image[ymin:ymax, xmin:xmax]
    cv2.imwrite('./thumbnails/{}.png'.format(int(cls)), crop)

for box, cls in zip(bboxes, classes):
    img = image.copy()
    xmin, ymin, xmax, ymax = map(int, box[:4])
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
    cv2.imwrite('./thumbnails/detail_{}.png'.format(int(cls)), img)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    
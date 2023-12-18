from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8x.pt")

# path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131359635_V.jpg"
path = r"C:\Users\tnkmo\Downloads\items3\items3\20230807_1332\20230807_133259536_V.jpg"
image = cv2.imread(path)
pred = model.predict(image, classes=[67, 73, 76])
bboxes = pred[0].boxes.xyxy.cpu().numpy()
classes = pred[0].boxes.cls.cpu().numpy()

# crops = []
for box, cls in zip(bboxes, classes):
    xmin, ymin, xmax, ymax = map(int, box[:4])
    crop = image[ymin:ymax, xmin:xmax]
    cv2.imwrite('./thumbnails/{}.png'.format(cls), crop)

for box, cls in zip(bboxes, classes):
    img = image.copy()
    xmin, ymin, xmax, ymax = map(int, box[:4])
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
    cv2.imwrite('./thumbnails/detail_{}.png'.format(cls), img)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    
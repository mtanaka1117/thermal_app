from ultralytics import YOLO
from more_itertools import peekable
import glob
import cv2

path = '/home/srv-admin/images/items1/1313/*_V.jpg'
file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('result.mp4',fourcc, 30.3, (640, 480))

model = YOLO("yolov8x.pt")

for i in file_list:
  try:
    results = model.predict(i, classes=[67, 73, 76])
    video.write(results[0].plot())
  except StopIteration:
        break

video.release()
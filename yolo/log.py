import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
from matplotlib import pyplot as plt
import datetime
from collections import deque, Counter

b_thermal_path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354592_T.jpg'
a_thermal_path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.jpg'

visible_path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358202_V.jpg'

# b_thermal_path = "/home/srv-admin/images/items1/1313/20230807_131354592_T.jpg"
# a_thermal_path = "/home/srv-admin/images/items1/1313/20230807_131358192_T.jpg"

# visible_path = "/home/srv-admin/images/items1/1313/20230807_131358202_V.jpg"

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])


im_b = cv2.imread(b_thermal_path,0)
im_a = cv2.imread(a_thermal_path,0)
im_v = cv2.imread(visible_path)

img_diff = cv2.absdiff(im_b, im_a)
_, img_th = cv2.threshold(img_diff,15,255,cv2.THRESH_BINARY)

kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(img_th, kernel, 2)
img_binary = cv2.warpAffine(dilate, affine_matrix, (im_v.shape[1], im_v.shape[0]))

# retval, labels, _, centroids = cv2.connectedComponentsWithStats(img_binary)
contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
# cv2.drawContours(im_v, contours, -1, (0,0,255), 2)

# cv2.imshow('img', im_v)
# cv2.waitKey(0)

points = []
for i in contours:
    M = cv2.moments(i)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    points.append((cx, cy))


model = YOLO("yolov8x.pt")
class_dic = model.model.names

pred = model.predict(im_v, classes=[67,73,76])
frame = pred[0].plot()
bbox = pred[0].boxes.xyxy.cpu().numpy()
classes = pred[0].boxes.cls.cpu().numpy()
# print(bbox)

polygon = []
for x1, y1, x2, y2 in bbox:
    polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))


# fig, ax = plt.subplots()  
# plt.axis([0,640,0,480])
# for i in polygon:
#     ax.add_patch(plt.Polygon(i,fill=False))

# detect_list = deque(maxlen=5)
detect_dic = {}


with open("example.txt", "w") as f:
    for pt in points:
        for poly, cls in zip(polygon, classes):
            if cv2.pointPolygonTest(poly, pt, False) >= 0:
                # print(class_dic[cls])
                detect_dic.setdefault(cls, 0)
                detect_dic[cls] += 1
                # f.write(str(datetime.datetime.now()) + ", " + "at: table, " + class_dic[cls])
                
print(detect_dic[67])


#         ax.scatter(pt[:1], pt[1:], color=color)
# ax.set_ylim(ax.get_ylim()[::-1])
# ax.xaxis.tick_top()
# ax.yaxis.tick_left()
# plt.show()

# mask_inv = cv2.bitwise_not(img_binary)
# back = cv2.bitwise_and(frame, frame, mask_inv)
# cut = cv2.cvtColor(cv2.bitwise_and(img_binary, img_binary, img_binary), cv2.COLOR_GRAY2BGR)
# paste = cv2.add(back, cut)


# cv2.imshow('img', im_v)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
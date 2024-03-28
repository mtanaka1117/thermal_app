# https://qiita.com/kagami_t/items/2b4db4e2464439a48fb4

import glob
import cv2
import numpy as np
from numba import jit
from more_itertools import peekable
from ultralytics import YOLO
import datetime
import csv
import os
import time
from collections import deque

def feature_compare(img1, img2):
    '''
    画像の特徴量を比較する関数
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    sift = cv2.ORB_create()
    _, des1 = sift.detectAndCompute(img1, None)
    _, des2 = sift.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    return ret

# https://qiita.com/TheJuniorTheSenior/items/b76fed3832c149907b82
# @jit
# def calc_dist(diff):
#     dist = np.sqrt(np.dot(diff.T, diff))
#     return dist

# def calc_min_dist(a, b, c):
#     '''
#     a,b,c: 3次元上の点
#     3点間の距離の和が最も小さいものを返す
#     '''
#     dist_ab = calc_dist(a-b)
#     dist_bc = calc_dist(b-c)
#     dist_ca = calc_dist(c-a)
    
#     return min(dist_ab+dist_ca, dist_bc+dist_ab, dist_ca+dist_bc)

# def get_center(img1, img2):
#     points = []
#     for i in contours:
#         M = cv2.moments(i)
#         cx = M["m10"] / M["m00"]
#         cy = M["m01"] / M["m00"]
#         points.append((cx, cy))
#     return 

# start = time.time()
# affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
#                         [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

affine_matrix = np.array([[1.15919938e+00, 7.27146534e-02, -5.70173323e+01],
                        [1.46108543e-04, 1.16974505e+00, -5.52789456e+01]])

# path = '/home/srv-admin/images/items*/*/*.jpg'
# path = '/home/srv-admin/images/items1/1313/*.jpg'
# path = r"C:\Users\tnkmo\Downloads\20240112_items2-selected\yolo1\20240112_1418\*jpg"
path = r"C:\Users\tnkmo\Downloads\yolo+1\yolo+1\20240112_1450\*jpg"
file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('no_label.mp4',fourcc, 30.3, (640, 480), isColor=True)

if '_V' in file_list.peek():
    bg_v = cv2.imread(next(file_list))
    bg_t = cv2.imread(next(file_list))
else:
    next(file_list)
    bg_v = cv2.imread(next(file_list))
    bg_t = cv2.imread(next(file_list))
    
b_img_v = bg_v.copy()
kernel = np.ones((5,5),np.uint8)

# model = YOLO("yolov8x.pt")
# bg_v_deque = deque([bg_v], maxlen=3)

bg_count = 0

for i in file_list:
    try:
        if '_V' in i and '_T' in file_list.peek():
            img_v = cv2.imread(i)
            blur_v = cv2.medianBlur(img_v, ksize=3)            
            diff_v = cv2.absdiff(blur_v, bg_v)
            diff_v = cv2.cvtColor(diff_v, cv2.COLOR_BGR2GRAY)
            _, img_th_v = cv2.threshold(diff_v,50,255,cv2.THRESH_BINARY)
            dilate_v = cv2.dilate(img_th_v,kernel,iterations=5)
            erode_v = cv2.erode(dilate_v, kernel, 2)

            img_t = cv2.imread(next(file_list))
            blur_t = cv2.medianBlur(img_t, ksize=3)
            diff_t = cv2.absdiff(blur_t, bg_t)
            diff_t = cv2.cvtColor(diff_t, cv2.COLOR_BGR2GRAY)
            affined_t = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
            _, img_th_t = cv2.threshold(affined_t,10,255,cv2.THRESH_BINARY)
            dilate_t = cv2.dilate(img_th_t,kernel,3)
            erode_t = cv2.erode(dilate_t, kernel, 2)

            undefined_obj = img_v.copy()
            contours, _ = cv2.findContours(img_th_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                area = cv2.contourArea(contour)
                # 面積が一定以上の場合にのみ矩形を描画
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(undefined_obj, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # touch_region = cv2.subtract(erode_t, erode_v)
            # contours, _ = cv2.findContours(touch_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # contours = list(filter(lambda x: cv2.contourArea(x) > 80, contours))
            # cv2.drawContours(img_v, contours, -1, (0,0,255), 2)
            # cv2.imshow('img', img_v)
            # cv2.waitKey(1)
            
            # points = []
            # for cnt in contours:
            #     M = cv2.moments(cnt)
            #     cx = M["m10"] / M["m00"]
            #     cy = M["m01"] / M["m00"]
            #     points.append((cx, cy))
            
            # pred = model.predict(img_v_color, classes=[25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 63, 64, 65, 66, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79])
            # frame = pred[0].plot()
            # bboxes = pred[0].boxes.xyxy.cpu().numpy()
            # classes = pred[0].boxes.cls.cpu().numpy()
            
            mask = cv2.subtract(erode_t, erode_v)
            mask_inv = cv2.bitwise_not(mask)
            back = cv2.bitwise_and(undefined_obj, undefined_obj, mask_inv)
            cut = cv2.bitwise_and(mask, mask, mask)
            cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            paste = cv2.add(back, cut)

            # cv2.imshow("img", frame)
            # cv2.waitKey(1)

            # polygon = []
            # for x1, y1, x2, y2 in bboxes:
            #     polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))

            # with open("example.csv", "a") as f:
            #     for poly, cls, bbox in zip(polygon, classes, bboxes):
                    # data = [[datetime.datetime.now(), "table", cls, list(bbox)]]
                    # writer = csv.writer(f)
                    # writer.writerows(data)
                    # for pt in points:
                    #     if cv2.pointPolygonTest(poly, pt, False) >= 0:
                    #         now = datetime.datetime.now()
                    #         data = [[datetime.datetime.now(), "table", cls, list(bbox)]]
                    #         writer = csv.writer(f)
                    #         writer.writerows(data)
                            
                    #         xmin, ymin, xmax, ymax = map(int, bbox[:4])
                    #         crop = img_v_color[ymin:ymax, xmin:xmax]
                    #         path = './thumbnails/{}'.format(int(cls))
                    #         if not os.path.exists(path): os.mkdir(path)
                    #         cv2.imwrite('./thumbnails/{}/{}.png'.format(int(cls), now), crop)
                            
                    #         overview = img_v_color.copy()
                    #         cv2.rectangle(overview, (xmin,ymin), (xmax,ymax), (0, 0, 255), thickness=5)
                    #         cv2.imwrite('./thumbnails/{}/detail_{}.png'.format(int(cls), now), overview)
                            
                    #         break

            if (feature_compare(b_img_v, img_v)<13 and bg_count==30):
                bg_count = 0
                bg_v = img_v.copy()
                b_img_v = img_v.copy()
                # print('更新')
            elif (feature_compare(b_img_v, img_v)<13):
                bg_count += 1
            else: b_img_v = img_v.copy()
            
            video.write(undefined_obj)

    except StopIteration:
        break

video.release()
# end = time.time()
# print(end-start)

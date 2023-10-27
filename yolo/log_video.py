import glob
import cv2
import numpy as np
from more_itertools import peekable
from ultralytics import YOLO
from matplotlib import pyplot as plt
import datetime

def feature_compare(img1, img2):
    '''
    画像の特徴量を比較する関数
    '''
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    return ret

# def get_center(img1, img2):
#     points = []
#     for i in contours:
#         M = cv2.moments(i)
#         cx = M["m10"] / M["m00"]
#         cy = M["m01"] / M["m00"]
#         points.append((cx, cy))
#     return 

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

# path = '/home/srv-admin/images/items1/1313/*.jpg'
path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\*.jpg'
file_list = peekable(sorted(glob.iglob(path)))

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# video = cv2.VideoWriter('log.mp4',fourcc, 30.3, (640, 480))

if '_V' in file_list.peek():
    bg_v = cv2.imread(next(file_list), 0)
    bg_t = cv2.imread(next(file_list), 0)
else:
    next(file_list)
    bg_v = cv2.imread(next(file_list), 0)
    bg_t = cv2.imread(next(file_list), 0)
    
b_img_v = bg_v.copy()
kernel = np.ones((5,5),np.uint8)

model = YOLO("yolov8x.pt")
class_dic = model.model.names

for i in file_list:
    try:
        if '_V' in i and '_T' in file_list.peek():
            img_v = cv2.imread(i, 0)
            img_v_color = cv2.imread(i)
            diff_v = cv2.absdiff(img_v, bg_v)
            _, img_th_v = cv2.threshold(diff_v,30,255,cv2.THRESH_BINARY)
            dilate_v = cv2.dilate(img_th_v,kernel,iterations=5)
            erode_v = cv2.erode(dilate_v, kernel, 2)

            img_t = cv2.imread(next(file_list), 0)
            diff_t = cv2.absdiff(img_t, bg_t)
            affined_t = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
            _, img_th_t = cv2.threshold(affined_t,12,255,cv2.THRESH_BINARY)
            dilate_t = cv2.dilate(img_th_t,kernel,3)
            erode_t = cv2.erode(dilate_t, kernel, 2)

            touch_region = cv2.subtract(erode_t, erode_v)
            contours, _ = cv2.findContours(touch_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = list(filter(lambda x: cv2.contourArea(x) > 80, contours))
            # cv2.drawContours(img_v_color, contours, -1, (0,0,255), 2)
            # cv2.imshow('img', img_v_color)
            # cv2.waitKey(1)
            
            points = []
            for cnt in contours:
                M = cv2.moments(cnt)
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                points.append((cx, cy))
            
            
            
            if (feature_compare(b_img_v, img_v)<12):
                bg_v = img_v.copy()
                # print('更新')
            else: b_img_v = img_v.copy()
            
            
            # pred = model.predict(img_v, classes=[67, 73, 76])
            # frame = pred[0].plot()
            # bbox = pred[0].boxes.xyxy.cpu().numpy()
            # classes = pred[0].boxes.cls.cpu().numpy()

            # polygon = []
            # for x1, y1, x2, y2 in bbox:
            #     polygon.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]))

            # with open("example.txt", "w") as f:
            #     for pt in points:
            #         for poly, cls in zip(polygon, classes):
            #             if cv2.pointPolygonTest(poly, pt, False) >= 0:
            #                 # print(class_dic[cls])
            #                 f.write(str(datetime.datetime.now()) + ", " + "at: table, " + class_dic[cls])
            

    except StopIteration:
        break

# video.release()




# fig, ax = plt.subplots()  
# plt.axis([0,640,0,480])
# for i in polygon:
#     ax.add_patch(plt.Polygon(i,fill=False))


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
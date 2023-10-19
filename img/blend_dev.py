import numpy as np
import cv2
from matplotlib import pyplot as plt
# https://github.com/chenmingxiang110/AugNet

def minus_to_zero(data):
    img = data.copy()
    img[img < 0] = 0
    return img

# 画像読み込み
b_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354826_T.jpg"
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131356859_T.jpg"
# a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358259_T.jpg"
b_visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354805_V.jpg"
a_visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131356836_V.jpg"
# a_visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358236_V.jpg"

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])


bg_t = cv2.imread(b_thermal_path, 0)
bg_v = cv2.imread(b_visible_path, 0)
img_t = cv2.imread(a_thermal_path, 0)
img_v = cv2.imread(a_visible_path, 0)

diff_t = cv2.absdiff(bg_t, img_t)
diff_v = cv2.absdiff(bg_v, img_v)

kernel = np.ones((5,5),np.uint8)
thermal = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
_, img_th_t = cv2.threshold(thermal,10,255,cv2.THRESH_BINARY)
# blur_t = cv2.GaussianBlur(img_th_t, (9,9), 0)
dilate_t = cv2.dilate(img_th_t,kernel,iterations=2)
erode_t = cv2.erode(dilate_t,kernel,2)


_, img_th_v = cv2.threshold(diff_v,30,255,cv2.THRESH_BINARY)
dilate_v = cv2.dilate(img_th_v,kernel,iterations=2)
erode_v = cv2.erode(dilate_v, kernel, iterations=2)



# アルゴリズムを指定しない場合
# n_labels, labels = cv2.connectedComponents(erode_t)
# retval_v, labels_v, stats_v, centroids_v = cv2.connectedComponentsWithStats(erode_v)
# retval_t, labels_t, stats_t, centroids_t = cv2.connectedComponentsWithStats(erode_t)

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.imshow(labels_t)
# plt.show()

zero_img = np.zeros(erode_t.shape)
print(zero_img.dtype)
cv2.imshow('test', zero_img)
cv2.waitKey()

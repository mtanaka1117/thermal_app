import glob
import cv2
import numpy as np
from more_itertools import peekable
# https://qiita.com/penpenpen/items/acfa194cee45190cd8f3
# https://qiita.com/best_not_best/items/c9497ffb5240622ede01

def minus_to_zero(data):
    img = data.copy()
    img[img < 0] = 0
    return img

def hist_compare(img1, img2):
    '''
    画像のヒストグラムを比較する関数
    '''
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

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

# def touch_region(img_t, img_v):
#     '''
    
#     '''
#     ct_t = np.count_nonzero(img_t)
#     ct_v = np.count_nonzero(img_v)
#     try:
#         if ct_t/ct_v > 0.2:
#             img = np.zeros(img_t.shape).astype(np.uint8)
#         else:
#             img = img_t.astype(np.uint8)
#     except ZeroDivisionError:
#         img = img_t.astype(np.uint8)
#     return img

path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\*.jpg'
file_list = peekable(glob.iglob(path))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('diff.mp4',fourcc, 30.3, (640, 480))

if '_V' in file_list.peek():
    bg_v = cv2.imread(next(file_list), 0)
    bg_t = cv2.imread(next(file_list), 0)
else:
    next(file_list)
    bg_v = cv2.imread(next(file_list), 0)
    bg_t = cv2.imread(next(file_list), 0)
    
affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

# b_img_t = bg_t.copy()
b_img_v = bg_v.copy()
kernel = np.ones((5,5),np.uint8)

for i in file_list:
    try:
        if '_V' in i and '_T' in file_list.peek():
            img_v = cv2.imread(i, 0)
            img_v_color = cv2.imread(i)
            diff_v = cv2.absdiff(img_v, bg_v)
            _, img_th_v = cv2.threshold(diff_v,30,255,cv2.THRESH_BINARY)
            dilate_v = cv2.dilate(img_th_v,kernel,iterations=6)
            erode_v = cv2.erode(dilate_v,kernel,2)
            
            img_t = cv2.imread(next(file_list), 0)
            diff_t = cv2.absdiff(img_t, bg_t)
            affined_t = cv2.warpAffine(diff_t, affine_matrix, (img_v.shape[1], img_v.shape[0]))
            _, img_th_t = cv2.threshold(affined_t,10,255,cv2.THRESH_BINARY)
            dilate_t = cv2.dilate(img_th_t,kernel,iterations=2)
            erode_t = cv2.erode(dilate_t,kernel,2)
            
            # mask = minus_to_zero(img_th_t - dilate_v)
            mask = minus_to_zero(erode_t - erode_v)
            # mask = touch_region(erode_t, erode_v)
            mask_inv = cv2.bitwise_not(mask)
            back = cv2.bitwise_and(img_v_color, img_v_color, mask_inv)
            cut = cv2.bitwise_and(mask, mask, mask)
            cut = cv2.cvtColor(cut, cv2.COLOR_GRAY2BGR)
            paste = cv2.add(back, cut)
            
            # if (hist_compare(b_img_v, img_v)>0.9992): 
            if (feature_compare(b_img_v, img_v)<12):
                bg_v = img_v.copy()
                print('更新')
            else: b_img_v = img_v.copy()
            
            # if (np.array_equal(b_img_t, img_t)): bg_t = img_t.copy()
            # else: b_img_t = img_t.copy()
            video.write(paste)
        
    except StopIteration:  #ここで例外処理
        break

video.release()



        

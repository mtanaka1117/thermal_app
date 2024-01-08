import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_compare(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    img1_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    img2_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2_hist = cv2.normalize(img2_hist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return cv2.compareHist(img1_hist, img2_hist, 0)

if __name__ == "__main__":
    img1 = cv2.imread("./thumbnails/67.png")
    img2 = cv2.imread("./thumbnails/67_2.png")
    
    hist = hist_compare(img1, img2)
    print("類似度：" + str(hist))
    


    
    # vtr = imgsim.Vectorizer()
    # vec1 = vtr.vectorize(img1)
    # vec2 = vtr.vectorize(img2)

    # dist = imgsim.distance(vec1, vec2)
    # print("distance =", dist)
    
    
    
    # akaze = cv2.SIFT_create()                                

    # # 特徴量の検出と特徴量ベクトルの計算
    # kp1, des1 = akaze.detectAndCompute(img1, None)
    # kp2, des2 = akaze.detectAndCompute(img2, None)

    # # Brute-Force Matcher生成
    # bf = cv2.BFMatcher()

    # # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    # matches = bf.knnMatch(des1, des2, k=2)

    # # データを間引きする
    # ratio = 0.8
    # good = []
    # for m, n in matches:
    #     if m.distance < ratio * n.distance:
    #         good.append([m])

    # # 対応する特徴点同士を描画
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    # cv2.imwrite("match.png", img3)
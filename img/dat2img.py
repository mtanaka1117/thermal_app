import cv2
import numpy as np
from PIL import Image

b_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354592_T.dat"
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.dat"

# 各画素16bit値、バイナリデータから画像を取得
# np.frombeffer()はbufferをarrayとして解釈 https://deepage.net/features/numpy-frombuffer.html
def thermal_image(path, path2):
    with open(path, "rb") as f:
        img_binary = f.read()
    data = np.frombuffer(img_binary, dtype = np.uint16).reshape([512, 640])/10 - 273.15

    with open(path2, "rb") as f:
        img_binary2 = f.read()
    data2 = np.frombuffer(img_binary2, dtype = np.uint16).reshape([512, 640])/10 - 273.15

    img_diff = cv2.absdiff(data, data2)
    _, img_th = cv2.threshold(img_diff,20,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    img_th = cv2.dilate(img_th, kernel, 1)
    contours, hierarchy = cv2.findContours(img_th.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    
    for i, contour in enumerate(contours):
        # 重心の計算
        m = cv2.moments(contour)
        x,y= m['m10']/m['m00'] , m['m01']/m['m00']
        print(f"Weight Center = ({x}, {y})")
    
    # image = Image.fromarray(img_th.astype(np.uint8))
    


data = thermal_image(b_thermal_path, a_thermal_path)
# image = Image.fromarray(img.astype(np.uint8))
# image.show()






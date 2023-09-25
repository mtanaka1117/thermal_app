import cv2
import numpy as np
from PIL import Image

b_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354592_T.dat"
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.dat"

# 閾値で二値化
def binalization(data, threshold):
    img = data.copy()
    img[data.astype(np.uint8) < threshold] = 0
    img[data.astype(np.uint8) >= threshold] = 255
    return img

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
    image = Image.fromarray(img_th.astype(np.uint8))
    # image.show()
    image.save(r"C:\Users\tnkmo\OneDrive\デスクトップ\thermal_app\img\result.png")
    # return 

    

data = thermal_image(b_thermal_path, a_thermal_path)
# image = Image.fromarray(img.astype(np.uint8))
# image.show()






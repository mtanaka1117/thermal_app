import cv2
from PIL import Image
import numpy as np

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
    # image = Image.fromarray(img_th.astype(np.uint8))
    return img_th

visible_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358202_V.jpg"
image1 = Image.open(visible_path)

affine_matrix = np.array([[ 1.15775321e+00, 2.06036561e-02, -8.65530736e+01],
                        [-3.59868529e-02, 1.16843440e+00, -4.39524932e+01]])

b_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131354592_T.dat"
a_thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358192_T.dat"
image2 = thermal_image(b_thermal_path, a_thermal_path)

image2 = cv2.warpAffine(image2, affine_matrix, (image2.shape[1], image2.shape[0]))
image2 = Image.fromarray(image2).crop((0, 0, 640, 480)).convert('1')

image1.paste(image2, (0,0), image2)
image1.show()

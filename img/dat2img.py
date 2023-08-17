import cv2
import os
import numpy as np
from struct import unpack_from
from PIL import Image
import matplotlib.pyplot as plt

def clip(pix):
    if pix < 0:
        pix = 0
    elif pix > 255:
        pix = 255
    return pix

def yuv_to_rgb(Y, U, V):
    R = clip(( 298 * (Y - 16) + 409 * (V - 128) + 128) >> 8)
    G = clip(( 298 * (Y - 16) - 100 * (U - 128) + 208 * (V - 128) - 128) >> 8)
    B = clip(( 298 * (Y - 16) + 516 * (V - 128) + 128) >> 8)
    return R, G, B

thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131357759_T.dat"
rgb_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131359932_V.dat"
output_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\test.jpg"

# with open(rgb_path, "rb") as f:
#     img_binary = f.read()
# data = np.frombuffer(img_binary, dtype = np.uint8).reshape([-1, 2])

# height = 480
# width = 640
# y = data[:, 1]
# u = data[:, 0].reshape([-1, 2])[:, 0]
# v = data[:, 0].reshape([-1, 2])[:, 1]

# rgb_list = np.zeros((height, width, 3))

# for i in range(0, height):
#     for j in range(0, width):
#         r, g, b = yuv_to_rgb(y[i*j], u[int(i*j/2)], v[int(i*j/2)])
#         rgb_list[i][j][0] = r
#         rgb_list[i][j][1] = g
#         rgb_list[i][j][2] = b

# print(rgb_list)
# image = Image.fromarray(rgb_list.astype(np.uint8))
# image.show()

# cv2.imshow('img', rgb_list)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# UYVYデータをRGBに変換する関数
def uyvy_to_rgb(uyvy_data, width, height):
    uyvy_image = np.frombuffer(uyvy_data, dtype=np.uint8).reshape((height, width * 2))
    bgr_image = cv2.cvtColor(uyvy_image, cv2.COLOR_YUV2BGR_UYVY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

# UYVYデータファイルのパス
file_path = rgb_path
# ビデオフレームの解像度
width = 640
height = 480

with open(file_path, 'rb') as f:
    uyvy_data = f.read()

data = np.frombuffer(uyvy_data, dtype = np.uint8).reshape([height, width*2])
# bgr_image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_UYVY)
    
# rgb_image = uyvy_to_rgb(uyvy_data, width, height)

# 変換されたRGB画像を表示
cv2.imshow('UYVY to RGB', data)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 閾値で二値化
def binalization(data, threshold):
    img = data.copy()
    img[data.astype(np.uint8) < threshold] = 0
    img[data.astype(np.uint8) >= threshold] = 255
    return img

# 各画素16bit値、バイナリデータから画像を取得
# np.frombeffer()はbufferをarrayとして解釈 https://deepage.net/features/numpy-frombuffer.html
def thermal_image(path):
    with open(path, "rb") as f:
        img_binary = f.read()
    data = np.frombuffer(img_binary, dtype = np.uint16).reshape([512, 640])/10 - 273.15
    # image = Image.fromarray(data.astype(np.uint8))
    # image.show()
    return data

    

# data = thermal_image(thermal_path)
# # 二値化
# img = binalization(data, 100)
# image = Image.fromarray(img.astype(np.uint8))
# image.show()






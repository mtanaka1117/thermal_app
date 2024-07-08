import glob
import cv2
import numpy as np
from more_itertools import peekable
import os
import matplotlib.pyplot as plt
import japanize_matplotlib

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

# 画像上のクリックした点の座標を取る
def get_coord(path):
    img = cv2.imread(path)
    cv2.imshow('test', img)
    cv2.setMouseCallback('test', onMouse)
    cv2.waitKey()


# path = r"C:\Users\tnkmo\Downloads\0617\table\20240617_140247277_T.jpg"
# get_coord(path)

# 166 135

path = r"C:\Users\tnkmo\Downloads\thermal\*_T.dat"
file_list = peekable(sorted(glob.iglob(path)))

TEMP_FIXED = 24.0
TEMP_MIN = -53.16


y = []
for file in file_list:
    # if i < 1800:
    with open(file, "rb") as f:
        img_binary = f.read()
        data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
        
        y.append(np.mean(data))
        
        # y.append(data[166][135])
        
        # y.append(data[166][135] - TEMP_MIN + TEMP_FIXED)
        
# print(min(y))
# print(max(y))
# print(len(y))


fig = plt.figure()
plt.plot(y, label="scissor", color="red")

plt.xlabel("時間（分）")
plt.ylabel("温度（℃）")
# plt.xticks([0, 1800, 3600, 5400, 7200], [0, 1, 2, 3, 4])

plt.tight_layout()
# plt.legend()
# plt.savefig('./thesis_image/temp_change.jpg')
plt.show()

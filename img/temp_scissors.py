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


# path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131348325_T.jpg"
# get_coord(path)

# 180 447

path = r"C:\Users\tnkmo\Downloads\scissors\*_T.dat"
file_list = peekable(sorted(glob.iglob(path)))

TEMP_FIXED = 24.0
TEMP_MIN = -56.87

y = []
for file in file_list:
    # if i < 1800:
        with open(file, "rb") as f:
            img_binary = f.read()
            data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
            # y.append(data[180][447])
            
            y.append(data[180][447] - TEMP_MIN + TEMP_FIXED)
        
# print(min(y))
# print(max(y))



fig = plt.figure()
plt.plot(y, label="scissor", color="red")

plt.xlabel("時間（秒）")
plt.ylabel("温度（℃）")
plt.xticks([0, 150, 300, 450], [0, 5, 10, 15])

plt.tight_layout()
plt.legend()
# plt.savefig('./thesis_image/temp_change.jpg')
plt.show()

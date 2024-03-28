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

# y   x
# 138 242
# 266 256
# 458 221
# 532 204


# path = r"C:\Users\tnkmo\Downloads\temp\20240112_1510\20240112_151010520_T.jpg"
# get_coord(path)


# HAND_TEMP_BOOK = -46.94
# HAND_TEMP_PAPER = -47.44
# HAND_TEMP_CUP = -50.08
# HAND_TEMP_PLASTIC = -49.7
# HAND_TEMP_FIXED = 36

# HAND_TEMP_BOOK = -59.22
# HAND_TEMP_PAPER = -59.69

HAND_TEMP_BOOK = -59.51
HAND_TEMP_PAPER = -59.92
HAND_TEMP_CUP = -59.96
HAND_TEMP_PLASTIC = -59.98
HAND_TEMP_FIXED = 24

path = r"C:\Users\tnkmo\Downloads\temp\*\*_T.dat"
file_list = peekable(sorted(glob.iglob(path)))


book = []
paper = []
cup = []
plastic = []
for i, file in enumerate(file_list):
    if i < 1800:
        with open(file, "rb") as f:
            img_binary = f.read()
            data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
            # book.append(data[186][203])
            # paper.append(data[211][337])
            # cup.append(data[221][458])
            # plastic.append(data[204][532])
            
            # book.append(data[242][138] - HAND_TEMP_BOOK + HAND_TEMP_FIXED)
            # paper.append(data[256][266] - HAND_TEMP_PAPER + HAND_TEMP_FIXED)
            
            book.append(data[186][203] - HAND_TEMP_BOOK + HAND_TEMP_FIXED)
            paper.append(data[211][337] - HAND_TEMP_PAPER + HAND_TEMP_FIXED)
            cup.append(data[221][458] - HAND_TEMP_CUP + HAND_TEMP_FIXED)
            plastic.append(data[204][532] - HAND_TEMP_PLASTIC + HAND_TEMP_FIXED)

# for y in list([book, paper, cup, plastic]):        
#     print(min(y))
#     print(max(y))

# fig, ax = plt.subplots(2, 2)
# ax[0,0].plot(book)
# ax[0,0].set_title('book')
# ax[0,1].plot(paper)
# ax[0,1].set_title('paper')
# ax[1,0].plot(cup)
# ax[1,0].set_title('cup')
# ax[1,1].plot(plastic)
# ax[1,1].set_title('plastic')

# for ax in fig.get_axes():
#     ax.set_ylim(23, 37)
#     ax.set_xlabel("時間（秒）")
#     ax.set_ylabel("温度（℃）")
#     ax.set_xticks([0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800], [0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 105, 120])


fig = plt.figure()
plt.plot(book, label="book", color="red")
plt.plot(paper, label="paper", color="orange")
plt.plot(cup, label="stainless cup", color="blue")
plt.plot(plastic, label="plastic", color="green")

plt.xlabel("時間（秒）")
plt.ylabel("温度（℃）")
plt.xticks([0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800], [0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 105, 120])

plt.tight_layout()
plt.legend()
plt.savefig('./thesis_image/temp_change.jpg')
plt.show()

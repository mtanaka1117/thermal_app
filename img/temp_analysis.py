import glob
import cv2
import numpy as np
from more_itertools import peekable
import os
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import norm
import statistics
import math
import pandas as pd

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
    if i < 5400:
        with open(file, "rb") as f:
            img_binary = f.read()
            data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
            # book.append(data[186][203])
            # paper.append(data[211][337])
            # cup.append(data[221][458])
            # plastic.append(data[204][532])
            
            # 手のひら温度
            # book.append(data[242][138] - HAND_TEMP_BOOK + HAND_TEMP_FIXED)
            # paper.append(data[256][266] - HAND_TEMP_PAPER + HAND_TEMP_FIXED)
            
            # 指先温度
            book.append(data[186][203] - HAND_TEMP_BOOK + HAND_TEMP_FIXED)
            paper.append(data[211][337] - HAND_TEMP_PAPER + HAND_TEMP_FIXED)
            cup.append(data[221][458] - HAND_TEMP_CUP + HAND_TEMP_FIXED)
            plastic.append(data[204][532] - HAND_TEMP_PLASTIC + HAND_TEMP_FIXED)

# for y in list([book, paper, cup, plastic]):        
#     print(min(y))
#     print(max(y))
    
# 平均と標準偏差を求める
# print(statistics.mean(plastic))
# print(math.sqrt(statistics.pvariance(plastic)))


# x=np.linspace(23, 25.5)
# y=norm(loc=mu,scale=sigma).pdf(x)
# plt.plot(x, y)
# plt.axvline(x=mu-2*sigma, color="green", ymax=0.3*y.max(), label="±2σ")
# plt.axvline(x=mu+2*sigma, color="green", ymax=0.3*y.max(), label="±2σ")
# plt.grid()
# plt.show()


# t,   mean,  std
# 112, 24.11, 0.21
# 115, 24.10, 0.10
# 125, 24.20, 0.21
# 113, 24.12, 0.07

book_mu = 24.11
book_sigma = 0.21
book_two_sigma = book_mu+2*book_sigma
book_one_sigma = book_mu+book_sigma

paper_mu = 24.10
paper_sigma = 0.10
paper_two_sigma = paper_mu+2*paper_sigma
paper_one_sigma = paper_mu+paper_sigma

cup_mu = 24.20
cup_sigma = 0.21
cup_two_sigma = cup_mu+2*cup_sigma
cup_one_sigma = cup_mu+cup_sigma

plastic_mu = 24.12
plastic_sigma = 0.07
plastic_two_sigma = plastic_mu+2*plastic_sigma
plastic_one_sigma = plastic_mu+plastic_sigma

book_pd = pd.Series(book)
paper_pd = pd.Series(paper)
cup_pd = pd.Series(cup)
plastic_pd = pd.Series(plastic)


fig, ax = plt.subplots(2, 2, figsize=(8, 6))
ax[0,0].plot(book_pd.rolling(10).mean(), label="book", color="red")
ax[0,0].axhline(y=book_mu, xmin=0, xmax=0.98, color="darkorange", label="μ")
ax[0,0].axhline(y=book_one_sigma, xmin=0, xmax=0.98, color="blue", label="+1σ")
ax[0,0].axhline(y=book_two_sigma, xmin=0, xmax=0.98, color="green", label="+2σ")
ax[0,0].set_title('book', size=16)

ax[0,1].plot(paper_pd.rolling(10).mean(), label="paper", color="red")
ax[0,1].axhline(y=paper_mu, xmin=0, xmax=0.98, color="darkorange", label="μ")
ax[0,1].axhline(y=paper_one_sigma, xmin=0, xmax=0.98, color="blue", label="+1σ")
ax[0,1].axhline(y=paper_two_sigma, xmin=0, xmax=0.98, color="green", label="+2σ")
ax[0,1].set_title('paper', size=16)

ax[1,0].plot(cup_pd.rolling(10).mean(), label="stainless", color="red")
ax[1,0].axhline(y=cup_mu, xmin=0, xmax=0.98, color="darkorange", label="μ")
ax[1,0].axhline(y=cup_one_sigma, xmin=0, xmax=0.98, color="blue", label="+1σ")
ax[1,0].axhline(y=cup_two_sigma, xmin=0, xmax=0.98, color="green", label="+2σ")
ax[1,0].set_title('stainless', size=16)

ax[1,1].plot(plastic_pd.rolling(10).mean(), label="plastic", color="red")
ax[1,1].axhline(y=plastic_mu, xmin=0, xmax=0.98, color="darkorange", label="μ")
ax[1,1].axhline(y=plastic_one_sigma, xmin=0, xmax=0.98, color="blue", label="+1σ")
ax[1,1].axhline(y=plastic_two_sigma, xmin=0, xmax=0.98, color="green", label="+2σ")
ax[1,1].set_title('plastic', size=16)

for ax in fig.get_axes():
    ax.set_ylim(23.8, 26.3)
    ax.set_xlabel("時間（秒）", fontsize=16)
    ax.set_ylabel("温度（℃）", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_xticks([0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400], [0, 20, 40, 60, 80, 100, 120, 140, 160, 180], rotation=30)
    ax.legend()


plt.tight_layout()
plt.savefig('./thesis_image/temp_graph.jpg')
plt.show()

# fig = plt.figure()
# plt.plot(book_pd.rolling(10).mean(), label="book", color="red")
# plt.plot(paper_pd.rolling(10).mean(), label="paper", color="red")
# plt.plot(cup_pd.rolling(10).mean(), label="stainless cup", color="red")
# plt.plot(plastic_pd.rolling(10).mean(), label="plastic", color="red")


# plt.axhline(y=book_one_sigma, xmin=0.065, xmax=0.98, color="blue", label="+1σ")
# plt.axhline(y=paper_one_sigma, xmin=0.065, xmax=0.98, color="blue", label="+1σ")
# plt.axhline(y=cup_one_sigma, xmin=0.065, xmax=0.98, color="blue", label="+1σ")
# plt.axhline(y=plastic_one_sigma, xmin=0.065, xmax=0.98, color="blue", label="+1σ")


# plt.axhline(y=book_two_sigma, xmin=0.065, xmax=0.98, color="green", label="+2σ")
# plt.axhline(y=paper_two_sigma, xmin=0.065, xmax=0.98, color="green", label="+2σ")
# plt.axhline(y=cup_two_sigma, xmin=0.065, xmax=0.98, color="green", label="+2σ")
# plt.axhline(y=plastic_two_sigma, xmin=0.065, xmax=0.98, color="green", label="+2σ")


# plt.rcParams["font.size"] = 20
# plt.xlabel("時間（秒）", {'fontsize':18})
# plt.ylabel("温度（℃）", {'fontsize':18})
# plt.xticks([0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400], [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
# plt.ylim(23.5, 26.5)

# plt.tick_params(labelsize=14)
# plt.tight_layout()
# plt.legend()
# plt.savefig('./thesis_image/book.jpg')
# plt.show()

import glob
import cv2
import numpy as np
from more_itertools import peekable
import matplotlib.pyplot as plt
import japanize_matplotlib

path_img = r"C:\Users\tnkmo\Downloads\table\20240701_152704787_T.jpg"
path_dat = r"C:\Users\tnkmo\Downloads\table\20240701_152704787_T.dat"

path = r"C:\Users\tnkmo\Downloads\table\*_T.dat"
file_list = peekable(sorted(glob.iglob(path)))

img = cv2.imread(path_img)


firstLoop = True
y = []
for file in file_list:
    with open(file, "rb") as f:
        img_binary = f.read()
        data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
        
        if firstLoop:
            THRES_TEMP = np.mean(data)
            firstLoop = False
            y.append(np.mean(data))
        
        if not firstLoop:
            data_fixed = data[data <= THRES_TEMP + 0.2]
            if data_fixed.size > 0:
                temp_fixed = np.mean(data_fixed)
                diff = abs(THRES_TEMP - temp_fixed)
                data = data + diff
                y.append(np.mean(data))
            else:
                y.append(THRES_TEMP)

fig = plt.figure()
plt.plot(y, label="scissor", color="red")

plt.xlabel("時間（分）")
plt.ylabel("温度（℃）")
# plt.xticks([0, 1800, 3600, 5400, 7200], [0, 1, 2, 3, 4])

plt.tight_layout()
# plt.legend()
# plt.savefig('./thesis_image/temp_change.jpg')
plt.show()


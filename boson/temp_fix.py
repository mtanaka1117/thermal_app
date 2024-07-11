import glob
import numpy as np
from more_itertools import peekable
import matplotlib.pyplot as plt
import japanize_matplotlib
import os
from tqdm import tqdm

path = "./data/0709/table/*_T.dat"
file_list = peekable(sorted(glob.iglob(path)))

firstLoop = True
y = []
z = []

for file in tqdm(file_list):
    with open(file, "rb") as f:
        img_binary = f.read()
        data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/100 - 273.15
        # z.append(np.mean(data))
        
        if firstLoop:
            THRES_TEMP = np.mean(data)
            firstLoop = False
            # y.append(THRES_TEMP)
        
        if not firstLoop:
            data_fixed = data[data <= THRES_TEMP + 0.5]
            if data_fixed.size > 0:
                temp_fixed = np.mean(data_fixed)
                diff = THRES_TEMP - temp_fixed
                data = (data + diff + 273.15) * 100
                data = data.astype(np.uint16)
                with open('./data/0709/table/{}'.format(os.path.basename(file)), 'wb') as f:
                    f.write(data.tobytes())

                # y.append(np.mean(data))

            else:
                data = (data + 273.15) * 100
                data = data.astype(np.uint16)
                with open('./data/0709/table/{}'.format(os.path.basename(file)), 'wb') as f:
                    f.write(data.tobytes())


# fig = plt.figure()
# plt.plot(y, label="fixed", color="red")
# plt.plot(z, label="original", color="blue")
# plt.hlines(THRES_TEMP, 0, 5000, color="orange")

# plt.xlabel("フレーム")
# plt.ylabel("温度（℃）")

# plt.tight_layout()
# plt.legend()
# plt.show()


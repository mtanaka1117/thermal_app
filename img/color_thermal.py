import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131356192_T.dat"
# thermal_path = r"C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358326_T.dat"

# "C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131356170_V.jpg"
# "C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\20230807_131358300_V.jpg"

TEMP_MIN = 60.0
TEMP_MAX = 160.0

def thermal_image(path, output_path):
    with open(path, "rb") as f:
        img_binary = f.read()
        data = np.frombuffer(img_binary, dtype = np.uint16).reshape([512, 640])/10 - 273.15
        data = data.astype(np.uint8)
        # print(np.amax(data), np.amin(data))
        
        fig = plt.figure()   
        sns.set()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = sns.heatmap(data, vmin=TEMP_MIN, vmax=TEMP_MAX, cmap="jet", cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.savefig(output_path)
        plt.show()


thermal_image(thermal_path, '')

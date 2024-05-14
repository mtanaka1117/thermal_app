import numpy as np
from PIL import Image


with open('boson.dat', 'rb') as f:
    img_binary = f.read()
data = np.frombuffer(img_binary, dtype=np.uint16).reshape([512, 640])/10 - 273.15
# data = 255*(data - data.min())/(data.max()-data.min())
image = Image.fromarray(data.astype(np.uint8))
image.show()


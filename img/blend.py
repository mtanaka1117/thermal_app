import cv2
from PIL import Image
import numpy as np


rgb_path = 'cropped_image.jpg'
img = cv2.imread(rgb_path)

affine_matrix = np.array([[ 1.53104533e+00, -1.88203939e-02,  3.54805456e+00],
                        [-3.01498812e-03,  1.49792624e+00,  2.71967074e+01]])

thermal_path = r'C:\Users\tnkmo\OneDrive\デスクトップ\thermal_app\img\20241223_203142717_T.jpg'
img2 = cv2.imread(thermal_path)

blend_img = cv2.warpAffine(img2, affine_matrix, (img.shape[1], img.shape[0]))

img.paste(img2, (0,0), img2)
cv2.imshow('test', img)

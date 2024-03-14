import glob
import cv2
import numpy as np
from more_itertools import peekable

path = r'C:\Users\tnkmo\Downloads\temp\*\*_T.jpg'
# path = '/home/srv-admin/images/items1/1313/*_V.jpg'
file_list = peekable(sorted(glob.iglob(path)))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('temp.mp4',fourcc, 30.3, (640, 512))

for i in file_list:
    img = cv2.imread(i)
    video.write(img)
    
video.release()
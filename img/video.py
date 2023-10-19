import glob
import cv2
import numpy as np
from more_itertools import peekable

path = r'C:\Users\tnkmo\Downloads\items1\items1\20230807_1313\*_V.jpg'
file_list = peekable(glob.iglob(path))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('video_V.mp4',fourcc, 30.3, (640, 480))

for i in file_list:
    img = cv2.imread(i)
    video.write(img)
    
video.release()
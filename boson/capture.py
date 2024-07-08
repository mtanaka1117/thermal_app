import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from flirpy.camera.boson import Boson
import numpy as np
import datetime as dt
import argparse

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

TEMP_MIN = -60.0
TEMP_MAX = -40.0

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode")
args = parser.parse_args()

cap_v = cv2.VideoCapture(1)
cap_t = Boson()

if args.mode == "calibration":
    while True:
        img_t = cap_t.grab()
        _, img_v = cap_v.read()
         
        img_t = img_t.astype(np.uint16).reshape([512, 640])/100 - 273.15
        img_t = 255.0*(img_t - img_t.min())/(img_t.max() - img_t.min())
        img_t = img_t.astype(np.uint8)
        
        img_col = cv2.applyColorMap(img_t, cv2.COLORMAP_JET)
        img_col = hconcat_resize_min([img_col, img_v, img_col])
        mergeImg = np.hstack((img_col, img_v))
        
        cv2.imshow('Boson', mergeImg)
        
        if cv2.waitKey(1) == 27:
            cv2.imwrite('./data/0701/test_table_T.jpg', img_t)
            cv2.imwrite('./data/0701/test_table_V.jpg', img_v)
            break  # esc to quit
            
    cv2.destroyAllWindows()

if args.mode == "capture":
    while True:
        img_t = cap_t.grab()
        _, img_v = cap_v.read()
        
        now = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        with open('./data/0701/table/{}_T.dat'.format(now), 'wb') as f:
            f.write(img_t)
        cv2.imwrite('./data/0701/table/{}_V.jpg'.format(now), img_v)
        
        img_t = img_t.astype(np.uint16).reshape([512, 640])/100 - 273.15
        img_t = 255.0*(img_t - TEMP_MIN)/(TEMP_MAX - TEMP_MIN)
        img_t = img_t.astype(np.uint8)
        
        img_col = cv2.applyColorMap(img_t, cv2.COLORMAP_JET)
        img_col = hconcat_resize_min([img_col, img_v, img_col])
        mergeImg = np.hstack((img_col, img_v))
        
        cv2.imshow('Boson', mergeImg)
        
        if cv2.waitKey(1) == 27:
            break  # esc to quit
            
    cv2.destroyAllWindows()



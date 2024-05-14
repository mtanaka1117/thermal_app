import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from flirpy.camera.boson import Boson
import numpy as np
import datetime as dt

TEMP_MIN = -60.0
TEMP_MAX = -40.0

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

cap_v = cv2.VideoCapture(2)
cap_t = Boson()

while True:
    img_t = cap_t.grab().astype(np.float32)
    _, img_v = cap_v.read()
    
    # now = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    # with open('./thermal/{}_T.dat'.format(now), 'wb') as f:
    #     f.write(img_t)
    # cv2.imwrite('./thermal/{}_V.dat'.format(now), img_v)
    
    img = img.astype(np.uint16).reshape([512, 640])/100 - 273.15
    img = 255.0*(img - TEMP_MIN)/(TEMP_MAX - TEMP_MIN)
    img = img.astype(np.uint8)
    
    img_col = cv2.applyColorMap(img_t, cv2.COLORMAP_INFERNO)
    img_col = hconcat_resize_min([img_col, img_v, img_col])
    mergeImg = np.hstack((img_col, img_v))
    
    cv2.imshow('Boson', mergeImg)
    
    if cv2.waitKey(1) == 27:
        break  # esc to quit
        
cv2.destroyAllWindows()



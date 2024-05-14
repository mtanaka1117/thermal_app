import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from flirpy.camera.boson import Boson
import numpy as np
import datetime

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

cap_v = cv2.VideoCapture(2)
cap_t = Boson()

while True:
    img_t = cap_t.grab().astype(np.float32)
    # Rescale to 8 bit
    img_t = 255*(img_t - img_t.min())/(img_t.max()-img_t.min())
    img_t = img_t.astype(np.uint8)
    
    _, img_v = cap_v.read()
    
    img_col = cv2.applyColorMap(img_t, cv2.COLORMAP_INFERNO)
    img_col = hconcat_resize_min([img_col, img_v, img_col])
    mergeImg = np.hstack((img_col, img_v))
    
    cv2.imshow('Boson', mergeImg)
        
    # now = datetime.datetime.now()
    # filename_t = './img/' + now.strftime('%Y%m%d_%H%M%S%f') + '_T.jpg'
    # filename_v = './img/' + now.strftime('%Y%m%d_%H%M%S%f') + '_V.jpg'
    # cv2.imwrite(filename_t, img_t)
    # cv2.imwrite(filename_v, img_v)
    # img_t.dump()
    if cv2.waitKey(1) == 27:
        break  # esc to quit
        
cv2.destroyAllWindows()



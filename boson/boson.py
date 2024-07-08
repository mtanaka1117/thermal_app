from flirpy.camera.boson import Boson
import cv2
import numpy as np
import datetime as dt

TEMP_MIN = -70.0
TEMP_MAX = -30.0

with Boson() as camera:
    while True:
        img = camera.grab()
        
        #書き込み
        # now = dt.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        # with open('./thermal/{}_T.dat'.format(now), 'wb') as f:
        #     f.write(img)
        
        img = img.astype(np.uint16).reshape([512, 640])/100 - 273.15
        img = 255.0*(img - TEMP_MIN)/(TEMP_MAX - TEMP_MIN)
        img = img.astype(np.uint8)
        
        cv2.imshow('Boson', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()
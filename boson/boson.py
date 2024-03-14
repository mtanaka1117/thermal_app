from flirpy.camera.boson import Boson
import cv2
import numpy as np

with Boson() as camera:
    while True:
        img = camera.grab().astype(np.float32)
        # Rescale to 8 bit
        img = 255*(img - img.min())/(img.max()-img.min())
        img_col = img.astype(np.uint8)
        # img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        cv2.imshow('Boson', img_col)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

cv2.destroyAllWindows()
import cv2
import numpy as np

def delete_shade(img):
    ksize = 31
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img, (ksize, ksize))
    rij = img/(blur+0.0000001)
    index_1 = np.where(rij >= 0.93)
    index_0 = np.where(rij < 0.93)
    rij[index_0] = 0
    rij[index_1] = 255
    return rij

bg = cv2.imread(r"c:\Users\tnkmo\Downloads\yolo+1\yolo+1\20240112_1450\20240112_145019442_V.jpg")
img = cv2.imread(r"C:\Users\tnkmo\Downloads\yolo+1\yolo+1\20240112_1450\20240112_145049108_V.jpg")
img2 = cv2.imread(r"C:\Users\tnkmo\Downloads\yolo+1\yolo+1\20240112_1450\20240112_145055135_V.jpg")

# kernel = np.ones((5,5), np.uint8)
# bg = delete_shade(bg)
# img = delete_shade(img)
# img2 = delete_shade(img2)
# diff = cv2.bitwise_xor(img, bg)
# diff = cv2.erode(diff, kernel, 1)
# diff = cv2.dilate(diff, kernel, 3)

# diff = cv2.absdiff(img, img2)

gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
_, thres = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)


# dist_transform = cv2.distanceTransform(thres,cv2.DIST_L1,3)
# ret, sure_fg = cv2.threshold(dist_transform,0.08*dist_transform.max(), 255, cv2.THRESH_BINARY)


cv2.imshow('test', thres)
cv2.waitKey()
cv2.destroyAllWindows()
# cv2.imwrite('test2.jpg',dist_transform)
# cv2.imwrite('test.jpg', sure_fg)
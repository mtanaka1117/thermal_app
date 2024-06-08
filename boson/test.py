import cv2

img = cv2.imread("test_table_V.jpg", 0)

_, table_mask = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('test', table_mask)
cv2.waitKey(0)

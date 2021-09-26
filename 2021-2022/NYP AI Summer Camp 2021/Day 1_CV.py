# Data structure to create image arrays
import numpy as np
import matplotlib.pyplot as plt
import cv2

# bw_arr = np.array([
#     [255, 0, 0],
#     [50, 100, 100],
#     [0, 0, 255]
# ])
#
# plt.imshow(bw_arr, cmap="gray")
# plt.show()

'''
b_w ==> 100
rgb ==> [255, 0 ,0]
'''

# rgb_arr = np.array([
#     [[255, 0, 0], [0, 255, 0]],
#     [[0, 0, 0], [10, 23, 40]]
# ])
#
# plt.imshow(rgb_arr)
# plt.show()

# CV2 ==> BGR
# PLT ==> RGB
# img_bgr = cv2.imread("face.jpg")
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# cv2.imshow("Face", img)
# cv2.waitKey()
# plt.imshow(img_rgb)
# plt.show()

#

cls = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("face.jpg")
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
boxes = cls.detectMultiScale(img_bw, 1.3, 5)

for box in boxes:
    x, y, width, height = box
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

cv2.imshow("Detection", img)
cv2.waitKey()
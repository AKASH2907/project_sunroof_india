import cv2
import matplotlib.pyplot as plt
import numpy as np

lat = 28.646088
longs = 77.214157
location = (lat, longs)

img = cv2.imread('polygon.tiff', 0)

ret, thresh = cv2.threshold(img, 127, 255, 0)
im2, cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

corners = cv2.goodFeaturesToTrack(img, 8, 0.01, 8)
#                                   img, no of corners, quality image, minimum

corners = np.int0(corners)
x = []
y = []
for corner in corners:
    x1, y1 = corner.ravel()
    x.append(x1)
    y.append(y1)
    cv2.circle(img, (x1, y1), 3, 0, -1)

# compute the center of the contour
M = cv2.moments(cnts[0])
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

x1 = []
x2 = []
for i in y:
    l = i - cY
    x1.append(l)
    x2.append(lat + l)

# draw the contour and center of the shape on the image
cv2.drawContours(img, [cnts[0]], -1, (0, 255, 0), 2)
cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
print (x)
print (x1)
print (x2)

plt.imshow(img, cmap='gray')
plt.show()

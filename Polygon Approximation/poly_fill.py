from __future__ import print_function
import cv2
from pylab import *
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from skimage.morphology import disk, opening
'''
def createLineIterator(P1, P2, im):
    """
    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    imageH = im.shape[0]
    imageW = im.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = float(dX) / float(dY)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = float(dY) / float(dX)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = im[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer
'''

im = cv2.imread('124.png')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # 1st image use
# img = cv2.bitwise_not(img)

rows, cols = img.shape
white_img = cv2.bitwise_not(np.zeros(im.shape, np.uint8))
white_polygon = cv2.bitwise_not(np.zeros(im.shape, np.uint8))
white_gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)

v = np.median(img)
sigma = 0.33
lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(img, lower_thresh, upper_thresh)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

kmeans = KMeans(n_clusters=20).fit(lines)

for line in kmeans.cluster_centers_:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(white_gray, (x1, y1), (x2, y2), 0, 2)
    '''
    pixels = createLineIterator((x1, y1), (x2, y2), img)
    pixels = pixels.astype(int)
    pixel_len = len(pixels)

    for i in range(pixel_len):
        x, y, intensity = pixels[i]
        if 0 < y < rows - 1 and 0 < x < cols - 1:
            # patch = edges[x - k_row/2: x + k_row/2,
            #         y - k_col/2: y + k_col/2]
            sum_patch = edges[y, x] + edges[y - 1, x] + edges[y + 1, x] + edges[y, x - 1] + edges[y - 1, x - 1] + edges[
                y + 1, x - 1] + edges[y, x + 1] + edges[y - 1, x + 1] + edges[y + 1, x + 1]
            if sum_patch > 0:
                img[y, x] = 255
    
x1 = []
y1 = []
# m = np.sort(m)
# c = np.sort(c)
print (m, c)
for i in range(len(m)-1):
    for j in range(len(c)-1):
        if m[i] - m[i+1] != 0:
            x = ((c[j + 1] - c[j]) / (m[i] - m[i + 1]))
            y = (((m[i]*c[i+1]) - (m[i+1]*c[j])) / (m[i] - m[i+1]))
            if 0 < x < cols and 0 < y < rows:
                x1.append(x)
                y1.append(y)
x1 = np.array(x1)
y1 = np.array(y1)
intersect = (np.column_stack((x1, y1)))
print (intersect)


for i, j, k in zip(r, c, s):
    r[i] = x1*c[j] + y1*s[k]
'''

contours = cv2.findContours(white_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for cnt in contours:
    cv2.drawContours(white_polygon, cnt, 0, 0, -1)
    man = []
    intense = []
    for col in range(cols):
        for row in range(rows):
            if cv2.pointPolygonTest(cnt, (col, row), False) == 1:
                man.append((row, col))
    for k in man:
        intense.append(im[k])
    intensity = mean(intense)
    # print (intensity)
    if intensity > 170:
        cv2.drawContours(white_polygon, [cnt], 0, 0, -1)

    # mean_val = np.mean(cnt)
    # if mean_val < 120:
    #     img[cnt] = 255
# man = map(int, man)
# for i, j in zip(contours, man):
#     if j < 90:
#         cv2.drawContours(img, i, 0, thickness=-1, color=0)
#     else:
#         cv2.drawContours(img, i, 0, thickness=-1, color=255)
# for cnt in contours:
#     cv2.drawContours(im, [cnt], 0, 0, -1)
white_gray1 = cv2.cvtColor(white_polygon, cv2.COLOR_BGR2GRAY)
opened = opening(white_gray1, selem=disk(4))
# kernel_sharp = np.array((
#          [-2, -2, -2],
#          [-2, 17, -2],
#          [-2, -2, -2]), dtype='int')
# opens = cv2.filter2D(opened, -1, kernel_sharp)
opened = Image.fromarray(opened)
opened.save('opened.png')
plt.imshow(opened, cmap='gray')
plt.show()

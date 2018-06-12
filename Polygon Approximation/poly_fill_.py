from __future__ import print_function
import cv2
from pylab import *
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from skimage.morphology import disk, opening


def grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def white_image(im):
    return cv2.bitwise_not(np.zeros(im.shape, np.uint8))


def hough_lines(im):
    original = im
    img = grayscale(im)
    blur = cv2.bilateralFilter(img, 5, sigmaColor=7, sigmaSpace=5)

    kernel_sharp = np.array((
        [-2, -2, -2],
        [-2, 17, -2],
        [-2, -2, -2]), dtype='int')
    sharp = cv2.filter2D(blur, -1, kernel_sharp)

    # White Blank Images for plotting Hough Lines and Polygon shapes
    white_img = white_image(im)
    white_gray = grayscale(white_img)

    # Canny Edge Detection
    v = np.median(sharp)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower_thresh, upper_thresh)

    # Hough Lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
    lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5).fit(lines)

    # Plotting Hough Lines
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

    contour_fill(original, white_gray)


def contour_fill(im, white_gray):
    img = grayscale(im)
    rows, cols = img.shape

    # White Blank Image for polygon drawing
    white_polygon = white_image(im)

    # Contours in the white image with Hough Lines
    contours = cv2.findContours(white_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    for cnt in contours:
        # Drawing the contours on image with Hough Lines
        cv2.drawContours(white_polygon, cnt, 0, 0, -1)

        # Points inside the contour
        inside_point = []
        # Intensity of points inside the contour
        intensity = []

        for col in range(cols):
            for row in range(rows):
                # Point polygon test
                if cv2.pointPolygonTest(cnt, (col, row), False) == 1:
                    inside_point.append((row, col))

        # Intensity of points inside the polygon
        for inside in inside_point:
            intensity.append(im[inside])
        mean_intensity = mean(intensity)

        # Mean Intensity Value threshold
        if mean_intensity > 150:
            cv2.drawContours(white_polygon, [cnt], 0, 0, -1)

    # Filling the polygon
    poly_gaps = cv2.cvtColor(white_polygon, cv2.COLOR_BGR2GRAY)
    opened = opening(poly_gaps, selem=disk(4))
    plt.imshow(opened, cmap='gray')
    plt.show()
    return opened


hough_lines(cv2.imread('erode.png'))
# image = cv2.imread('ob.png')
# hough_lines(image)

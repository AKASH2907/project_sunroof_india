import cv2
from scipy import ndimage
from PIL import Image
from pylab import *
from PIL.Image import *
import math
from shapely.geometry import Polygon

tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
zoom = 18
# 156543.03392804062 for tileSize 256 pixels
originShift = 2 * math.pi * 6378137 / 2.0


def grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def LatLonToPixels(lat, lon):
    # Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913
    mx = lon * originShift / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * originShift / 180.0

    # Converts EPSG:900913 to pyramid pixel coordinates in given zoom level
    res = initialResolution / (2 ** zoom)
    pixel_x = (mx + originShift) / res
    pixel_y = (my + originShift) / res

    PixelsToLatlon(pixel_x, pixel_y)


def PixelsToLatlon(pixel_x, pixel_y):
    # Converts pixel coordinates in given zoom level of pyramid to EPSG:900913
    res = initialResolution / (2 ** zoom)
    mx = pixel_x * res - originShift
    my = pixel_y * res - originShift

    # Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum
    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    print (lat, lon)
    return lat, lon


def PixelsToMeters(px, py):
    res = initialResolution / (2**zoom)
    m_x = px * res - originShift
    m_y = py * res - originShift
    return m_x, m_y


def image_map(im):
    a = Image.getpixel(im, (0, 0))
    if type(a) == int:
        return im
    else:
        rows, cols = im.size
        img_array = np.asarray(im)
        neim = np.zeros((cols, rows))
        for i in range(cols):
            for j in range(rows):
                t = img_array[i, j]
                ts = sum(t) / len(t)
                neim[i, j] = ts
        return neim


def center_corner(im):
    neim = image_map(im)
    img_array = np.asarray(neim, dtype=np.float64)
    ix = ndimage.sobel(img_array, 0)
    iy = ndimage.sobel(img_array, 1)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # Gaussian Filter
    ix2 = ndimage.gaussian_filter(ix2, sigma=2)
    iy2 = ndimage.gaussian_filter(iy2, sigma=2)
    ixy = ndimage.gaussian_filter(ixy, sigma=2)
    rows, cols = img_array.shape
    result = np.zeros((rows, cols))
    r = np.zeros((rows, cols))
    rmax = 0

    # Finding Corners
    for i in range(rows):
        for j in range(cols):
            moments = np.array([[ix2[i, j], ixy[i, j]], [ixy[i, j], iy2[i, j]]], dtype=np.float64)
            r[i, j] = np.linalg.det(moments) - 0.04 * (np.power(np.trace(moments), 2))
            if r[i, j] > rmax:
                rmax = r[i, j]
    for i in range(rows - 1):
        for j in range(cols - 1):
            if r[i, j] > 0.01 * rmax and r[i, j] > r[i - 1, j - 1] and r[i, j] > r[i - 1, j + 1] \
                    and r[i, j] > r[i + 1, j - 1] and r[i, j] > r[i + 1, j + 1]:
                result[i, j] = 1

    pc, pr = np.where(result == 1)  # pr, pc
    # pr, pc - x, y coordinate of corners in the polygon
    coord = []
    x = [pr[0]]
    y = [pc[0]]
    for i in range(1, len(pr)):
        if abs((pr[i-1] - pr[i])) >= 5:
            x.append(pr[i])
        elif abs((pr[i-1] - pr[i])) <= 5 and abs((pc[i-1] - pc[i])) >= 5:
            x.append(pr[i])
    for i in range(1, len(pc)):
        if abs((pc[i-1] - pc[i])) >= 5:
            y.append(pc[i])
        elif abs((pc[i-1] - pc[i])) <= 5 and abs((pr[i-1] - pr[i])) >= 5:
            y.append(pc[i])
    for i, j in zip(x, y):
        coord.append((i, j))
    print (coord)

    coord = np.array(coord)
    # pts = coord.reshape((-1, 1, 2))
    # cv2.polylines(gray, [pts], True, 255)
    # print (PolyArea([166,  57,  57, 117, 119,  57, 56, 174], [80,  81, 147, 149, 216, 220, 284, 276]))
    # print (PolyArea(x, y))
    find_corners(pr, pc)
    return x, y


def center(im):
    gray = np.array(im)
    rows, cols = gray.shape
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    print (cnts[0].shape)
    counts = np.reshape(np.array(cnts[0]), (cnts[0].shape[0], cnts[0].shape[2]))
    print (counts)
    inside_point = 0
    for col in range(cols):
        for row in range(rows):
            if cv2.pointPolygonTest(cnts[0], (col, row), False) == 1:
                inside_point += 1
    print (inside_point * 0.27456)
    # print (counts)
    # compute the center of the contour
    moments = cv2.moments(cnts[0])
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    return center_x, center_y, counts


def find_corners(cornerx, cornery):
    cent_x, cent_y, _ = center(img)
    for x in cornerx:
        long_pixel = x - cent_x
    for y in cornery:
        lat_pixel = y - cent_y
        # LatLonToPixels(lat_pixel, long_pixel)


img = open('2.jpg').convert('L')
gray = np.array(img)
a, b = center_corner(img)
print (a, b)
points = []
cnt = center(img)[2]
print (cnt)
r = []
t = []
for i, j in cnt:
    for x, y in zip(a, b):
        if x-1 <= i <= x+1:
            if y-1 <= j <= y+2:
                r.append(i)
                t.append(j)
                points.append((i, j))
print (points)
points = np.array(points)
pts = points.reshape((-1, 1, 2))
cv2.polylines(gray, [pts], True, 255)
pol = Polygon(points)
print (pol.area)
print (PolyArea(r, t))
meters = []
for i, j in zip(r, t):
    meters.append(PixelsToMeters(i, j))
print (meters)
# meters = np.array(meters)
pp = Polygon(meters)
print (pp.area)

import math

zoom = 19
tileSize = 256
initialResolution = 2 * math.pi * 6378137 / tileSize
# 156543.03392804062 for tileSize 256 pixels
originShift = 2 * math.pi * 6378137 / 2.0
earthc = 6378137 * 2 * math.pi
factor = math.pow(2, zoom)
map_width = 256 * (2 ** zoom)


def LatLonToPixels(lat, lon, zoom):
    """Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"""

    mx = lon * originShift / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)

    my = my * originShift / 180.0
    res = initialResolution / (2 ** zoom)
    px = (mx + originShift) / res
    py = (my + originShift) / res
    return px, py


def MetersToPixels(mx, my, zoom):
    "Converts EPSG:900913 to pyramid pixel coordinates in given zoom level"

    res = initialResolution / (2 ** zoom)
    px = (mx + originShift) / res
    py = (my + originShift) / res
    return px, py


# Dont forget you have to convert your projection to EPSG:900913
# mx = -8237494.4864285  # -73.998672  28.737181, 77.063889
# my = 4970354.7325767  # 40.714728
# mx = 28.657555
# my = 77.174684
# mx = 28.652371
mx, my = 28.740963, 77.115281
# Meter per Pixel is dependent on latitude
MeterPerPixel = math.cos(mx * math.pi/180) * earthc / map_width
print (MeterPerPixel)
# MapWidthDistance = 480 * MeterPerPixel
# ActualMeterPerPixel = MapWidthDistance / imgWidthAfterResize
pixel_x, pixel_y = LatLonToPixels(mx, my, zoom)
print (pixel_x, pixel_y)
x = pixel_x + 140
y = pixel_y + 0
print (140 * MeterPerPixel)


def PixelsToLatlon(px, py, zoom):
    """Converts pixel coordinates in given zoom level of pyramid to EPSG:900913"""

    res = initialResolution / (2 ** zoom)
    mx = px * res - originShift
    my = py * res - originShift
    # """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"""

    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0

    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon


def MetersToLatLon(mx, my):
    """Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"""

    lon = (mx / originShift) * 180.0
    lat = (my / originShift) * 180.0

    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon
llx, lly = PixelsToLatlon(x, y, zoom)
m, n = MetersToLatLon(llx, lly)
print (llx, lly)
print (m, n)

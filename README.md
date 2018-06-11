# Rooftop-Assessment-for-solar-power-using-Satellite-Imagery
Analysed each house rooftop's solar power potential using Google Satellite Images

AI-based technology to assess your Rooftop Solar potential

Individual rooftops of each and every house are identified and segmented out. If you really think that's an easy task, go have a look at the image quality and resolution of rooftop in Google Maps. India doesn't even have a 3D map. Project Sunroof would be very easy in India if we just have 3D map by Google or any map service provider like MapMyIndia, Open Street maps, etc.In US, the Google Map has a clear view at 26/27 zoom level whereas in India you can zoom upto only 22 zoom level. The image quality at 22 zoom level is so bad that you can't even figure outby yourself where the boundaries of each house lies. 

## Aerial Rooftop Detection Methods
### 1) Hough Transform: 
It is used to localize shapes of different types of rooftops. When applied to the image, it gives very less true positives. The main problem was to set threshold parameter of Hough Transform. Windowed Hough Transform: Used to detect exact shapes like squares and rectangles. The main limitation of this method was that it won’t work for other structures if not perfectly
square or a rectangle present in the image.
### 2) Adaptive Canny Edge: 
Applying auto canny on the low-quality image of rooftop results in exact edge detection of rooftops.
Contour Area localization and then applied threshold to detect rooftop. It was also a failure.
### 3) Watershed Segmentation: 
Segmentation on the images from maps to count the number of buildings and to plot rooftop area of each building present in the image. It failed in the case of the densely populated area.

## Gabor Filter
* Gabor Filter analyses whether there is any specific frequency content in the image in specific directions in a localized region around the point or region of analysis.
* Gaussian Mixture Model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of the Gaussian distribution.
* Gabor filter enhances one region relative to other depending on the frequency and theta values. After applying Gabor filter, two Gaussian Mixture models were fit in the histogram of the grayscale image. Two Gaussian Mixture Models separate the image into foreground that is rooftops and background.

## Building Extraction Methods
### 1) Edge Sharpening
Due to the poor quality of the image, to mark the rooftop area edge sharpening of the image is to be done. After that skimage morphological opening is done to fill the gaps in between edges.
### 2) Active Contours
Using the GitHub repository, Active Contour was applied on the rooftop area to extract the optimal area for the solar panel. Active Contours is divided into two, with edges and without edges. Without edges can’t be used in our case as it works on the region segmentation and due to the poor quality of image region, wise segmentation was not possible.

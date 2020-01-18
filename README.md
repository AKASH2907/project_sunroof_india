# Rooftop Assessment for solar power potential using Satellite Images
Analysis of each house rooftop's solar power potential using Google Satellite Images.
![Project Sunroof:India](https://user-images.githubusercontent.com/22872200/47620059-c3bf0880-db0b-11e8-97fa-a353c0c9585c.png)
(This is a screenshot from Google sunroof project)
AI-based technology to assess your Rooftop Solar potential

Individual rooftops of each and every house are identified and segmented out. If you really think that's an easy task, **go have a look at the image quality and resolution of rooftop in Google Maps**. India doesn't even have a 3D map. **Project Sunroof** would be very easy in India if we just have 3D map by Google or any map service provider like MapMyIndia, Open Street maps, etc.In US, the Google Map has a clear view at 22 zoom level whereas in India you can zoom upto only 20/21 zoom level. The image quality at 20 zoom level is so bad that you can't even figure out by yourself where the boundaries of each house lies. Examples of  the dataset is as below on which this algorithms were implemented:

![Rooftops dataset](https://user-images.githubusercontent.com/22872200/47659682-7ac68d00-dbbb-11e8-8952-65bee36efbc0.jpg)

This repository includes:
* Aerial Rooftop Detection Methods:
  * Hough Transforms
  * Watershed Segmentation
  * Adaptive Canny Edge Detection
* Foreground Background Separation (Gabor Filter)
* Building Rooftop Extraction Methods: 
  * Edge Sharpening
  * Active Contours
* Rooftop Polygon Approximation
  * Pixel-based Polygon Filling
  * Region-based Polygon Filling
* Google Maps to Image Pixels
* Optimal Solar Panel Area on Rooftop

## Getting Started

* Install the dependencies from requirements.txt file:
```python
pip install -r requirements.txt
```

* Edge Extraction:
  * [auto_canny.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Edge%20Extraction/auto_canny.py) - Auto-Canny Edge Detection Algorithm on Sharpened Image.
  * [edge Sharpen.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Edge%20Extraction/edge_sharpen.py) - Extarction of edges from an image using Auto Canny Edge detection & Histogram Equalization algorithm.
  * [watershed_pyrMeanShift.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Edge%20Extraction/Watershed_pyrMeanShift.py) -  Watershed Segmentation to segment out rooftop images.

* Gabor Filter:
  * [gabor_test.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Gabor%20Filter/Gabor_test.py) - Gabor Filter to separate foreground rooftops from background of an image.
  
* Active Contours:
  * [active_contours.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Active%20Contours/plot_active_contours.py) - Apply active contour on bilate sharpened image for improved rooftop area extraction. The sharpening helps to localize edges better in the image.
  
* Polygons Approximation:
  * [poly_fill.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Polygon%20Approximation/poly_fill.py) - Approximate the shape of Polygon oh house rooftop
  * [polygon_fill.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Polygon%20Approximation/polygon_fit.py) - Polygon filling pixelwise of rooftop
  
* Google Maps to Image Pixels:
  * [lat_long_metre_pixel.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Google%20Map%20to%20Pixels/lat_long_metre_pixel.py) - Convert the number of pixels into square metre area by calculating the ratio of conversion using latitude and longitude of that area.
  
* Solar Panels Placement:
  * [canny_corner.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Solar%20Panel%20Placement/Canny%20and%20Corners/canny_corners.py) - 1st approach to combine two features i.e. Canny Edge Detection and Harris Corners to localize optimal rooftop area
  * [canny_contours.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Solar%20Panel%20Placement/Corners%20%26%20Contours/canny_contours.py) - 2nd approach to combine two features that are Canny Edge Detection and Contours to localize optimal rooftop area.
  * [panels_atlast.py](https://github.com/AKASH2907/project_sunroof_india/blob/master/Solar%20Panel%20Placement/panels_atlast.py) - Final, Solar Panel placement in Optimal Rooftop Area. You can test with the test images provided.
  
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

![screenshot from 2018-06-12 03-13-14](https://user-images.githubusercontent.com/22872200/41258653-94aeb85e-6dee-11e8-879e-a780f923dc32.png)


## Building Extraction Methods
### 1) Edge Sharpening
Due to the poor quality of the image, to mark the rooftop area edge sharpening of the image is to be done. After that skimage morphological opening is done to fill the gaps in between edges.
### 2) Active Contours
Using the GitHub repository, Active Contour was applied on the rooftop area to extract the optimal area for the solar panel. Active Contours is divided into two, with edges and without edges. Without edges can’t be used in our case as it works on the region segmentation and due to the poor quality of image region, wise segmentation was not possible.

![screenshot from 2018-06-13 02-01-33](https://user-images.githubusercontent.com/22872200/41315710-f0ce448c-6ead-11e8-8930-cebbc835dd02.png)

## Polygons Approximation
### 1)Hough Transform: 
Hough Transform was initially used to analyse the shape of the rooftop. Using K-Means clustering the number of Hough lines were reduced to 4 to 6 to outline the rooftop and obstacle boundaries.
### 2)Pixel wise Polygon filling: 
Applying Contour on the rooftop and moving around the contour in a clockwise direction each pixel and its surroundings was marked as rooftop area.
### 3)Region Based Polygon filling:
After applying Hough Transform in combination with K-Means clustering, the rooftop area was divided into different regions. Checking the intensity of different patches, the area was marked as a rooftop area or not.

![screenshot from 2018-06-14 03-03-23](https://user-images.githubusercontent.com/22872200/41379466-a6d9e750-6f7f-11e8-858e-ddf5d3f43849.png)

## Google Maps to Image Pixels
* The whole region is on top of the pyramid (zoom=0) covered by 256x256 pixels tile,every lower zoom level resolution is always divided by two.
* At every zoom level, there is Meter per pixel value which gives distance in meters according to the difference in pixel values.

## Optimal Rooftop Area for Solar Panels
* Corners and Canny: Where corners and Canny results were overlapping those corners were selected. The problems with corners that they can’t be accessed in a localized manner. To draw a polygon out of that was impossible.

* Canny and Contours: Contours can be accessed in a clockwise manner. On two images, Canny was applied. One is edge sharpened image and the other is canny edge map image. Contours in Edge sharpened the image using threshold gives rooftop boundaries. Contours on canny edge map using threshold gives obstacle boundaries on the rooftop. Combining both the results and plotting it on a white patch gives the exact rooftop optimal area for solar panel placement.

![solar_panels](https://user-images.githubusercontent.com/22872200/41616111-bdc51256-741a-11e8-83e4-0c8253d6429a.png)


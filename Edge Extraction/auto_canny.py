# import the necessary packages
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt


def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "-F:\CV Coding\rooftops", required=True,
# help="F:\CV Coding\rooftops")
# args = vars(ap.parse_args())F:\CV Coding\rooftops
images = glob.glob("*.jpg")

# loop over the images
for imagePath in images:
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print (gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    fg = cv2.addWeighted(blurred, 1.5, gray, -0.5, 0)
    kernel_sharp = np.array((
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]), dtype='int')
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = laplacian.clip(min=0)
    print (laplacian)
    auto = auto_canny(fg)
    auto1 = auto_canny(blurred)
    im = cv2.filter2D(auto, -1, kernel_sharp)
    # dst = cv2.addWeighted(gray, 0.5, auto, 0.5, 0)
    # dst1 = cv2.addWeighted(gray, 0.5, auto1, 0.5, 0)
    x = laplacian.astype(np.uint8)
    print (x)
    auto2 = auto_canny(x)
    im1 = cv2.filter2D(auto2, -1, kernel=kernel_sharp)

    plt.figure()
    plt.title("fg")
    plt.imshow(auto, cmap='gray')
    plt.figure()
    plt.title("Blur")
    plt.imshow(auto1, cmap='gray')
    plt.figure()
    plt.title("laplace")
    plt.imshow(laplacian, cmap='gray')
    plt.figure()
    plt.title("lapalace1")
    plt.imshow(x, cmap='gray')
    plt.figure()
    plt.title("edge laplace")
    plt.imshow(im1, cmap='gray')
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()

plt.figure("Original")
plt.close()
plt.figure("Nothing")
plt.close()
plt.figure("Blur/Smooth")
plt.close()

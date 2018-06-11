from __future__ import print_function
from skimage.filters import gabor
from sklearn.mixture import GaussianMixture
from PIL import Image
from pylab import *
from numpy import *


def gabor_fit_func(img, frequency, theta):
    filter_img = gabor_filter(img, frequency, theta)
    hist1, hist2 = convert(filter_img)
    mean1, mean2, std1, std2 = gaussian_curve(hist2)
    return (gabor_cost_func(mean1, mean2, std1, std2))


def gabor_cost_func(m1, m2, v1, v2):
    J = (m2 - m1) / (v1 + v2)
    return J


def gabor_filter(gray, frequency, theta):
    mask = 10
    sigma = mask / 2
    filt_real, filt_imag = gabor(gray, frequency, theta=theta * np.pi, sigma_x=sigma,
                                 sigma_y=sigma, n_stds=mask)
    return filt_real


def convert(filt_image):
    # To plot the histogram, 2D array filt_real is flattened in 1D array
    hist = filt_image.flatten()
    # Converting the horizontal stacked array into vertical stack to analyze
    # the Gaussian Mixture models
    hist2 = np.vstack(hist)
    return hist, hist2


def gaussian_curve(hist2):
    nmodes = 2
    GMModel = GaussianMixture(n_components=nmodes, covariance_type='full')
    GMModel.fit(hist2)
    # mu1,mu2- Two mean values relative two Gaussian Models in the curve
    mu1, mu2 = np.round(GMModel.means_, 2)
    v1, v2 = np.round(GMModel.covariances_, 2)
    v11, v22 = (np.sqrt(v1), np.sqrt(v2))
    return mu1, mu2, v11, v22


if __name__ == '__main__':
    img = array(Image.open('example3.png').convert('L'))
    frequency = input('Enter frequency: ')
    theta = input('Enter Theta: ')
    gabor_fit_func(img, frequency, theta)

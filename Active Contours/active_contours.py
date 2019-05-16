import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from cv2 import imread
import scipy

# Bilate filtered image. Sharpened image helps to give a better edge approximation.
img = imread('bilate.png')
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 1000)
x = 83 + 100*np.cos(s)
y = 33 + 100*np.sin(s)
init = np.array([x, y]).T
print (init)
'''
alpha Higher values make snake contract faster
beta Highervalue make snake smoother
Gamma Timestepping parameter
'''
snake = active_contour(img, init, alpha=-1, beta=7, gamma=0.001, max_iterations=500)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
# ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
# plt.imshow(img, cmap='gray')
plt.show()

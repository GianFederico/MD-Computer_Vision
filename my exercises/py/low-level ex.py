import cv2
import numpy as np
import matplotlib.pyplot as plt

#equalization histogram for contrast enhancement + plots

img= cv2.imread('assets/Q2_1_2.tiff', -1)
new_img = cv2.equalizeHist(img)

images = [img, new_img]
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
for ind, p in enumerate(images):
    ax = axs[ind]
    ax.imshow(p)
    ax.axis('off')
axs[0].set_title('Original', fontsize=16)
axs[1].set_title('Equalized', fontsize=16)
plt.show()

 
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum() 
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'blue')
plt.title('Original histogram')
plt.hist(new_img.flatten(),256,[0,256], color = 'gray')
plt.xlim([0,256])
plt.legend(('cumulative distr func','histogram'), loc = 'upper left')
plt.show()

hist,bins = np.histogram(new_img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'blue')
plt.title('Equalized histogram OK')
plt.hist(new_img.flatten(),256,[0,256], color = 'gray')
plt.xlim([0,256])
plt.legend(('cumulative distr func','histogram'), loc = 'upper left')
plt.show()
#_____________________________________________________



#blurring and smoothing in order to remove noise

noised_lena= cv2.imread('assets/lena.png', -1)

img_0 = cv2.blur(noised_lena, ksize = (7, 7))
img_1 = cv2.GaussianBlur(noised_lena, ksize = (7, 7), sigmaX = 0)   
img_2 = cv2.medianBlur(noised_lena, 7)
img_4 = cv2.bilateralFilter(noised_lena, 7, sigmaSpace = 80, sigmaColor =80)

# Plot the images
images = [noised_lena, img_0, img_1, img_2, img_4]
fig, axs = plt.subplots(nrows = 1, ncols = 5, figsize = (10, 5))
for ind, p in enumerate(images):
    ax = axs[ind]
    ax.imshow(p)
    ax.axis('off')
axs[0].set_title('Original', fontsize=16)
axs[1].set_title('Average blur', fontsize=16)
axs[2].set_title('Gaussian blur', fontsize=16)
axs[3].set_title('Median blur', fontsize=16)
axs[4].set_title('Bilateral filter', fontsize=16)
plt.show()
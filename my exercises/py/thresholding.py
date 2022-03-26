import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('assets/skin.jpeg') #loading image
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting in grayscale

cv2.imshow('Image converted in grayscale',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blurred_img = cv2.GaussianBlur(gray_img, ksize = (7, 7), sigmaX = 0)    #blurring with gaussian filter

cv2.imshow('Blurred image (gaussian filter)',blurred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

hist,bins = np.histogram(img.flatten(),256,[0,256]) #creating and showing histogram's plot
cdf = hist.cumsum() 
plt.title('grayscale blurred image histogram')
plt.hist(blurred_img.flatten(),256,[0,256], color = 'gray')
plt.xlim([0,256])
plt.legend(('cumulative distribution functions','histogram'), loc = 'upper left')
plt.show()

ret,otzu_thresh = cv2.threshold(blurred_img,180,255,cv2.THRESH_OTSU)    #using otsu's method for thresholding

images = [blurred_img, otzu_thresh] #plotting otsu's results
plt.title('OTZU thresholding')
plt.imshow(otzu_thresh, cmap = 'gray')
plt.show()


bin_img=255-otzu_thresh #creating the binary mask for the bitwise_or operator
cv2.imshow('Binary image',bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

last=cv2.bitwise_or(img, img, mask=bin_img) #applying bitwise_or operator and showing its results
cv2.imshow('Final image',last)
cv2.waitKey(0)
cv2.destroyAllWindows()
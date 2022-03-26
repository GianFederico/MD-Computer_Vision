import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('assets/building.jpeg') #loading image
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting in grayscale
blurred_img = cv2.GaussianBlur(gray_img, ksize = (7, 7), sigmaX = 0) 
blurred_img= cv2.resize(blurred_img, (400,600))

# Sobel Edge Detection
sobelx = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=blurred_img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Display Prewitt Edge Detection 
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(blurred_img, cv2.CV_64F, kernelx)
prewitty = cv2.filter2D(blurred_img, cv2.CV_64F, kernely)
prewittxy = prewitty+prewittx
cv2.imshow('prewitt X Y using manual function', prewittxy)
cv2.waitKey(0)

# Dispaly gradient for both images
sobel_grad=cv2.magnitude(sobelx, sobely)    #slightly smoother on the approximation of edges
prewitt_grad=cv2.magnitude(prewittx, prewitty)  #slightly sharper

imgs = [sobel_grad, prewitt_grad]
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 5))
for ind, p in enumerate(imgs):
    ax = axs[ind]
    ax.imshow(p)
    ax.axis('off')
axs[0].set_title('Sobel gradient', fontsize=16)
axs[1].set_title('Prewitt gradient', fontsize=16)
plt.show()

# thresholding gradient imgs 
ret,otzu_thresh_sobel = cv2.threshold(sobelxy,100,255,cv2.THRESH_BINARY)
ret,otzu_thresh_prewitt = cv2.threshold(prewitt_grad,100,255,cv2.THRESH_BINARY) 

canny_img=cv2.Canny(blurred_img, 150, 100) 
cv2.imshow('canny', canny_img)
cv2.waitKey(0)


images = [img, otzu_thresh_sobel, otzu_thresh_prewitt, canny_img]
fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (10, 5))
for ind, p in enumerate(images):
    ax = axs[ind]
    ax.imshow(p)
    ax.axis('off')
axs[0].set_title('Original', fontsize=16)
axs[1].set_title('Sobel', fontsize=16)
axs[2].set_title('Prewitt', fontsize=16)
axs[3].set_title('Canny Edge', fontsize=16)
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('assets/Q2_1_1.tiff', -1)
cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()




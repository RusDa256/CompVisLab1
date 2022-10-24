import cv2 as cv
import numpy as np
from skimage import measure


img = cv.imread("whiteballssample.jpg")
new_w = int(img.shape[1]/2)
new_h = int(img.shape[0]/2)
dsize = (new_w, new_h)

img1 = cv.resize(img, dsize)

img = cv.resize(img1, dsize)

img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
print("THRESHOLD = ",  ret)

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 3)
dilation = cv.dilate(erosion,kernel,iterations = 1)

balls = measure.label(erosion)
print("Количесво шаров = ", balls.max())

contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img1, contours, -1, (0,255,0), 3)


rad = []
for i in contours:
    rad.append(cv.minEnclosingCircle(i)[1])

print("count: ", len(rad), "\nmean: ",np.mean(rad), "\nvariance: ",np.var(rad))


cv.imshow('Balls', img1)
cv.waitKey()
cv.destroyAllWindows()
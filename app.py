import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image

img = cv2.imread('hand-ball1.jpg')

# Laterally invert the image / flip the image
img  = cv2.flip(img, 1);

# converting from BGR to HSV color space
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# Range for lower orange
lower_red = np.array([2,150,150])
upper_red = np.array([7,255,255])
maskForBall1 = cv2.inRange(hsv, lower_red, upper_red)
 
# Range for upper range
lower_red = np.array([170,150,150])
upper_red = np.array([180,255,255])
maskForBall2 = cv2.inRange(hsv,lower_red,upper_red)
 
# Generating the final mask to detect orange
maskForBall1 = maskForBall1+maskForBall2

# Range for hand color
lower_red = np.array([10,50,150])
upper_red = np.array([50,200,255])
maskForHand1 = cv2.inRange(hsv, lower_red, upper_red)

maskForBall1 = cv2.morphologyEx(maskForBall1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
maskForBall1 = cv2.morphologyEx(maskForBall1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

maskForHand1 = cv2.morphologyEx(maskForHand1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
maskForHand1 = cv2.morphologyEx(maskForHand1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
# creating an inverted mask to segment out the ball from the rest of the frame
maskForBall2 = cv2.bitwise_not(maskForBall1)
maskForHand1 = cv2.bitwise_and(maskForHand1, maskForBall2)

dilated1 = cv2.dilate(maskForBall1, np.ones((3,3), np.uint8), iterations=15)
eroded1 = cv2.erode(dilated1, np.ones((3,3), np.uint8), iterations=3)
dilated2 = cv2.dilate(maskForHand1, np.ones((3,3), np.uint8), iterations=7)
eroded2 = cv2.erode(dilated2, np.ones((3,3), np.uint8), iterations=3)
contours1, h = cv2.findContours(eroded1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours2, h = cv2.findContours(eroded2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

(x,y),radius = cv2.minEnclosingCircle(contours1[0])
center = (int(x),int(y))
cv2.circle(eroded1,center,int(radius),(255,0,0), -1)

cv2.fillPoly(eroded2, color = (255, 0, 0) , pts = contours2 )

eroded1 = cv2.bitwise_and(eroded1,eroded1,mask=eroded2)
eroded1 = np.array(eroded1)

pixelPctg = eroded1[np.where(eroded1 >= 1)].size / eroded1.size * 100
print(pixelPctg)

if pixelPctg > 0.1: 
    print("rankoje")
else: 
    print("ne rankose")

cv2.waitKey(0)
cv2.destroyAllWindows()
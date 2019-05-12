import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image

class Algorithm:

	def __init__(this):
		this.stepCount = 0
		this.feetIntersection = False
		this.ballInHands = False
		this.turnover = False

	def execute(this, img):
		txt1 = ""
		txt2 = ""

		# Laterally invert the image / flip the image
		img  = cv2.flip(img, 1);

		# converting from BGR to HSV color space
		hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
		# Range for ball color
		lower_green = np.array([77, 60, 50])
		upper_green = np.array([83, 255, 200])
		maskForBall = cv2.inRange(hsv, lower_green, upper_green)

		# Range for hand color
		lower_yellow = np.array([19,130,130])
		upper_yellow = np.array([25,255,255])
		maskForHand = cv2.inRange(hsv, lower_yellow, upper_yellow)

		# Range for shoes color
		lower_red = np.array([173,100,100])
		upper_red = np.array([180,250,255])
		maskForShoes = cv2.inRange(hsv, lower_red, upper_red)

		maskForBall = morphDilate(morphOpen(maskForBall))
		maskForHand = morphDilate(morphOpen(maskForHand))
		maskForShoes =  morphDilate(morphOpen(maskForShoes))
		 
		# creating an inverted mask to segment out the ball from the rest of the frame
		maskForNotBall = cv2.bitwise_not(maskForBall)
		maskForHand = cv2.bitwise_and(maskForHand, maskForNotBall)

		maskForBall = erode(dilate(maskForBall, 15), 3)
		maskForHand = dilate(erode(maskForHand, 3), 7)
		maskForShoes = dilate(erode(maskForShoes, 3), 15)

		ballContours, h = cv2.findContours(maskForBall, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		handContours, h = cv2.findContours(maskForHand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		shoeContours, h = cv2.findContours(maskForShoes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(shoeContours) == 1 and not this.feetIntersection and this.ballInHands: # paprastai shoe conturu buna du, jeigu vienas assuminam kad susikryziavo
			this.feetIntersection = True
			this.stepCount = this.stepCount + 1
			if this.stepCount > 2:
				this.turnover = True
		elif len(shoeContours) > 1 and this.feetIntersection:
			this.feetIntersection = False
			if not this.ballInHands:
				this.stepCount = 0
			
		(handX,handY,handW,handH) = cv2.boundingRect(handContours[0])
		ballAboveHands = False;

		# nekazka
		if (len(ballContours) > 0):
			(ballX,ballY,ballW,ballH) = cv2.boundingRect(ballContours[0])
			handCenter = ((handX + handW)/2, (handY + handH)/2)
			ballCenter = ((ballX + ballW)/2, (ballY + ballH)/2)
			if ballCenter[1] > handCenter[1]:
				ballAboveHands = True;
		#nekazka

		ballContour = chooseLargestContours(ballContours)
		if ballContour is not None:
			(x,y),radius = cv2.minEnclosingCircle(ballContour)
			center = (int(x),int(y))
			cv2.circle(maskForBall,center,int(radius),(255,0,0), -1)

			# fill hand
			cv2.fillPoly(maskForHand, color = (255, 0, 0), pts = handContours )

			handAndBall = cv2.bitwise_and(maskForBall,maskForBall,mask=maskForHand)
			handAndBall = np.array(handAndBall)

			pixelPctg = handAndBall[np.where(handAndBall >= 1)].size / handAndBall.size * 100
			print(pixelPctg)
			txt2 = "Zinsgniai: " + str(this.stepCount)

			if ballAboveHands:
			    txt = "Kamuolys virs ranku"
			else:
				if pixelPctg > 0.001: 
				    this.ballInHands = True
				    txt = "rankose, " + str(pixelPctg) 
				else: 
				    this.ballInHands = False # cia galima butu fix kad tarkim 5 step counter, else --, False if reach 0, if true = padarai vel 5
					# kad nesakytu kad nebe rankose tik todel kad durnas algoritmas nerado
				    txt = "ne rankose, " + str(pixelPctg) 
		else:
			 txt = "kamuolys nerastas"

		font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
		x = 1000 #position of text
		y = 200 #position of text
		cv2.putText(img, txt, (x,y),font, 2, (0,255,0), 3) #Draw the text

		x1 = 1000 #position of text
		y1 = 400 #position of text
		cv2.putText(img, txt1, (x1,y1),font, 2, (0,255,0), 3) #Draw the text

		x1 = 1000 #position of text
		y1 = 600 #position of text
		cv2.putText(img, txt2, (x1,y1),font, 2, (0,255,0), 3) #Draw the text

		if this.turnover == True:
			x1 = 1000 #position of text
			y1 = 800 #position of text
			cv2.putText(img, "TRAVEL! TURNOVER COMMITED!" , (x1,y1),font, 2, (0,0,255), 3) #Draw the text

		return img;

def morphOpen(image):
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

def morphDilate(image):
	return cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

def erode(image, iterations):
	return cv2.erode(image, np.ones((3,3), np.uint8), iterations=iterations)

def dilate(image, iterations):
	return cv2.dilate(image, np.ones((3,3), np.uint8), iterations=iterations)

def chooseLargestContours(contours):
	contour = [contour for contour in contours if contour.size == max([contour.size for contour in contours])]
	if len(contour) > 0:
		return contour[0]
	else:
		return None

def printImg(image):
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600,600)
	cv2.imshow("image", image)

def correct_rotation(frame, rotateCode):  
    	return cv2.rotate(frame, rotateCode) 

	
vid = cv2.VideoCapture('kursiniuivid(0).mp4')
algorithm = Algorithm()

while(True):
    # Capture frame-by-frame
	ret, frame = vid.read()
	frame = algorithm.execute(frame)

    # Display the resulting frame
	printImg(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
from ImageProcessingUtils import *
import cv2
import numpy as np

lower_green = np.array([74, 80, 42])
upper_green = np.array([84, 255, 200])
lower_yellow = np.array([17,140,130])
upper_yellow = np.array([28,255,255])
lower_red = np.array([173,100,100])
upper_red = np.array([180,250,255])

class Algorithm:

	def __init__(this):
		this.step_count = 0
		this.feet_intersection = False
		this.ball_in_hands_counter = 0
		this.turnover = False

	def ball_in_hands(this):
		return this.ball_in_hands_counter > 0

	def compute_step(this, shoe_contours):
		if this.ball_in_hands() and len(shoe_contours) == 1 and not this.feet_intersection:
			this.feet_intersection = True
			this.step_count = this.step_count + 1

		elif len(shoe_contours) > 1:
			this.feet_intersection = False
			if this.ball_in_hands_counter == 0:
				this.step_count = 0

	def compute_turnover(this):
		if this.step_count > 2:
			this.turnover = True

	def execute(this, img):
		txt1 = ""
		txt2 = ""
		txt = ""

		img  = flip_img(img)
		hsv = convert_to_hsv(img)

		mask_for_ball = segment_by_color(hsv, lower_green, upper_green)
		mask_for_hand = segment_by_color(hsv, lower_yellow, upper_yellow)
		mask_for_shoes = segment_by_color(hsv, lower_red, upper_red)

		mask_for_ball = morph_dilate(morph_open(mask_for_ball))
		mask_for_hand = morph_dilate(morph_open(mask_for_hand))
		mask_for_shoes =  morph_dilate(morph_open(mask_for_shoes))
		 
		# creating an inverted mask to segment out the ball from the rest of the frame
		maskForNotBall = cv2.bitwise_not(mask_for_ball)
		mask_for_hand = cv2.bitwise_and(mask_for_hand, maskForNotBall)

		mask_for_ball = erode(dilate(mask_for_ball, 15), 3)
		mask_for_hand = dilate(erode(mask_for_hand, 3), 7)
		mask_for_shoes = dilate(erode(mask_for_shoes, 3), 15)

		ball_contours, h = find_contours(mask_for_ball)
		hand_contours, h = find_contours(mask_for_hand)
		shoe_contours, h = find_contours(mask_for_shoes)

		this.compute_step(shoe_contours)
		this.compute_turnover()
		
		if len(hand_contours) < 1:
			return img;	

		(handX,handY,handW,handH) = cv2.boundingRect(hand_contours[0])
		ball_above_hands = False;

		#improve later - ranka virs kamuolio
		# if (len(ball_contours) > 0):
		#	(ballX,ballY,ballW,ballH) = cv2.boundingRect(ball_contours[0])
		#	handCenter = ((handX + handW)/2, (handY + handH)/2)
		#	ballCenter = ((ballX + ballW)/2, (ballY + ballH)/2)
		#	if ballCenter[1] > handCenter[1]:
		#		ball_above_hands = True;
		#improve later

		ball_contour = choose_largest_contours(ball_contours)
		if ball_contour is not None:
			(x,y),radius = cv2.minEnclosingCircle(ball_contour)
			center = (int(x),int(y))
			cv2.circle(mask_for_ball,center,int(radius),(255,0,0), -1)
			bc, h = find_contours(mask_for_ball)
			cv2.drawContours(img, bc, -1, (0,255,0), 3)


			# fill hand
			cv2.fillPoly(mask_for_hand, color = (255, 0, 0), pts = hand_contours )

			hand_and_ball = cv2.bitwise_and(mask_for_ball,mask_for_ball,mask=mask_for_hand)
			hand_and_ball = np.array(hand_and_ball)

			pixel_pctg = hand_and_ball[np.where(hand_and_ball >= 1)].size / hand_and_ball.size * 100
			txt2 = "Zinsgniai: " + str(this.step_count)

			if pixel_pctg > 0.001: 
			    txt = "rankose, " + str(pixel_pctg) 
			    this.ball_in_hands_counter = 3
			else: 
			    this.ball_in_hands_counter = max(this.ball_in_hands_counter - 1, 0)
			    if this.ball_in_hands_counter == 0: txt = "ne rankose, " + str(pixel_pctg) 

		else:
			 txt = "kamuolys nerastas"

		create_background(img, (1200, 150), (2000, 400))
		put_text(img, txt, (1200, 200))
		put_text(img, txt1, (1200, 250))
		put_text(img, txt2, (1200, 300))
		if this.turnover == True:
			put_text(img, "TURNOVER! TRAVEL", (1000, 800), (0,0,255))

		return img;
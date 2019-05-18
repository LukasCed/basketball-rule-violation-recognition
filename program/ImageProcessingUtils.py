import cv2
import numpy as np

def morph_open(image):
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

def morph_dilate(image):
	return cv2.morphologyEx(image, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

def erode(image, iterations):
	return cv2.erode(image, np.ones((3,3), np.uint8), iterations=iterations)

def dilate(image, iterations):
	return cv2.dilate(image, np.ones((3,3), np.uint8), iterations=iterations)

def segment_by_color(image, lowerRange, upperRange):
	return cv2.inRange(image, lowerRange, upperRange)

def find_contours(image):
	return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def choose_largest_contours(contours):
	contour = [contour for contour in contours if contour.size == max([contour.size for contour in contours])]
	if len(contour) > 0:
		return contour[0]
	else:
		return None

def flip_img(image):
	# Laterally invert the image / flip the image
	return cv2.flip(image, 1);

def convert_to_hsv(image):
	# Laterally invert the image / flip the image
	return cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	
def print_img(image):
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600,600)
	cv2.imshow("image", image)

def correct_rotation(image, rotateCode):  
    	return cv2.rotate(image, rotateCode) 

def put_text(image, text, coord, color = (0,255,0)):  
	font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
	x = coord[0]
	y = coord[1]
	cv2.putText(image, text, (x,y), font, 2, color, 3) #Draw the text

def create_background(image, coordStart, coordEnd):
	rectangle_bgr = (0, 0, 0)
	# make the coords of the box with a small padding of two pixels
	box_coords = ((coordStart[0], coordStart[1]), (coordEnd[0], coordEnd[1]))
	cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)


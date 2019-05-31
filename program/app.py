import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image
from Algorithm import Algorithm
from ImageProcessingUtils import print_img
	

vid = cv2.VideoCapture('vidai/10T.mp4')
algorithm = Algorithm()

while(not algorithm.turnover):
    # Capture frame-by-frame
	ret, frame = vid.read()
	frame = algorithm.execute(frame)

    # Display the resulting frame
	print_img(frame)
	key = cv2.waitKey(1) & 0xff
	if key == ord('p'):

		while True:

			key2 = cv2.waitKey(1) or 0xff
			cv2.imshow('frame', frame)

			if key2 == ord('p'):
				break

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
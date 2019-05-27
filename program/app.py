import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image
from Algorithm import Algorithm
from ImageProcessingUtils import print_img
	

vid = cv2.VideoCapture('vidai/normal.mp4')
algorithm = Algorithm()

while(!algorithm.turnover):
    # Capture frame-by-frame
	ret, frame = vid.read()
	frame = algorithm.execute(frame)

    # Display the resulting frame
	print_img(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()
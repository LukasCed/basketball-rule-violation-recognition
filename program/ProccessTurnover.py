import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image
from Algorithm import Algorithm
from ImageProcessingUtils import print_img

def proccess_turnover(video_path):
	vid = cv2.VideoCapture(video_path)
	algorithm = Algorithm()
	ret, frame = vid.read()

	while(not algorithm.turnover and not frame is None):
	    # Capture frame-by-frame
		frame = algorithm.execute(frame)	
	    # Display the resulting frame
		# print_img(frame)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		#	break	
		ret, frame = vid.read()



	# When everything done, release the capture
	vid.release()
	cv2.destroyAllWindows()
	print("video", video_path, "is done processing")
	return algorithm.turnover
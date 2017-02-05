########################################################################
#
# File:   project1_cv.py
# Author: Tom Wilmots and John Larkin
# Date:   January 27th, 2017
#
# Written for ENGR 27 - Computer Vision
# Note: 
# 	The point of this project was to threshold a sequence of images to distinguish objects of interest in the background. In addition, we apply morphological operators in order to remove noise and imperfections. 
#	Perform connected components analysis to distinguish between separate objects, and identify their locations in the image. 
# 	Track the connected components over time. 
#	The last two lines are commonly known as "blob tracking"
########################################################################

'''
Thresholding
------------
Averaging - spatial and temporal averaging. 
	Spatial - also known as adaptive thresholding. Looks at the difference between a given pixel and the average of the pixels around it. 
			  	cv2 has this with the adaptiveThreshold function.
	Temporal - looks at diff between given pixel at some location and the average intensity over time
				You can take temp avg by adding together multiple frames and dividing by the number of frames. 
				Watch out for overflow. 
				Convert the frames from 8bit int format to more precision before summing together 
				You can use the OpenCv threshold function
RGB thresholding - simplest way to execture:
		1. Convert RGB to grayscale 
		2. Threshold accordingly 
		Other methods:
			planar decision boundary 
			distance from a reference RGB value
			Find some OpenCV or NumPy functionality to perform the equivalent operations more quickly
Intermediate image storage - If you are averaging together multiple images, first convert to storage format that will allow these operations.
		Movie frames will be provided in numpy.uint8 format. But we should use numpy.float32 for intermediate range

Morphological Operators
-----------------------
These help to remove noise and speckles. 
OpenCV provides 
erode - implement erosion
dilate - implement dilation
morphologyEx - implement opening / closing 
Goal:
	Produce the best image possible to send to the next part of the pipeline

Connected Components Analysis
-----------------------------
cv2.findContours - retrieves the outlines of connected components of non-zero pixels
This corresponds to outlines of objects of interest in your scene 
Additional analysis of contours yields info such as area, centroid, principal axes
(see regions.py for more)

Tracking
--------
System should extract position of each object's centroid in each frame. It should be able 
to track objects' trajectories over time by associating the connected comp in current frame with previous frame
'''

import cv2 
import numpy as np
import sys
import cvk2

def load_in_image_or_video():
	# This is going to query the user for an input 
	while True:
		video_or_image = int(raw_input("Is your file a video or image? Enter 0 for image or 1 for video: "))
		if video_or_image == 0:
			# we have an image
			image_name = raw_input("Enter the filename of the image to manipulate (or 0 for default): ")
			if image_name == '0':
				original = cv2.imread('Images/default.png')
				break
			else:
				image_name = 'Images/' + image_name
				original = cv2.imread(image_name)
				if original is None:
					print("There was an error. Please try again.")
				else: 
					break #we're good
		elif video_or_image == 1:
			video_name = raw_input("Enter the filename of the video to manipulate (or 0 for default): ")
			if video_name == '0':
				original = cv2.VideoCapture('Videos/default.avi')
				break
			else:
				video_name = 'Videos/' + video_name	
				original = cv2.VideoCapture(video_name)
				if not original or not original.isOpened():
					print("There was an error. Please try again.")
				else:
					break #we're
	return (original, video_or_image)

def display_initial_input(orig, flag):
	if flag == 0:
		# we have an image
		cv2.imshow('Image', orig)
		cv2.waitKey()
	else:
		# Fetch the first frame and bail if none.
		ok, frame = orig.read()

		if not ok or frame is None:
		    print('No frames in video')
		    sys.exit(1)

		# Now set up a VideoWriter to output video.
		w = frame.shape[1]
		h = frame.shape[0]

		fps = 30

		# One of these combinations should hopefully work on your platform:
		fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
		#fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')

		filename = 'captured.'+ext

		writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
		if not writer:
		    print('Error opening writer')
		else:
		    print('Opened', filename, 'for output.')
		    writer.write(frame)

		# Loop until movie is ended or user hits ESC:
		while 1:

		    # Get the frame.
		    ok, frame = orig.read(frame)

		    # Bail if none.
		    if not ok or frame is None:
		        break

		    # Write if we have a writer.
		    if writer:
		        writer.write(frame)

		    # Throw it up on the screen.
		    cv2.imshow('Video', frame)

		    # Delay for 5ms and get a key
		    k = cv2.waitKey(5)

		    # Check for ESC hit:
		    if k % 0x100 == 27:
		        break



if __name__ == "__main__":
	(original, video_or_image) = load_in_image_or_video()
	display_initial_input(original, video_or_image)



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

def fixKeyCode(code):
    return np.uint8(code).view(np.int8)

def labelAndWaitForKey(image, text):

    # Get the image height - the first element of its shape tuple.
    h = image.shape[0]

    display = image.copy()


    text_pos = (16, h-16)                # (x, y) location of text
    font_face = cv2.FONT_HERSHEY_SIMPLEX # identifier of font to use
    font_size = 1.0                      # scale factor for text
    
    bg_color = (0, 0, 0)       # RGB color for black
    bg_size = 3                # background is bigger than foreground
    
    fg_color = (255, 255, 255) # RGB color for white
    fg_size = 1                # foreground is smaller than background

    line_style = cv2.LINE_AA   # make nice anti-aliased text

    # Draw background text (outline)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                bg_color, bg_size, line_style)

    # Draw foreground text (middle)
    cv2.putText(display, text, text_pos,
                font_face, font_size,
                fg_color, fg_size, line_style)

    cv2.imshow('Image', display)

    # We could just call cv2.waitKey() instead of using a while loop
    # here, however, on some platforms, cv2.waitKey() doesn't let
    # Ctrl+C interrupt programs. This is a workaround.
    while fixKeyCode(cv2.waitKey(15)) < 0: pass

def load_in_image_or_video():
	# This is going to query the user for an input 
	while True:
		video_or_image = int(raw_input("Is your file a video or image? Enter 0 for image or 1 for video: "))
		if video_or_image == 0:
			# we have an image
			image_name = raw_input("Enter the filename of the image to manipulate (or 0 for default): ")
			if image_name == '0':
				name = 'Images/default.png'
				original = cv2.imread(name)
				break
			else:
				name = 'Images/' + image_name
				original = cv2.imread(name)
				if original is None:
					print("There was an error. Please try again.")
				else: 
					break #we're good
		elif video_or_image == 1:
			video_name = raw_input("Enter the filename of the video to manipulate (or 0 for default): ")
			if video_name == '0':
				name = 'Videos/default.avi'
				original = cv2.VideoCapture(name)
				break
			else:
				name = 'Videos/' + video_name	
				original = cv2.VideoCapture(name)
				if not original or not original.isOpened():
					print("There was an error. Please try again.")
				else:
					break #we're
	return (original, video_or_image, name)

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

		labelAndWaitForKey(frame, 'First Frame')
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

def temporal_averaging_movie(orig, name):
	orig = cv2.VideoCapture(name)

	ok, frame = orig.read()

	# Now set up a VideoWriter to output video.
	w, h = frame.shape[1], frame.shape[0]

	labelAndWaitForKey(frame, 'First Frame')
	fps = 30

	# Loop until movie is ended or user hits ESC:
	# Let's get our average of the movie

	sum_of_image = np.zeros((w,h,3),dtype='float32')
	count = 0

	#while 1:
	for i in range(100):
		count += 1
		# Get the frame.
		ok, frame = orig.read(frame)

		sum_of_image += frame.astype('float32')
		
		# Bail if none.
		if not ok or frame is None:
			break

		# Throw it up on the screen. Let's not show it 
		#cv2.imshow('Video', frame)

		# Delay for 5ms and get a key
		k = cv2.waitKey(5)

		# Check for ESC hit:
		if k % 0x100 == 27:
			break

	normalized_scene = sum_of_image / count
	return normalized_scene.astype('uint8')

def show_movie_with_thresh(back, orig, name):
	mov = cv2.VideoCapture(name)
	ok, frame = mov.read()
	if not ok or frame is None:
	    print('No frames in video')
	    sys.exit(1)

	# Now set up a VideoWriter to output video.
	w = frame.shape[1]
	h = frame.shape[0]

	labelAndWaitForKey(frame, 'First Frame')
	fps = 30

	# One of these combinations should hopefully work on your platform:
	fourcc, ext = (cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), 'avi')
	#fourcc, ext = (cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 'mov')

	filename = 'threshed.'+ext

	writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))

	# Loop until movie is ended or user hits ESC:
	while 1:

	    # Get the frame.
	    ok, frame = mov.read(frame)

	    # Bail if none.
	    if not ok or frame is None:
	        break

	    t_frame = frame.astype('float32') - back.astype('float32')
	    t_frame = t_frame.astype('uint8')
	    to_show = cv2.convertScaleAbs(t_frame)

	    # Write if we have a writer.
	    if writer:
	        writer.write(to_show)

	    # Throw it up on the screen.
	    cv2.imshow('Video', to_show)

	    # Delay for 5ms and get a key
	    k = cv2.waitKey(5)

	    # Check for ESC hit:
	    if k % 0x100 == 27:
	        break

if __name__ == "__main__":
	(original, video_or_image, name) = load_in_image_or_video()
	ans = raw_input("Would you like to display the image/video? (yes/no): ")
	if ans.lower() == "yes":
		display_initial_input(original, video_or_image)
	if video_or_image == 1:
		# we have a video, let's get the average for a few frames
		average_scene = temporal_averaging_movie(original, name)
		labelAndWaitForKey(average_scene, "100 Frame Average")
		show_movie_with_thresh(average_scene, original, name)





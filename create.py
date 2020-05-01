import cv2
import numpy as np
import pickle, os
def nothing(x):
	pass

cam = cv2.VideoCapture(0)

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def store_images(g_id):
	total_pics = 150
	pic_no = 0
	cv2.namedWindow("Trackbars")
	cv2.createTrackbar('L - H', 'Trackbars', 0, 179, nothing)
	cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
	cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
	cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
	cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
	cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

	cv2.setTrackbarPos("L - H", "Trackbars", 0)
	cv2.setTrackbarPos('L - S', 'Trackbars', 40)
	cv2.setTrackbarPos('L - V', 'Trackbars', 46)

	create_folder("gestures/"+str(g_id))
	while True:
		ret, frame = cam.read()
		frame = cv2.flip(frame,1)
		l_h = cv2.getTrackbarPos("L - H", "Trackbars")
		l_s = cv2.getTrackbarPos("L - S", "Trackbars")
		l_v = cv2.getTrackbarPos("L - V", "Trackbars")
		u_h = cv2.getTrackbarPos("U - H", "Trackbars")
		u_s = cv2.getTrackbarPos("U - S", "Trackbars")
		u_v = cv2.getTrackbarPos("U - V", "Trackbars")
		img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
		lower_blue = np.array([l_h, l_s, l_v])
		upper_blue = np.array([u_h, u_s, u_v])
		imcrop = img[100:300, 425:625]
		hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		# cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
		cv2.imshow("Trackbars", frame)
		cv2.imshow("mask", mask)
		if cv2.waitKey(1) & 0xFF == ord('s'):
			while True:
				ret, frame = cam.read()
				frame = cv2.flip(frame,1)
				l_h = cv2.getTrackbarPos("L - H", "Trackbars")
				l_s = cv2.getTrackbarPos("L - S", "Trackbars")
				l_v = cv2.getTrackbarPos("L - V", "Trackbars")
				u_h = cv2.getTrackbarPos("U - H", "Trackbars")
				u_s = cv2.getTrackbarPos("U - S", "Trackbars")
				u_v = cv2.getTrackbarPos("U - V", "Trackbars")
				img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
				lower_blue = np.array([l_h, l_s, l_v])
				upper_blue = np.array([u_h, u_s, u_v])
				imcrop = img[100:300, 425:625]
				hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
				mask = cv2.inRange(hsv, lower_blue, upper_blue)
		# cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
				cv2.imshow("Trackbars", frame)
				cv2.imshow("mask", mask)
				cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
				cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", mask)
				pic_no+=1
				print(pic_no)
				if pic_no == total_pics:
					cam.release()
					cv2.destroyAllWindows()
		# key = cv2.waitKey(1)
		if(cv2.waitKey(1) & 0xFF == ord('q')):
			cam.release()
			cv2.destroyAllWindows()
		

g_id = input("Enter gesture no.: ")
store_images(g_id)




import numpy as np
import cv2
import pickle
#here we import the haarcascadeclassifier on which we will train
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face-trainner.yml')

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):

	ret, frame= cap.read() #for continuosly reading the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting the frame to gray
	faces = face_cascade.detectMultiScale(gray)
	#Now the below for loop is for getting the region of interest
	for (x,y,w,h) in faces:
		#x,y,w,h are the co-ordinate values of region of interest (face)
		roi_gray = gray[y:y+h,x:x+w]
		roi_color = frame[y:y+h, x:x+w]
	
	#now recognizing
		id_, conf = recognizer.predict(roi_gray)
		if conf>=45 and conf <=85:
			font = cv2.FONT_HERSHEY_SIMPLEX
			name=labels[id_]
			color = (255,255,255)
			stroke =2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		color =(255,0,0) #BGR
		stroke =3 #size of the lines
		end_cord_x= x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y),color,stroke)

	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
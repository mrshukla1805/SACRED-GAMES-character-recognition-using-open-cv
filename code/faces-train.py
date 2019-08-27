import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

#Creating the face classifier for creating the training data
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#Creating the recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids ={}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
	#print(root,dirs,files)
	for file in files:
		if file.endswith("jpg"):
			path = os.path.join(root,file)
			#print(path)
			label = os.path.basename(root).replace(" ","-").lower()
			#print(label)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id+=1
			id_ = label_ids[label]
			#print(id_)
			#print(label_ids)
			#the below step converts into grayscale
			pil_image = Image.open(path).convert("L")
			size = (550,550)
			#Antialias basically defines the quality of filtering
			final_image = pil_image.resize(size,Image.ANTIALIAS)
			#this converts the image into array of pixels
			image_array = np.array(final_image,"uint8")
			
			#And now from this array we create our training and labeling data

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				#print(roi)
				x_train.append(roi)
				y_labels.append(id_)


with open("labels.pickle",'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")
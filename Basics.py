import cv2
import numpy as np
import face_recognition

imgNasser = face_recognition.load_image_file('Images/Walid.jpg')
imgNasser = cv2.cvtColor(imgNasser,cv2.COLOR_BGR2RGB)

imgNasserTest = face_recognition.load_image_file('Images/WalidTest.jpg')
imgNasserTest = cv2.cvtColor(imgNasserTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgNasser)[0]
encodeNasser = face_recognition.face_encodings(imgNasser)[0]
cv2.rectangle(imgNasser,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgNasserTest)[0]
encodeNasserTest = face_recognition.face_encodings(imgNasserTest)[0]
cv2.rectangle(imgNasserTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeNasser],encodeNasserTest)
FaceDis = face_recognition.face_distance([encodeNasser], encodeNasserTest)
print(results, FaceDis)

cv2.imshow('Nasser',imgNasser)
cv2.imshow('NasserTest', imgNasserTest)
cv2.waitKey(0)
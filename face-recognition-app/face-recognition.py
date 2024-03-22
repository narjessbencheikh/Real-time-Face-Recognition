import cv2
from simple_facerec import SimpleFacerec

#Encode Images from a folder
sf = SimpleFacerec()
sf.load_encoding_images("images/")

#Face Recognition In Real-Time On A Webcam

 #Load Camera 
cap = cv2.VideoCapture(0)

#Take Webcam Stream
while True :
  ret, frame = cap.read()

#Face Location And Face Recognition
  
  #Detect Faces 
  face_locations, face_names = sf.detect_known_faces(frame)
  for face_loc, name in zip(face_locations, face_names):
    # (y1, x2, y2, x1)
    #y1 : the upper-left corner, x2 : the upper-right corner, y2 : the lower-right corner, x1 : the lower-left corner of the bounding box
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,200), 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,200), 2)
  

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1)
  if key == 27:
    break

cap.release()
cv2.destroyAllWindows()
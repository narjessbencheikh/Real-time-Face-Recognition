#Importing opencv and face_recognition libraries
import cv2
import face_recognition

#Face Encoding First Image
img = cv2.imread("Cristiano1.jpg") #load the image 
#we need to convert it into RGB color format because OpenCV, by default, reads images in BGR format, 
#while the face_recognition library expects images in RGB format.
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img_encoding = face_recognition.face_encodings(rgb_img)[0] #face encoding with the functions of the Face recognition library

#Face Encoding Second Image
img2 = cv2.imread("images/Cristiano.webp") 
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

#Comparison Of the 2 images after encoding
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Img", img)
cv2.imshow("Img 2", img2)
cv2.waitKey(0)
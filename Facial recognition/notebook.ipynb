# Face recognition

# Importing the libraries
import cv2

# Loading the cascades
with open(r'C:\Users\Steve Thomas\Desktop\Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition\haarcascade_frontalface_default.xml', 'r') as file:
    print("File is accessible")


face_cascade = cv2.CascadeClassifier(r'C:\Users\Steve Thomas\Desktop\Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\Steve Thomas\Desktop\Computer_Vision_A_Z_Template_Folder\Module 1 - Face Recognition\haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some Face recognition with the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture")
    exit()
    
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Frame not read correctly")
        break
    
    # Debugging: Check if frame is valid
    if frame is None or frame.size == 0:
        print("Empty frame, skipping")
        continue
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Perform detection
    canvas = detect(gray, frame)
    
    # Display the video frame
    cv2.imshow('Video', canvas)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    
# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
    
            

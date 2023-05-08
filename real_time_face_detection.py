import cv2
import face_recognition


webcamcam_video_stream = cv2.VideoCapture(0)

all_face_locations = []


while True:
    ret, curr_frame = webcamcam_video_stream.read()
    
    curr_frame = cv2.resize(curr_frame, (0,0), fx=1, fy=1)
    all_face_locations = face_recognition.face_locations(curr_frame, 2, model='hog')
    
    for i, curr_face_locations in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = curr_face_locations
        
        cv2.rectangle(curr_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255),2)
        
    cv2.imshow('Webcam', curr_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        

webcamcam_video_stream.release()
cv2.destroyAllWindows()

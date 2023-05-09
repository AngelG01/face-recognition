import cv2
import face_recognition


image_to_detect = cv2.imread('pexels-fauxels-3184398.jpg')

#cv2.imshow('test', image_to_detect)

all_faces_locations = face_recognition.face_locations(image_to_detect, model='hog')
print(f'Number of faces is {len(all_faces_locations)}')

for index, curr_face_location in enumerate(all_faces_locations):
    top_pos, right_pos, bottom_pos, left_pos = curr_face_location
    curr_face_img = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    cv2.imshow(f'{index}', curr_face_img)

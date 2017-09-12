import cv2
import numpy as np

offset = (20, 40)

face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

    for face_coordinates in faces:

        x, y, width, height = face_coordinates
        x_off, y_off = offset
        x1, x2, y1, y2 = x - x_off, x + width + x_off, y - y_off, y + height + y_off

        gray_face = gray_image[y1:y2, x1:x2]
        x = x.astype('float32')
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        gray_face= x
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)


        color = np.asarray((255, 0, 0))
        color = color.astype(int)
        color = color.tolist()

        x, y, w, h = face_coordinates
        cv2.rectangle(rgb_image, (x, y), (x + w, y + h), color, 2)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

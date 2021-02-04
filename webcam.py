import cv2

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# For every frame of the image, if there is a face, draw a square box.

while True:
    success, img = cap.read()
    faces = faceCascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in faces:
        imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (15,15))
        #cv2.circle(img, (x+w//2, y+h//2), 70, (0,255,0),cv2.BL)

    cv2.imshow("Result",img)
    if cv2.waitKey(1)   & 0xFF == ord('q'):
        break
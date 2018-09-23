import cv2

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("./haarcascade_smile.xml")

img = cv2.imread("smiles.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0))
    face_gray = gray[y:y + h, x:x + w]
    face_color = img[y:y + h, x:x + w]
    smiles = smile_cascade.detectMultiScale(face_gray, 1.5, 20)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255))

cv2.imshow("Figura", img)

cv2.waitKey(0)

import cv2

captura = cv2.VideoCapture(0)
cv2.namedWindow('tela')
cv2.resizeWindow('tela', 1280, 720)

while True:
    _, frame1 = captura.read()
    _, frame2 = captura.read()
    frame = cv2.absdiff(frame1, frame2)
    cv2.imshow('tela', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
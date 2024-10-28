import cv2

camera = cv2.VideoCapture(0)

cv2.namedWindow('Embarcados')
# cv2.resizeWindow('Embarcados', 640, 480)

while True:
    ret, frame = camera.read()

    ## Alguns exemplos de processamento com a imagem (descomente para testar)
    # frame = cv2.flip(frame, 0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.Canny(frame, 100, 200)

    cv2.imshow('Embarcados', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # pressione 'q' para sair
        break

camera.release()
cv2.destroyAllWindows()

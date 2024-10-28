import cv2

imagem = cv2.imread('lenna.jpg')

cv2.imshow('Tela', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import time

def nada():
    pass

captura = cv2.VideoCapture(0)
cv2.namedWindow('tela')
cv2.resizeWindow('tela', 1280, 720)
cv2.createTrackbar('blur', 'tela', 1, 151, nada)
cv2.createTrackbar('limiar', 'tela', 1, 151, nada)
cv2.createTrackbar('erosao', 'tela', 0, 151, nada)
cv2.createTrackbar('dilatacao', 'tela', 0, 151, nada)
cv2.setTrackbarPos('limiar', 'tela', 20)
cv2.setTrackbarPos('blur', 'tela', 1)
cv2.setTrackbarPos('erosao', 'tela', 1)
cv2.setTrackbarPos('dilatacao', 'tela', 8)

while True:
    limiar_par = cv2.getTrackbarPos('limiar', 'tela')
    blur_par = cv2.getTrackbarPos('blur', 'tela')
    erosao_iter = cv2.getTrackbarPos('erosao', 'tela')
    dilatacao_iter = cv2.getTrackbarPos('dilatacao', 'tela')
    if blur_par % 2 == 0:
        blur_par += 1
    _, frame1 = captura.read()
    _, frame2 = captura.read()
    frame = cv2.absdiff(frame1, frame2)
    frame_cinza = cv2.cvtColor(frame, 
                               cv2.COLOR_RGB2GRAY)
    frame_cinza = cv2.GaussianBlur(frame_cinza,
                                   (blur_par, blur_par), 0)
    frame_binario = cv2.threshold(frame_cinza, 
                                  limiar_par, 
                                  255, 
                                  cv2.THRESH_BINARY)[1]
    frame_erosao = cv2.erode(frame_binario, 
                             None, 
                             iterations=erosao_iter)
    frame_dilatacao = cv2.dilate(frame_erosao, 
                             None, 
                             iterations=dilatacao_iter)
    contours, hierarchy = cv2.findContours(frame_dilatacao, 
                                           cv2.RETR_LIST, 
                                           cv2.CHAIN_APPROX_NONE)
    if contours:
        frame1 = cv2.drawContours(frame1, 
                         contours, 
                         -1, 
                         (0, 255, 0), 
                         3)
    cv2.imshow('tela', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
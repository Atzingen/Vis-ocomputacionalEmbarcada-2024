import cv2
import numpy as np

def emtpy_callback(_):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('Embarcados')
# cv2.resizeWindow('Filtros', 640, 480)
screen_width, screen_height = 1920, 1080
padding_height = 100


cv2.createTrackbar("LH", "Embarcados", 0, 179, emtpy_callback)
cv2.createTrackbar("LS", "Embarcados", 0, 255, emtpy_callback)
cv2.createTrackbar("LV", "Embarcados", 0, 255, emtpy_callback)
cv2.createTrackbar("UH", "Embarcados", 179, 179, emtpy_callback)
cv2.createTrackbar("US", "Embarcados", 255, 255, emtpy_callback)
cv2.createTrackbar("UV", "Embarcados", 255, 255, emtpy_callback)
# cv2.setTrackbarPos("LH", "Embarcados", 102)
# cv2.setTrackbarPos("LS", "Embarcados", 122)
# cv2.setTrackbarPos("LV", "Embarcados", 44)
# cv2.setTrackbarPos("UH", "Embarcados", 123)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000
params.maxArea = 1000000
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("LH", "Embarcados")
    ls = cv2.getTrackbarPos("LS", "Embarcados")
    lv = cv2.getTrackbarPos("LV", "Embarcados")
    uh = cv2.getTrackbarPos("UH", "Embarcados")
    us = cv2.getTrackbarPos("US", "Embarcados")
    uv = cv2.getTrackbarPos("UV", "Embarcados")

    lower = np.array([lh, ls, lv])
    upper = np.array([uh, us, uv])
    mask = cv2.inRange(hsv, lower, upper)

    inverted_mask = cv2.bitwise_not(mask)
    keypoints = detector.detect(inverted_mask)
    frame_with_keypoints = frame.copy()
    # Loop over the detected keypoints
    for kp in keypoints:
        # Get the coordinates of the keypoint
        x, y = int(kp.pt[0]), int(kp.pt[1])
        # Get the size of the keypoint
        size = int(kp.size)

        # Draw a circle around the keypoint (increase thickness here)
        frame_with_keypoints = cv2.circle(frame, (x, y), size, (0, 255, 255), thickness=5)
    # frame_with_keypoints = cv2.drawKeypoints(frame, 
    #                                          keypoints, 
    #                                          np.array([]), 
    #                                          (0, 255, 255),
    #             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    frame_height, frame_width = frame.shape[:2]
    combined_width = 2 * frame_width
    scale_factor = min(screen_width / combined_width, screen_height / frame_height, 1)

    # Resize se necessário
    if scale_factor < 1:
        new_width = int(frame_width * scale_factor)
        new_height = int(frame_height * scale_factor)
        inverted_mask = cv2.resize(inverted_mask, (new_width, new_height))
        frame_with_keypoints = cv2.resize(frame_with_keypoints, (new_width, new_height))
    mask_bgr = cv2.bitwise_not(cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR))
    combined_frame = cv2.hconcat([mask_bgr, frame_with_keypoints])
    padded_frame = np.zeros((combined_frame.shape[0] + padding_height, combined_frame.shape[1], 3), dtype=np.uint8)
    padded_frame[:combined_frame.shape[0], :] = combined_frame

    cv2.imshow("Embarcados", padded_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
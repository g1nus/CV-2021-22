import cv2
import numpy as np

webcam = False

if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")


while video.isOpened():
    ret, frame = video.read()
    frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Applying the function
    fast = cv2.FastFeatureDetector_create(20, True, cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
    
    # Drawing the keypoints
    kp = fast.detect(gray_frame, None)
    print(len(kp))
    kp_image = cv2.drawKeypoints(frame, kp, None, color=(0, 255, 0))
    # Displaying the image
    cv2.imshow("Video", kp_image)

    if cv2.waitKey(1) == ord('q') or not ret:
        break

video.release()
cv2.destroyAllWindows()
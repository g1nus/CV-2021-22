import cv2
import numpy as np

"""
TODO: tweak the kp predicted (line 23: fast initialization) an maybe refractor code
"""

webcam = False

if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")

ret, frame = video.read()
frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Applying the function
fast = cv2.FastFeatureDetector_create(20, True, cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)
# Extract the keypoints
kps = fast.detect(gray_frame, None)
corners_predicted = cv2.KeyPoint.convert(kps)
total_points = len(corners_predicted)
print(f"=============\ntotal corners detected: {total_points}\n=============\n")
kp_image = cv2.drawKeypoints(frame, kps, None, color=(0, 255, 0))
# Displaying the image
cv2.imshow("Video", kp_image)
cv2.waitKey(0)

while video.isOpened():
    prev_frame = frame.copy()
    prev_corners = corners_predicted

    ret, frame = video.read()
    frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))

    corners_predicted, status_predicted, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_corners, None)
    loss_points = {"right": 0, "left": 0, "top": 0, "bottom": 0, "total": 0}
    for idx, item in enumerate(corners_predicted.astype(int)):
        x, y = item
        x = int(x)
        y = int(y)
        if x > int(WIDTH/5):
            loss_points["right"] += 1
        elif x < 0:
            loss_points["left"] += 1
        elif y > int(HEIGHT/5):
            loss_points["bottom"] += 1
        elif y < 0:
            loss_points["top"] += 1
        cv2.circle(frame, (x, y), 6, (0, 255, 0), 1)
    
    cv2.imshow("Video", frame)

    loss_points["total"] = loss_points["right"] + loss_points["left"] + loss_points["top"] + loss_points["bottom"]

    print(f"Total points lost/total : {loss_points['total']}/{total_points} (r: {loss_points['right']} - l: {loss_points['left']} - t: {loss_points['top']} - b: {loss_points['bottom']})")
    
    if(loss_points["total"] > len(corners_predicted)/2):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect keypoints
        kps = fast.detect(gray_frame, None)
        corners_predicted = cv2.KeyPoint.convert(kps)
        total_points = len(corners_predicted)
        print(f"=============\ntotal corners detected: {total_points}\n=============\n")
    

    if cv2.waitKey(1) == ord('q') or not ret:
        break

video.release()
cv2.destroyAllWindows()
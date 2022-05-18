import cv2
import numpy as np

webcam = False
NUM_FEATURES = 80

sift = cv2.SIFT_create(NUM_FEATURES, 6, 0.1)


if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")

# Naming a window
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)   
# Using resizeWindow()
cv2.resizeWindow("Video", int(WIDTH/5 * 1.06), int(HEIGHT/5 * 1.06))

# I detect the first keypoints
ret, frame = video.read()
frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detect keypoints and draw them on screen
kps, dsc = sift.detectAndCompute(frame, None)
frame = cv2.drawKeypoints(gray_frame, kps, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
corners_predicted = cv2.KeyPoint.convert(kps)
print(f"initially detected: {len(corners_predicted)}")

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

    # Displaying the image
    cv2.imshow("Video", frame)
    
    loss_points["total"] = loss_points["right"] + loss_points["left"] + loss_points["top"] + loss_points["bottom"]

    if(loss_points["total"] > len(corners_predicted)/2):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detect keypoints
        kps, dsc = sift.detectAndCompute(frame, None)
        corners_predicted = cv2.KeyPoint.convert(kps)

    if cv2.waitKey(1) == ord('q') or not ret:
        break
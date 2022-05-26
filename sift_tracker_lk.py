import cv2
import numpy as np

webcam = False
NUM_FEATURES = 80

n_frames = 0
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

"""
FUNCTIONS
"""

# gets a frame in the resized ratio
def getResizedFrame(video, ratio):
    global n_frames
    ret, frame = video.read()
    if not ret:
        print("no more frames to read...")
        exit()
    frame = cv2.resize(frame, (int(WIDTH/ratio), int(HEIGHT/ratio)))
    n_frames += 1
    return frame

# detects keypoints and draws them on frame
def detectKeypoints(frame):
    global sift
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect keypoints and draw them on screen
    kps, dsc = sift.detectAndCompute(frame, None)
    frame = cv2.drawKeypoints(gray_frame, kps, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    corners_predicted = cv2.KeyPoint.convert(kps)
    return frame, corners_predicted

# draws circles over the predicted corners
def drawCircleCorners(frame, corners_predicted, radius, colour, thickness):
    for item in corners_predicted.astype(int):
        x, y = item
        x = int(x)
        y = int(y)
        cv2.circle(frame, (x, y), radius, colour, thickness)
    return frame

# returns details regarding how many corners are lost
def getLoss(corners_predicted):
    global WIDTH, HEIGHT
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
    loss_points["total"] = loss_points["right"] + loss_points["left"] + loss_points["top"] + loss_points["bottom"]
    return loss_points

"""
MAIN CODE
"""

# I read the first frame and get the first set of keypoints
frame = getResizedFrame(video, 5)
frame, corners_predicted = detectKeypoints(frame)
print(f"initially detected: {len(corners_predicted)}")

while video.isOpened():
    # I keep information about the previous processing (will be useful for the prediction)
    prev_frame = frame.copy()
    prev_corners = corners_predicted

    # read the next frame and predict the new position of the corners
    frame = getResizedFrame(video, 5)
    corners_predicted, status_predicted, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, prev_corners, None)
    frame = drawCircleCorners(frame, corners_predicted, 6, (0, 255, 0), 1)
    print(f"n_frames: {n_frames}")

    # check the lost corners and, in case, detect new keypoints
    loss_points = getLoss(corners_predicted)
    if(loss_points["total"] > len(corners_predicted)/2):
        _, corners_predicted = detectKeypoints(frame)

    # Displaying the image
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
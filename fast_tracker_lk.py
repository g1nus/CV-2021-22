import cv2

alpha = 0.7
webcam = False
n_frames = 0
fast = cv2.FastFeatureDetector_create(20, True, cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)

if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")

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
    global fast
    work_frame = frame.copy()
    gray_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    #detect keypoints and draw them on screen
    kps = fast.detect(gray_frame, None)
    work_frame = cv2.drawKeypoints(work_frame, kps, None, color=(0, 255, 0))
    corners_predicted = cv2.KeyPoint.convert(kps)
    return work_frame, corners_predicted

# draws circles over the predicted corners
def drawCircleCorners(frame, corners_predicted, radius, colour, thickness):
    global alpha
    work_frame = frame.copy()
    overlay = frame.copy()
    for item in corners_predicted.astype(int):
        x, y = item
        x = int(x)
        y = int(y)
        cv2.circle(overlay, (x, y), radius, colour, thickness)
    work_frame = cv2.addWeighted(overlay, alpha, work_frame, 1 - alpha, 0)
    return work_frame

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
clear_frame = getResizedFrame(video, 5)
work_frame, corners_predicted = detectKeypoints(clear_frame)
total_points = len(corners_predicted)
# Displaying the image
cv2.imshow("Video", work_frame)

while video.isOpened():
    # I keep information about the previous processing (will be useful for the prediction)
    prev_frame = clear_frame.copy()
    prev_corners = corners_predicted

    # read the next frame
    clear_frame = getResizedFrame(video, 5)
    # predict the new position only if there are detected corners
    if(total_points > 0):
        corners_predicted, status_predicted, err = cv2.calcOpticalFlowPyrLK(prev_frame, clear_frame, prev_corners, None)
        work_frame = drawCircleCorners(clear_frame, corners_predicted,  6, (0, 255, 0), 2)
        
        # analyze how many points were lost
        loss_points = getLoss(corners_predicted)
    
    #if I lost too many poins I detect new corners
    if(loss_points["total"] > len(corners_predicted)/2):
        _, corners_predicted = detectKeypoints(clear_frame)
        total_points = len(corners_predicted)
    
    cv2.imshow("Video", work_frame)
    if(total_points > 0):
        print(f"nth_frame: {n_frames} | total_kp: {total_points} | loss: {round(loss_points['total']/total_points, 2)}")
    else:
        print(f"nth_frame: {n_frames} | ZERO corners detected")
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
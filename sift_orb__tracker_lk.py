import time
import cv2
import pandas as pd

"""
TODO: create a history of stats
        - total of overlapping points               <========!!!!
        - overlap feature points of the 3 detectors
"""

alpha = 0.7
webcam = False
sift = True
NUM_FEATURES = 80

n_frames = 0
last_frame_update = 0
detector = None

data_hist = {'frame' : [], 'last_frame_update': [], 'corners': [], 'time': []}

if sift:
    detector = cv2.SIFT_create(NUM_FEATURES, 6, 0.1)
else:
    detector = cv2.ORB_create(NUM_FEATURES)

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

def saveData():
    data_hist['time'][len(data_hist['time']) - 1] = time.time()
    df = pd.DataFrame(data_hist)
    print(df)
    df.to_csv(f'output/{"SIFT" if sift else "ORB"}.csv', index = False, header = True)

def saveFrame(frame, info):
    global sift
    if sift:
        cv2.imwrite(f'output/frame{n_frames}_{info}_sift.jpg', frame)
    else:
        cv2.imwrite(f'output/frame{n_frames}_{info}_orb.jpg', frame)

# gets a frame in the resized ratio
def getResizedFrame(video, ratio):
    global n_frames
    ret, frame = video.read()
    if not ret:
        print("no more frames to read...")
        saveData()
        exit()
    frame = cv2.resize(frame, (int(WIDTH/ratio), int(HEIGHT/ratio)))
    n_frames += 1
    return frame

# detects keypoints and draws them on frame
def detectKeypoints(frame):
    global detector, last_frame_update
    work_frame = frame.copy()
    gray_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    #detect keypoints and draw them on screen
    kps, dsc = detector.detectAndCompute(work_frame, None)
    work_frame = cv2.drawKeypoints(gray_frame, kps, work_frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    saveFrame(work_frame, 'keypoints')
    corners_predicted = cv2.KeyPoint.convert(kps)

    last_frame_update = n_frames
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

def saveFrameWithKeypoints(frame):
    f = frame.copy()
    nf, _ = detectKeypoints(f)

def appendHistoryData(corners):
    global last_frame_update, n_frames
    data_hist['frame'].append(n_frames)
    data_hist['last_frame_update'].append(last_frame_update)
    data_hist['corners'].append(corners)
    if n_frames == 1:
        data_hist['time'].append(time.time())
    else:
        data_hist['time'].append(data_hist['time'][len(data_hist['time']) - 1])

"""
MAIN CODE
"""

# I read the first frame and get the first set of keypoints
clear_frame = getResizedFrame(video, 5)
work_frame, corners_predicted = detectKeypoints(clear_frame)
print(f"initially detected: {len(corners_predicted)}")
appendHistoryData(len(corners_predicted))
# Displaying the image
cv2.imshow("Video", work_frame)

while video.isOpened():
    # I keep information about the previous processing (will be useful for the prediction)
    prev_frame = clear_frame.copy()
    prev_corners = corners_predicted

    # read the next frame and predict the new position of the corners
    clear_frame = getResizedFrame(video, 5)
    corners_predicted, status_predicted, err = cv2.calcOpticalFlowPyrLK(prev_frame, clear_frame, prev_corners, None)
    work_frame = drawCircleCorners(clear_frame, corners_predicted, 6, (0, 255, 0), 2)

    # check the lost corners and, in case, detect new keypoints
    loss_points = getLoss(corners_predicted)
    if(loss_points["total"] > len(corners_predicted)/2):
        _, corners_predicted = detectKeypoints(clear_frame)

    print(f"nth_frame: {n_frames} | loss: {loss_points['total']}")

    appendHistoryData(len(corners_predicted) - loss_points['total'])

    if(n_frames == 220):
        saveFrame(clear_frame, 'clear')
        saveFrame(work_frame, 'work')
        #saveFrameWithKeypoints(clear_frame)
    elif(n_frames == 100):
        saveFrame(clear_frame, 'clear')
        saveFrame(work_frame, 'work')
        #saveFrameWithKeypoints(clear_frame)

    # Displaying the image
    cv2.imshow("Video", work_frame)
    if cv2.waitKey(1) == ord('q'):
        break

"""
Cleanup and close
"""

saveData()
video.release()
cv2.destroyAllWindows()
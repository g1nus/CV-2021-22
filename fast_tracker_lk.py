import time
import cv2
import pandas as pd

"""
TODO: create a history of stats
"""

alpha = 0.7
webcam = False
n_frames = 0
last_frame_update = 0
fast = cv2.FastFeatureDetector_create(20, True, cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)

data_hist = {'frame' : [], 'last_frame_update': [], 'corners': [], 'clusters': [], 'time': []}

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

def saveData():
    data_hist['time'][len(data_hist['time']) - 1] = time.time()
    df = pd.DataFrame(data_hist)
    print(df)
    df.to_csv(f'output/FAST.csv', index = False, header = True)

def saveFrame(frame, info):
    cv2.imwrite(f'output/frame{n_frames}_{info}_fast.jpg', frame)

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
    global fast, last_frame_update
    work_frame = frame.copy()
    gray_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    equ_frame = cv2.equalizeHist(gray_frame)
    #detect keypoints and draw them on screen
    kps = fast.detect(equ_frame, None)
    work_frame = cv2.drawKeypoints(work_frame, kps, None, color=(0, 255, 0))
    saveFrame(work_frame, 'keypoints')
    corners_predicted = cv2.KeyPoint.convert(kps)

    last_frame_update = n_frames
    return work_frame, corners_predicted

# detects keypoints and draws them on frame
def detectKeypointsToSave(frame):
    global fast
    work_frame = frame.copy()
    gray_frame = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    equ_frame = cv2.equalizeHist(gray_frame)
    #detect keypoints and draw them on screen
    kps = fast.detect(equ_frame, None)
    work_frame = cv2.drawKeypoints(work_frame, kps, None, color=(0, 255, 0))
    saveFrame(work_frame, 'screen_keypoints')
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
    legal_points = list()
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
        else:
            legal_points.append((x,y))
    loss_points["total"] = loss_points["right"] + loss_points["left"] + loss_points["top"] + loss_points["bottom"]
    return loss_points, legal_points

def getAggregatePoints(legal_points, clusters):
    x, y = legal_points.pop(0)
    #print(f"{legal_points}\n===============================\npicked point = {(x, y)} | total len({len(legal_points)})")
    #close_ones = list(filter(lambda item: (item[0] != x or item[1] != y) and item[0] >= x - 10 and item[0] <= x + 10 and item[1] >= y - 10 and item[1] <= y + 10, legal_points))
    close_ones = list(filter(lambda item: item[0] >= x - 10 and item[0] <= x + 10 and item[1] >= y - 10 and item[1] <= y + 10, legal_points))
    clusters.append([(x, y)])
    clusters[len(clusters) - 1].extend(close_ones)
    #print(f"close ones are {clusters[len(clusters) - 1]} | {(x,y) not in clusters[len(clusters) - 1]}")
    legal_points = [p for p in legal_points if p not in clusters[len(clusters) - 1]]

    new_points_found = True
    extra_points = list()
    while new_points_found:
        for point in clusters[len(clusters) - 1]:
            x, y = point
            extra_points.extend(list(filter(lambda item: (item[0] != x or item[1] != y) and item[0] >= x - 10 and item[0] <= x + 10 and item[1] >= y - 10 and item[1] <= y + 10, legal_points)))
            legal_points = [p for p in legal_points if p not in extra_points]

        if(len(extra_points) > 0):
            ep = list(extra_points)
            new_points_found = True
            clusters[len(clusters) - 1].extend(ep)
            legal_points = [x for x in legal_points if x not in clusters[len(clusters) - 1]]
            extra_points = []
        else:
            new_points_found = False

    return legal_points, clusters


def saveFrameWithKeypoints(frame):
    f = frame.copy()
    nf, _ = detectKeypointsToSave(f)

def appendHistoryData(corners, clusters):
    if(corners < 0):
        print("WARNING")
        cv2.waitKey(0)
    global last_frame_update, n_frames
    data_hist['frame'].append(n_frames)
    data_hist['last_frame_update'].append(last_frame_update)
    data_hist['corners'].append(corners)
    data_hist['clusters'].append(clusters)
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
total_points = len(corners_predicted)
appendHistoryData(total_points, 0)
# Displaying the image
cv2.imshow("Video", work_frame)

size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/5),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/5))
fps = 28
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
vout = cv2.VideoWriter()
success = vout.open('output_fast.mp4',fourcc,fps,size,True) 

while video.isOpened():
    # I keep information about the previous processing (will be useful for the prediction)
    prev_frame = clear_frame.copy()
    prev_corners = corners_predicted

    clusters = list()

    total_legal_points = 0

    # read the next frame
    clear_frame = getResizedFrame(video, 5)
    # predict the new position only if there are detected corners
    if(total_points > 0):
        corners_predicted, status_predicted, err = cv2.calcOpticalFlowPyrLK(prev_frame, clear_frame, prev_corners, None)
        work_frame = drawCircleCorners(clear_frame, corners_predicted,  6, (0, 255, 0), 2)
        
        # analyze how many points were lost
        loss_points, legal_points = getLoss(corners_predicted)
        total_legal_points = len(legal_points)
        while(len(legal_points) > 0):
            legal_points, clusters = getAggregatePoints(legal_points, clusters)
    
    #if I lost too many poins I detect new corners
    if(loss_points["total"] > len(corners_predicted)/2) or (total_points < 10 and (n_frames - last_frame_update) > 10):
        _, corners_predicted = detectKeypoints(clear_frame)
        total_points = len(corners_predicted)
    
    if(n_frames == 220 or n_frames == 100 or n_frames == 440):
        saveFrame(clear_frame, 'screen_clear')
        saveFrame(work_frame, 'screen_work')
        saveFrameWithKeypoints(clear_frame)

    cv2.imshow("Video", work_frame)
    vout.write(work_frame)
    if(total_points > 0):
        print(f"nth_frame: {n_frames} ({n_frames - last_frame_update}) | total_kp: {total_points} | loss: {round(loss_points['total']/total_points, 2)} | clusters: {len(clusters)}")
        appendHistoryData(total_legal_points, len(clusters))
    else:
        print(f"nth_frame: {n_frames} | ZERO corners detected | clusters: {len(clusters)}")
        appendHistoryData(0, 0)
    if cv2.waitKey(1) == ord('q'):
        break

saveData()
video.release()
vout.release()
cv2.destroyAllWindows()
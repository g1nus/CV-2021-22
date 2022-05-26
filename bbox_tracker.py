import cv2
import numpy as np

webcam = False
prev_dilated_edges = []
n_frames = 0
trackers = cv2.legacy.MultiTracker_create()
previous_boxes = []
previous_boxes_len = 0
original_box_len = 0

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

def boxDiff(box1, box2):
    (x1, y1, w1, h1) = [int(v) for v in box1]
    (x2, y2, w2, h2) = [int(v) for v in box2]
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    w_diff = abs(w1 - w2)
    h_diff = abs(h1 - h2)
    return max(x_diff, y_diff, w_diff, h_diff)

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

# find contours of the frame
def findContours(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_frame, (9,9), 0)
    thresh_frame = cv2.threshold(img_blur, 0, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchies = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchies

# track the bboxes obtained by looking at the contours
def setUpTrackers(frame, contours, hierarchies):
    global previous_boxes, trackers
    for c, s in zip(contours, hierarchies[0]):
        x,y,w,h = cv2.boundingRect(c)
        if w > 40 and h > 5 and w < int(WIDTH/10) and h < int(HEIGHT/10) and s[3] == -1:
            previous_boxes.append({"box": (x,y,w,h), "diff": 0})
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
            tracker = cv2.legacy.TrackerMedianFlow_create()
            trackers.add(tracker, frame, (x,y,w,h))

# update trackers to track only clean boxes
def updateTrackers(clean_boxes,frame):
    global previous_boxes, trackers
    trackers = cv2.legacy.MultiTracker_create()
    previous_boxes = clean_boxes
    for p_box in previous_boxes:
        tracker = cv2.legacy.TrackerMedianFlow_create()
        trackers.add(tracker, frame, p_box['box'])

# get how much the boxes have changed (overall average)
def getAvgDiff(boxes):
    global previous_boxes
    avg_diff = 0
    for idx, box in enumerate(boxes):
        b_diff = boxDiff(previous_boxes[idx]['box'], box)
        previous_boxes[idx]['diff'] = b_diff
        avg_diff += b_diff
        previous_boxes[idx]['box'] = box
        
    avg_diff = avg_diff/previous_boxes_len
    return avg_diff

# get only boxes that morphed into a normal way
def getCleanBoxes(avg_diff):
    global previous_boxes
    return list(filter(lambda p_box: (p_box['diff'] <= 6 or p_box['diff'] <= (avg_diff * 1.5)), previous_boxes))

# draw boxes on screen
def drawBoxes(frame, avg_diff):
    global previous_boxes
    # detect boxes that have changed shape too fast
    for pbox in previous_boxes:
        (x, y, w, h) = [int(v) for v in pbox['box']]
        if(pbox['diff'] > 6 and pbox['diff'] > (avg_diff * 1.5)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

# initial (or reset) setup
def setUpScene(frame):
    global trackers, previous_boxes, previous_boxes_len, original_box_len
    trackers = cv2.legacy.MultiTracker_create()
    contours, hierarchies = findContours(frame)
    previous_boxes = []
    previous_boxes_len = 0
    setUpTrackers(frame, contours, hierarchies)
    previous_boxes_len = len(previous_boxes)
    original_box_len = previous_boxes_len

"""
MAIN CODE
"""

frame = getResizedFrame(video, 5)
setUpScene(frame)
print(f"Drawn bboxes: {previous_boxes_len}")

# Displaying the image
cv2.imshow("Video", frame)
cv2.waitKey(0)

while video.isOpened():
    # get a frame
    frame = getResizedFrame(video, 5)

    # update the position of the tracked objects
    (success, boxes) = trackers.update(frame)

    # check how much the boxes have changed (in average)
    avg_diff = getAvgDiff(boxes)
    drawBoxes(frame, avg_diff)
    cv2.imshow("Video", frame)

    # get only the boxes which are not abnormal
    clean_boxes = getCleanBoxes(avg_diff)
    print(f"nth_frame {n_frames} | clean: ({len(clean_boxes)}/{original_box_len}) | the avg difference is: {round(avg_diff, 2)}")

    # remove bad boxes or update whole scene if there are not enough bboxes left
    if (len(previous_boxes) - len(clean_boxes)) > 0:
        if(len(clean_boxes) < original_box_len/2):
            print("setting up new scene")
            setUpScene(frame)
            #print(f"Drawn bboxes: {previous_boxes_len}")
        else:
            print("clean the scene")
            # remove illegal boxes from tracking
            updateTrackers(clean_boxes, frame)

    prev_dilated_edges.append(frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
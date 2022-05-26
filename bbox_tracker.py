import cv2
import numpy as np

def box_diff(box1, box2):
    (x1, y1, w1, h1) = [int(v) for v in box1]
    (x2, y2, w2, h2) = [int(v) for v in box2]
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    w_diff = abs(w1 - w2)
    h_diff = abs(h1 - h2)
    return max(x_diff, y_diff, w_diff, h_diff)

webcam = False

if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")
prev_dilated_edges = []

#create first bounding boxes
ret, frame = video.read()
frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(gray_frame, (9,9), 0)
thresh_frame = cv2.threshold(img_blur, 0, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, hierarchies = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found boxes: {len(contours)}")

trackers = cv2.legacy.MultiTracker_create()

previous_boxes = []
previous_boxes_len = 0

for c, s in zip(contours, hierarchies[0]):
    x,y,w,h = cv2.boundingRect(c)
    if w > 40 and h > 5 and w < int(WIDTH/10) and h < int(HEIGHT/10) and s[3] == -1:
        previous_boxes.append({"box": (x,y,w,h), "diff": 0})
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
        tracker = cv2.legacy.TrackerMedianFlow_create()
        trackers.add(tracker, frame, (x,y,w,h))

previous_boxes_len = len(previous_boxes)
original_box_len = previous_boxes_len
print(f"Drawn bboxes: {previous_boxes_len}")

# Displaying the image
cv2.imshow("Video", frame)
cv2.waitKey(0)

while video.isOpened():
    ret, frame = video.read()
    frame = cv2.resize(frame, (int(WIDTH/5), int(HEIGHT/5)))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_frame, (9,9), 0)
    #edges = cv2.Canny(img_blur,60,150, L2gradient=True)
    #kernel = np.ones((3,3), np.uint8) # used for the dilation
    #dilated_edges = cv2.dilate(edges, kernel, iterations=4)
    thresh_frame = cv2.threshold(img_blur, 0, 200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    #shown_frame = dilated_edges
    #if len(prev_dilated_edges) > 3:
    #    for idx, prev_f in enumerate(reversed(prev_dilated_edges)):
    #        print(idx)
    #        if idx > 1:
    #            break
    #        shown_frame = cv2.addWeighted(shown_frame, 0.8, prev_f, 0.8, 0)

    #contours, hierarchies = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(hierarchies)
    #for c, s in zip(contours, hierarchies[0]):
    #    x,y,w,h = cv2.boundingRect(c)
    #    if w > 40 and h > 5 and w < int(WIDTH/10) and h < int(HEIGHT/10) and s[3] == -1:
    #        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)

    (success, boxes) = trackers.update(frame)

    avg_diff = 0
    for idx, box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        b_diff = box_diff(previous_boxes[idx]['box'], box)
        previous_boxes[idx]['diff'] = b_diff
        avg_diff += b_diff
        print(f"[{previous_boxes[idx]['box']}] - [{box}], box diff: {b_diff}")
        previous_boxes[idx]['box'] = box
        
    avg_diff = avg_diff/previous_boxes_len

    print(f"the avg difference is: {avg_diff}")

    culprit_ids = []
    # detect boxes that have changed shape too fast
    for idx, pbox in enumerate(previous_boxes):
        (x, y, w, h) = [int(v) for v in pbox['box']]
        if(pbox['diff'] > 6 and pbox['diff'] > (avg_diff * 1.5)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            culprit_ids.append(idx)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Displaying the image
    cv2.imshow("Video", frame)

    if len(culprit_ids) > 0:
        if((len(previous_boxes) - len(culprit_ids)) < original_box_len/2):
            trackers = cv2.legacy.MultiTracker_create()
            contours, hierarchies = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            previous_boxes = []
            previous_boxes_len = 0
            for c, s in zip(contours, hierarchies[0]):
                x,y,w,h = cv2.boundingRect(c)
                if w > 40 and h > 5 and w < int(WIDTH/10) and h < int(HEIGHT/10) and s[3] == -1:
                    previous_boxes.append({"box": (x,y,w,h), "diff": 0})
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                    tracker = cv2.legacy.TrackerMedianFlow_create()
                    trackers.add(tracker, frame, (x,y,w,h))
            previous_boxes_len = len(previous_boxes)
            original_box_len = previous_boxes_len
            print(f"Drawn bboxes: {previous_boxes_len}")
        else:
            # remove illegal boxes from tracking
            trackers = cv2.legacy.MultiTracker_create()
            previous_boxes = list(filter(lambda p_box: (p_box['diff'] <= 6 or p_box['diff'] <= (avg_diff * 1.5)), previous_boxes))
            for p_box in previous_boxes:
                tracker = cv2.legacy.TrackerMedianFlow_create()
                trackers.add(tracker, frame, p_box['box'])

    prev_dilated_edges.append(frame)

    if cv2.waitKey(1) == ord('q') or not ret:
        break

video.release()
cv2.destroyAllWindows()
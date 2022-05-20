import cv2
from cv2 import dilate
from matplotlib.pyplot import show
import numpy as np
from sqlalchemy import true

webcam = False

if webcam:
    video = cv2.VideoCapture(0)
else:
    video = cv2.VideoCapture("videos/industrial.mp4")

WIDTH = video.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"{WIDTH} x {HEIGHT}")
prev_dilated_edges = []

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

    contours, hierarchies = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(hierarchies)
    for c, s in zip(contours, hierarchies[0]):
        x,y,w,h = cv2.boundingRect(c)
        if w > 40 and h > 5 and w < int(WIDTH/10) and h < int(HEIGHT/10) and s[3] == -1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

    # Displaying the image
    cv2.imshow("Video", frame)

    prev_dilated_edges.append(frame)

    if cv2.waitKey(1) == ord('q') or not ret:
        break

video.release()
cv2.destroyAllWindows()
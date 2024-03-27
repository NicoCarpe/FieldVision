import cv2
import numpy as np
from DLT import select_points, estimate_homography, apply_homography

# Function to initialize MeanShift tracking with user-selected ROI
def select_user_rois(frame):
    # User selects the rois in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')

    # returns a list of tuples, each representing an roi in (x, y, w, h) format
    return rois

# Read video
cap = cv2.VideoCapture("./assets/singles_2.mp4")

# Retrieve the very first frame from the video
_, frame = cap.read()
frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

# Variables to store ROI coordinates and flags
track_windows = select_user_rois(frame)
if track_windows is None:
    print("No ROI selected. Exiting.")
    exit()

tennis_court = cv2.imread('./assets/tennis_court_background.png')
tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))

num_points = 4
pts1 = select_points(frame, num_points)
pts2 = select_points(tennis_court.copy(), num_points)

H = estimate_homography(pts1, pts2)

# Termination criteria, either 15 iteration or by at least 2 pt
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)

# Background subtractor
backSub = cv2.createBackgroundSubtractorKNN()

players = [[0,0], [0,0]]

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    if frame is None:
        break
    
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    fgMask = cv2.GaussianBlur(fgMask, (5, 5), 0)
    _, fgMask = cv2.threshold(fgMask, 230, 255, 0)

    # Convert BGR to HSV format COLOR_BGR2HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # crop out the field
    field_mask = np.zeros_like(fgMask)
    points = np.array([[360, 130], [360, 1000], [1500, 1000], [1500, 130]])
    points = points // 2
    cv2.fillPoly(field_mask, [points], color=(255, 255, 255))   
    
    field = cv2.bitwise_and(field_mask, fgMask)

    # debug
    show = field # cv2.resize(field, (frame.shape[1]//2, frame.shape[0]//2))
    cv2.imshow('dst', show)

    # Applying meanshift to get the new region
    for i, track_window in enumerate(track_windows):
        _, track_windows[i] = cv2.CamShift(field, track_window, termination) # meanshift

        # Draw track window on the frame
        x, y, w, h = track_windows[i]
        vid = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        players[i] = [x + w//2, y +h]
        vid = cv2.circle(vid, (players[i][0], players[i][1]), 5, (0, 0, 255), -1)

    # Show results
    cv2.imshow('Tracker', vid)

    # warped_image = apply_homography(vid, H)

    # apply homography to the points to get the new points
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)

    # draw the new points
    radar = tennis_court.copy()
    for i, point in enumerate(new_points[0]):
        x, y = point
        radar = cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    cv2.imshow('Radar', radar)

    k = cv2.waitKey(10)
    if k == ord('q') or k == 27:  # Quit on 'q' or Esc key
        break

cap.release()

cv2.destroyAllWindows()

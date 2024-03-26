import cv2
import numpy as np

# Function to initialize MeanShift tracking with user-selected ROI
def select_user_rois(frame):
    # User selects the rois in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')

    # returns a list of tuples, each representing an roi in (x, y, w, h) format
    return rois

# Read video
cap = cv2.VideoCapture("video_1.mp4")

# Retrieve the very first frame from the video
_, frame = cap.read()
frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

# Variables to store ROI coordinates and flags
track_windows = select_user_rois(frame)
if track_windows is None:
    print("No ROI selected. Exiting.")
    exit()

# Termination criteria, either 15 iteration or by at least 2 pt
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)

# Background subtractor
backSub = cv2.createBackgroundSubtractorKNN()

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    if frame is None:
        break
    
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    fgMask = cv2.GaussianBlur(fgMask, (3, 3), 0)
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
        _, track_windows[i] = cv2.meanShift(field, track_window, termination)

        # Draw track window on the frame
        x, y, w, h = track_windows[i]
        vid = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    # Show results
    cv2.imshow('Tracker', vid)

    k = cv2.waitKey(10)
    if k == ord('q') or k == 27:  # Quit on 'q' or Esc key
        break

# Release cap object
cap.release()

# Destroy all opened windows
cv2.destroyAllWindows()

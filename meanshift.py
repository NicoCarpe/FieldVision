import cv2
import numpy as np

# Function to initialize MeanShift tracking with user-selected ROI
def select_roi(event, x, y, flags, param):
    global p, q, r, s, selecting_roi, roi_selected, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        p, q, r, s = x, y, 0, 0
        selecting_roi = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            frame_copy = frame.copy()  # Create a copy of the frame
            r, s = x - p, y - q
            cv2.rectangle(frame_copy, (p, q), (x, y), (0, 255, 0), 2)  # Draw the rectangle dynamically
            cv2.putText(frame_copy, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        print('ROI selected at [{}, {}, {}, {}]'.format(p, q, r, s))
        roi_selected = True

# Read video
cap = cv2.VideoCapture('field_map.mp4')

# Retrieve the very first frame from the video
_, frame = cap.read()
frame_copy = frame.copy()

# Variables to store ROI coordinates and flags
p, q, r, s = 0, 0, 0, 0
selecting_roi = False
roi_selected = False

# Display the first frame to let the user select the ROI
cv2.imshow('Select ROI', frame)
cv2.setMouseCallback('Select ROI', select_roi)

# Wait for the user to select the ROI
while not roi_selected:
    cv2.imshow('Select ROI', frame_copy)  # Show the frame with dynamically drawn rectangle
    cv2.waitKey(10)

# Release the 'Select ROI' window
cv2.destroyWindow('Select ROI')

# Set the region for the tracking window based on the selected ROI
track_window = (p, q, r, s)

# Create the region of interest
r_o_i = frame[q:q + s, p:p + r]

# Convert BGR to HSV format for ROI
hsv_roi = cv2.cvtColor(r_o_i, cv2.COLOR_BGR2HSV)

# Apply mask on the HSV ROI
mask = cv2.inRange(hsv_roi, np.array((0., 61., 33.)), np.array((180., 255., 255.)))

# Get histogram for hsv channel of ROI
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# Normalize the retrieved values for ROI
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Termination criteria, either 15 iteration or by at least 2 pt
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)

while True:
    _, frame = cap.read()

    if frame is None:
        break
    
    # add gaussian blur to frame
    # frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Convert BGR to HSV format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    bp = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Applying meanshift to get the new region
    _, track_window = cv2.meanShift(bp, track_window, termination)

    # Draw track window on the frame
    x, y, w, h = track_window
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

import numpy as np
import cv2 as cv2

def select_user_rois(frame):
    # user selects the ROIs in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')
    return rois[0]

def adjust_hsv_mask(frame):
    def nothing(x):
        pass

    cv2.namedWindow('Adjust HSV Mask')
    cv2.createTrackbar('H Lower', 'Adjust HSV Mask', 0, 180, nothing)
    cv2.createTrackbar('H Upper', 'Adjust HSV Mask', 180, 180, nothing)
    cv2.createTrackbar('S Lower', 'Adjust HSV Mask', 0, 255, nothing)
    cv2.createTrackbar('S Upper', 'Adjust HSV Mask', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'Adjust HSV Mask', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'Adjust HSV Mask', 255, 255, nothing)

    while True:
        h_lower = cv2.getTrackbarPos('H Lower', 'Adjust HSV Mask')
        h_upper = cv2.getTrackbarPos('H Upper', 'Adjust HSV Mask')
        s_lower = cv2.getTrackbarPos('S Lower', 'Adjust HSV Mask')
        s_upper = cv2.getTrackbarPos('S Upper', 'Adjust HSV Mask')
        v_lower = cv2.getTrackbarPos('V Lower', 'Adjust HSV Mask')
        v_upper = cv2.getTrackbarPos('V Upper', 'Adjust HSV Mask')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (h_lower, s_lower, v_lower), (h_upper, s_upper, v_upper))
        cv2.imshow('Adjust HSV Mask', mask)

        if cv2.waitKey(10) == 27:  # Esc key to quit
            break

    cv2.destroyWindow('Adjust HSV Mask')
    lower_hsv = np.array((h_lower, s_lower, v_lower))
    upper_hsv = np.array((h_upper, s_upper, v_upper))
    return lower_hsv, upper_hsv

to_track = "./assets/singles_1.mp4" # "./assets/singles_1.mp4"
cap = cv2.VideoCapture(to_track) 

# take first frame of the video
ret, frame = cap.read()
# resize
if to_track:
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

track_window = select_user_rois(frame)
x, y, w, h = track_window 

lower_hsv, upper_hsv = adjust_hsv_mask(frame)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Set pixels in the above range to black and others to white
mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)

roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()
    if to_track:
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        cv2.imshow('dst', dst)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
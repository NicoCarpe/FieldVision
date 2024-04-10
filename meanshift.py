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

def crop_field_from_frame(frame):

    # create a mask for the tennis field
    field_mask = np.zeros_like(frame)
    field = select_points(frame)
    points = np.array([field[0], field[3], field[2], field[1]])
    cv2.fillPoly(field_mask, [points], color=(255, 255, 255))
    outside_field_mask = cv2.bitwise_not(field_mask)

    midpoint = (field[0] + field[2]) // 2
    field_color = frame[midpoint[1]-10, midpoint[0]-10]

    # Fill the frame outside the field with black
    frame[outside_field_mask == 255] = 0
    indices = np.all(frame == [0, 0, 0], axis=-1)
    frame[indices] = field_color

    return frame, indices, field_color

def transform_and_draw_points_on_court(players, H, tennis_court):
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    radar = tennis_court.copy()
    for x, y in new_points[0]:
        cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('Radar', radar)

    return radar

def calc_histogram_rois(frame, rois, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hists = []
    for roi in rois:
        # extract the ROI coordinates
        x, y, w, h = roi  

        # use ROI coordinates to crop the relevant region from the HSV image
        hsv_roi = hsv[y:y+h, x:x+w]

        # apply mask with all h values and user-defined s and v thresholds
        mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)

        # construct a histogram of hue and saturation values and normalize it
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        print(roi_hist.shape)

        roi_hists.append(roi_hist)

    return roi_hists


if __name__ == "__main__":
    # Read video
    cap = cv2.VideoCapture("./assets/singles_1.mp4")

    # Retrieve the very first frame from the video
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

    # Crop the field from the frame
    frame, indices, field_color = crop_field_from_frame(frame)


    # Variables to store ROI coordinates and flags
    track_windows = select_user_rois(frame)

    if track_windows is None:
        print("No ROI selected. Exiting.")
        exit()
        

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))

    pts1 = select_points(frame.copy())
    pts2 = select_points(tennis_court.copy())

    H = estimate_homography(pts1, pts2)

    # Termination criteria, either 15 iteration or by at least 2 pt
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    backSub.apply(frame)

    # Initialize the histogram
    roi_hists = calc_histogram_rois(frame, track_windows, np.array([0, 0, 0]), np.array([180, 255, 255]))

    players = [[0,0], [0,0]]
    image_idx = 0
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        if frame is None:
            break
        frame[indices] = field_color
        
        # Apply background subtraction
        field_mask = backSub.apply(frame)
        field_mask = cv2.GaussianBlur(field_mask, (5, 5), 0)
        _, field_mask = cv2.threshold(field_mask, 230, 255, 0)
        field_mask = cv2.dilate(field_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) # dilate

        # Apply mask to the frame
        field = cv2.bitwise_and(frame, frame, mask=field_mask)
        # cv2.imwrite('field_mask.png', field)

        # Convert BGR to HSV format COLOR_BGR2HSV
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
        dsts = []
        for i, roi_hist in enumerate(roi_hists):
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dsts.append(dst)

        # debug
        show = field # cv2.resize(field, (frame.shape[1]//2, frame.shape[0]//2))
        cv2.imshow('dst', dst)
        # save the frame
        # cv2.imwrite(f'./result/frame_{image_idx}.png', show)
        # image_idx += 1

        # Applying meanshift to get the new region
        for i, track_window in enumerate(track_windows):
            print(track_window)
            _, track_windows[i] = cv2.meanShift(field_mask, track_window, termination) # meanshift

            # Draw track window on the frame
            x, y, w, h = track_windows[i]
            vid = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            players[i] = [x + w//2, y +h]

        # Show results
        cv2.imshow('Tracker', vid)

        warped_image = apply_homography(vid, H)
        transform_and_draw_points_on_court(players, H, tennis_court)

        k = cv2.waitKey(10)
        if k == ord('q') or k == 27:  # Quit on 'q' or Esc key
            break

    cap.release()

    cv2.destroyAllWindows()
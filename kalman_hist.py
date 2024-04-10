import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography


def calc_histogram_rois(frame, rois):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hists = []
    for roi in rois:
        # extract the ROI coordinates
        x, y, w, h = roi  

        # use ROI coordinates to crop the relevant region from the HSV image
        hsv_roi = hsv[y:y+h, x:x+w]

        # apply mask with all h values and user-defined s and v thresholds
        mask = cv2.inRange(hsv_roi,
                           np.array((0., 0., 0.)),
                           np.array((180., 256., 256.)))

        # construct a histogram of hue and saturation values and normalize it
        roi_hist = cv2.calcHist([hsv_roi],
                                [0, 1],
                                mask,
                                [30, 40],
                                [0, 180, 0, 256])
        
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        roi_hists.append(roi_hist)

    return roi_hists


def select_user_rois(frame):
    # user selects the ROIs in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')
    return rois


def initialize_kalman_filters(rois, dt):
    kalman_filters = []
    for roi in rois:
        x, y, w, h = roi

        # Initial position and velocity
        initial_x = x + w / 2
        initial_y = y + h / 2
        initial_vx = 0
        initial_vy = 0

        # Initialize Kalman filter with 4 state variables (x, y, vx, vy) and 2 measurements (x, y)
        kf = cv2.KalmanFilter(4, 2)

        # State Transition Matrix (F)
        kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                        [0, 1, 0, dt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

        # Measurement Matrix (H)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)

        # Process Noise Covariance (Q)
        kf.processNoiseCov = np.array([[0.002, 0.02, 0, 0],
                                       [0.02, 0.002, 0, 0],
                                       [0, 0, 0.002, 0.02],
                                       [0, 0, 0.02, 0.002]], np.float32)

        # Measurement Noise Covariance (R)
        kf.measurementNoiseCov = np.array([[0.1, 0],
                                           [0, 0.1]], np.float32)

        # Error Covariance Matrix (P)
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial State (x)
        kf.statePost = np.array([initial_x, initial_y, initial_vx, initial_vy], np.float32)

        kalman_filters.append(kf)

    return kalman_filters


def combine_backprojections_in_grid(bp_images, frame):
    # Determine the maximum width and height of the backprojection images
    max_width = max(bp.shape[1] for bp in bp_images)
    max_height = max(bp.shape[0] for bp in bp_images)
    
    # Calculate the number of images per row and per column to fit them in a grid
    num_images = len(bp_images)
    num_columns = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_columns))
    
    # Create a blank canvas to place the backprojection images
    grid_height = max_height * num_rows
    grid_width = max_width * num_columns
    combined_bp = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Position each backprojection image in the grid
    for i, bp in enumerate(bp_images):
        row = i // num_columns
        col = i % num_columns
        start_y = row * max_height
        start_x = col * max_width
        combined_bp[start_y:start_y + bp.shape[0], start_x:start_x + bp.shape[1]] = bp

    # Resize the combined backprojections to fit the original frame size, if necessary
    if combined_bp.shape[0] > frame.shape[0] or combined_bp.shape[1] > frame.shape[1]:
        combined_bp = cv2.resize(combined_bp, (frame.shape[1], frame.shape[0]))
    
    return combined_bp


def tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, roi_hists, dt):
    players = []
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    bp_images = []

    for i, roi in enumerate(rois):
        kf = kalman_filters[i]
        roi_hist = roi_hists[i]

        # Prediction using Kalman filter
        prediction = kf.predict()
        
        # Update ROI based on prediction for Meanshift initialization
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        roi = (pred_x - roi[2] // 2, pred_y - roi[3] // 2, roi[2], roi[3])

        img_bp = cv2.calcBackProject([img_hsv], [0, 1], roi_hist, [0, 180, 0, 255], 1)

       # Apply Meanshift using updated ROI
        ret, new_roi = cv2.meanShift(img_bp, roi, termination)
        rois[i] = new_roi
        x, y, w, h = new_roi
        center = np.array([x + (w / 2), y + (h / 2)], np.float32)

        # store player positions (we want the marker to be at the bottom of their feet)
        players.append((int(center[0]), int(center[1]) + (h / 2)))

        measured_x = x + w / 2
        measured_y = y + h / 2
        measurement = np.array([measured_x, measured_y], np.float32)

        # Correct the Kalman filter with the new measurement
        kf.correct(measurement.reshape(-1, 1))


        # draw new_roi on image - in BLUE
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)      
        
        # draw predicton on image - in GREEN
        frame = cv2.rectangle(frame,
                             (int(prediction[0][0] - (0.5 * w)),
                              int(prediction[1][0] - (0.5 * h))),
                             (int(prediction[0][0] + (0.5 * w)),
                              int(prediction[1][0] + (0.5 * h))),
                             (0, 255, 0),
                              2)

        img_bp = cv2.calcBackProject([img_hsv], [0, 1], roi_hists[i], [0, 180, 0, 255], 1)
        bp_images.append(img_bp)

    # After the loop, combine the backprojections in a grid
    combined_bp = combine_backprojections_in_grid(bp_images, frame)

    # Show the combined backprojections
    cv2.imshow("Combined Backprojections", combined_bp)
    cv2.imshow("Tracking Window", frame)
    
    return rois, players



def transform_and_draw_points_on_court(players, H, tennis_court):
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    radar = tennis_court.copy()
    for x, y in new_points[0]:
        cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('Radar', radar)


def main():
    fps = 30.0
    dt = 1 / fps
    cap = cv2.VideoCapture("./assets/vb_clip.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame. Exiting.")
        exit()
    
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
    
    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)
    
    # calculate histograms for ROI
    roi_hists = calc_histogram_rois(frame, rois)

    # possibly look at interatively increasing the search window if there is poor matching
    # or implementing a pyramidal update on top
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 2)
    
    kalman_filters = initialize_kalman_filters(rois, fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        
        rois, players = tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, roi_hists, dt)
        
        transform_and_draw_points_on_court(players, H, tennis_court)
        
        out.write(frame)
        
        if cv2.waitKey(10) == 27:  # Esc key to quit
            break
        
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

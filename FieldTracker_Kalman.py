import cv2
import numpy as np
from DLT import select_points, estimate_homography, apply_homography


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
        roi_hist = cv2.calcHist([hsv_roi],
                                [0, 1],
                                mask,
                                [180, 256],
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

        # Extract the center of the ROI as the initial position
        initial_x = x + w / 2  
        initial_y = y + h / 2  

        # Assuming initial velocities (vx, vy) and accelerations (ax, ay) are zero
        initial_vx = 0
        initial_vy = 0
        initial_ax = 0
        initial_ay = 0

        # Initialize the Kalman filter
        kf = cv2.KalmanFilter(6, 4)  # 6 state variables (x, y, vx, vy, ax, ay), 4 measurements (x, y, vx, vy)

        # State transition matrix (F)
        kf.transitionMatrix = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                                        [0, 1, 0, dt, 0, 0.5*dt**2],
                                        [0, 0, 1, 0, dt, 0],
                                        [0, 0, 0, 1, 0, dt],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]], np.float32)

        # Measurement matrix (H)
        kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0]], np.float32)

        # Process noise covariance (Q)
        # Adjust these values based on the expected level of noise in your system
        kf.processNoiseCov = np.array([[0.1, 0, 0, 0, 0, 0],
                                       [0, 0.1, 0, 0, 0, 0],
                                       [0, 0, 0.1, 0, 0, 0],
                                       [0, 0, 0, 0.1, 0, 0],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]], np.float32)

        # Measurement noise covariance (R)
        kf.measurementNoiseCov = np.array([[0.1, 0, 0, 0],
                                           [0, 0.1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)  
        
        # Initial state estimation error covariance (P)
        kf.errorCovPost = np.eye(6, dtype=np.float32) 
        
        # Initialize the state vector (x)
        kf.statePost = np.array([initial_x, initial_y, initial_vx, initial_vy, initial_ax, initial_ay], np.float32)
        
        kalman_filters.append(kf)

    return kalman_filters


def calculate_velocity(current_position, last_position, dt):
    dx = current_position[0] - last_position[0]
    dy = current_position[1] - last_position[1]
    vx = dx / dt
    vy = dy / dt
    return vx, vy


def tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, roi_hists, last_positions, dt):
    players = []
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    # to store the updated last positions
    new_last_positions = []  

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
        new_last_positions.append(center)

        # Calculate velocity for Kalman filter measurement update
        if last_positions and len(last_positions) > i:
            vx, vy = calculate_velocity(center, last_positions[i], dt)
        else:
            vx, vy = 0, 0  # No velocity if it's the first frame

        measurement = np.array([center[0], center[1], vx, vy], np.float32)
        
        # Correct the Kalman filter with the new measurement
        kf.correct(measurement.reshape(-1, 1))


        # draw new_roi on image - in BLUE
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)      

        # debuging
        img_bp_show = cv2.rectangle(img_bp,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)
        cv2.imshow("Backprojection", img_bp_show)

        # draw predicton on image - in GREEN
        frame = cv2.rectangle(frame,
                             (int(prediction[0][0] - (0.5 * w)),
                              int(prediction[1][0] - (0.5 * h))),
                             (int(prediction[0][0] + (0.5 * w)),
                              int(prediction[1][0] + (0.5 * h))),
                             (0, 255, 0),
                              2)
    
    cv2.imshow("Tracking Window", frame)
    
    return rois, players, cv2.cvtColor(img_bp_show, cv2.COLOR_GRAY2BGR)


def transform_and_draw_points_on_court(players, H, tennis_court):
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    radar = tennis_court.copy()
    for x, y in new_points[0]:
        cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('Radar', radar)

    return radar

def main():
    fps = 60
    dt = 1 / fps
    cap = cv2.VideoCapture("./assets/doubles_clip.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame. Exiting.")
        exit()

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # create a mask for the tennis field
    field_mask = np.zeros_like(frame)
    field = select_points(frame)
    points = np.array([field[0], field[3], field[2], field[1]])
    cv2.fillPoly(field_mask, [points], color=(255, 255, 255))
    outside_field_mask = cv2.bitwise_not(field_mask)
    # # Fill the frame outside the field with black

    midpoint = (field[0] + field[2]) // 2
    field_color = frame[midpoint[1]-10, midpoint[0]-10]
    frame[outside_field_mask == 255] = 0
    indices = np.all(frame == [0, 0, 0], axis=-1)
    frame[indices] = field_color

    # interactive HSV mask adjustment
    lower_hsv, upper_hsv = adjust_hsv_mask(frame)

    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)
    
    # possibly look at interatively increasing the search window if there is poor matching
    # or implementing a pyramidal update on top
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_kalman.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))


    kalman_filters = initialize_kalman_filters(rois, fps)
    
    last_positions = [] 
    # calculate histograms for ROI
    roi_hists = calc_histogram_rois(frame, rois, lower_hsv, upper_hsv)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame[indices] = field_color

        rois, players, debug_show = tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, roi_hists, last_positions, dt)
        
        radar = transform_and_draw_points_on_court(players, H, tennis_court)
        
        # minimize the radar to fit the video
        radar = cv2.resize(radar, (radar.shape[1]//2, radar.shape[0]//2))
        # overlay the radar on the video
        frame[0:radar.shape[0], 0:radar.shape[1]] = radar

        # minimize the debug screen to fit the video
        debug_show = cv2.resize(debug_show, (debug_show.shape[1]//5, debug_show.shape[0]//5))
        # overlay the debug screen on the video
        frame[0:debug_show.shape[0], debug_show.shape[1]:2*debug_show.shape[1]] = debug_show

        out.write(frame)

        if cv2.waitKey(10) == 27:  # Esc key to quit
            break
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

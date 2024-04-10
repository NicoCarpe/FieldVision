import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography


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


def tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub):
    players = []
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

     # Apply background subtraction
    field_mask = back_sub.apply(frame)
    field_mask = cv2.GaussianBlur(field_mask, (5, 5), 0)
    _, field_mask = cv2.threshold(field_mask, 230, 255, 0)
    field_mask = cv2.dilate(field_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) # dilate

    # Apply mask to the frame
    field = cv2.bitwise_and(frame, frame, mask=field_mask)

    for i, roi in enumerate(rois):
        kf = kalman_filters[i]

        # Prediction using Kalman filter
        prediction = kf.predict()
        
        # Update ROI based on prediction for Meanshift initialization
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        roi = (pred_x - roi[2] // 2, pred_y - roi[3] // 2, roi[2], roi[3])

       # Apply Meanshift using updated ROI
        ret, new_roi = cv2.meanShift(field_mask, roi, termination)
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

    
    # show field mask and tracking window
    cv2.imshow("background subtraction", field)
    cv2.imshow("Tracking Window", frame)
    
    return rois, players



def transform_and_draw_points_on_court(players, H, tennis_court):
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    radar = tennis_court.copy()
    for x, y in new_points[0]:
        cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('Radar', radar)


def main():
    fps = 60
    dt = 1 #/ fps
    cap = cv2.VideoCapture("./assets/doubles_clip.mp4")
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame. Exiting.")
        exit()

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)

    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()
    back_sub.apply(frame)
    

    # possibly look at interatively increasing the search window if there is poor matching
    # or implementing a pyramidal update on top
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 2)
    
    kalman_filters = initialize_kalman_filters(rois, fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        rois, players = tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub)
        
        transform_and_draw_points_on_court(players, H, tennis_court)
        
        if cv2.waitKey(10) == 27:  # Esc key to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

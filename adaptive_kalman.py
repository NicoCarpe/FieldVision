import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography
from utils.KalmanFilter import AdaptiveKalmanFilter 


def select_user_rois(frame):
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')
    return rois


def initialize_kalman_filters(rois, dt):
    kalman_filters = []
    for roi in rois:
        x, y, w, h = roi

        # Define matrices for the Kalman Filter initialization
        state_transition_matrix = np.array([[1, 0, dt, 0], 
                                            [0, 1, 0, dt], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]], np.float32)
        
        measurement_matrix = np.array([[1, 0, 0, 0], 
                                       [0, 1, 0, 0]], np.float32)
        
        process_noise_cov = np.array([[0.005, 0, 0, 0], 
                                      [0, 0.005, 0, 0], 
                                      [0, 0, 0.05, 0], 
                                      [0, 0, 0, 0.05]], np.float32)
        
        measurement_noise_cov = np.array([[0.2, 0], 
                                          [0, 0.2]], np.float32)
        
        error_cov_post = np.eye(4, dtype=np.float32)

        initial_state = np.array([x + w / 2, y + h / 2, 0, 0], np.float32)

        # Initialize an AdaptiveKalmanFilter
        kf = AdaptiveKalmanFilter(dt, 
                                  state_transition_matrix, 
                                  measurement_matrix, 
                                  process_noise_cov, 
                                  measurement_noise_cov, 
                                  error_cov_post, 
                                  initial_state)
        
        kalman_filters.append(kf)

    return kalman_filters


def roi_overlap(roi1, roi2):
    x_left = max(roi1[0], roi2[0])
    y_top = max(roi1[1], roi2[1])
    x_right = min(roi1[0] + roi1[2], roi2[0] + roi2[2])
    y_bottom = min(roi1[1] + roi1[3], roi2[1] + roi2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    roi1_area = roi1[2] * roi1[3]
    return overlap_area / roi1_area


def calculate_alpha(field_mask, roi):
    x, y, w, h = roi
    roi_mask = field_mask[y:y+h, x:x+w]
    filled_pixels = np.count_nonzero(roi_mask)
    total_pixels = w * h
    return filled_pixels / total_pixels if total_pixels > 0 else 0


def constrain_to_frame(roi, frame_dimensions):
    x, y, w, h = roi
    frame_width, frame_height = frame_dimensions

    # Ensure the ROI does not go beyond the frame boundary
    x = max(0, min(x, frame_width - w))
    y = max(0, min(y, frame_height - h))

    return (x, y, w, h)


def tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub):
    players = []
    frame_dimensions = (frame.shape[1], frame.shape[0])
    
    # Apply background subtraction
    field_mask = back_sub.apply(frame)
    field_mask = cv2.GaussianBlur(field_mask, (5, 5), 0)
    _, field_mask = cv2.threshold(field_mask, 230, 255, 0)
    field_mask = cv2.dilate(field_mask, None, iterations=2) 

    # Apply mask to the frame
    #field = cv2.bitwise_and(frame, frame, mask=field_mask)

    for i, roi in enumerate(rois):
        kf = kalman_filters[i]

        # Prediction using Kalman filter
        prediction = kf.predict()

        # Update ROI based on prediction for Meanshift initialization
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        updated_roi = (pred_x - roi[2] // 2, pred_y - roi[3] // 2, roi[2], roi[3])

        # Constrain the updated ROI before applying MeanShift
        updated_roi = constrain_to_frame(updated_roi, frame_dimensions)

        # Apply MeanShift using updated ROI
        ret, new_roi = cv2.meanShift(field_mask, updated_roi, termination)
        rois[i] = new_roi
        x, y, w, h = new_roi
        center = np.array([x + (w / 2), y + (h / 2)], np.float32)
        
        # store player positions (we want the marker to be at the bottom of their feet)
        players.append((int(center[0]), int(center[1]) + (h / 2)))

        measured_x = x + w / 2
        measured_y = y + h / 2
        measurement = np.array([measured_x, measured_y], np.float32)  
        
        occluded = any(roi_overlap(new_roi, other_roi) > 0.4 for j, other_roi in enumerate(rois) if i != j)
        alpha = calculate_alpha(field_mask, new_roi)
        
        kf.adjust_for_occlusion(occluded)
        kf.dynamic_adjustment(alpha)

        # Update the Kalman filter with the new measurement
        kf.update(measurement)

        # draw new_roi on image - in BLUE
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)      
        
        # draw predicton on image - in GREEN
        frame = cv2.rectangle(  frame, 
                                (int(prediction[0] - (0.5 * w)), 
                                int(prediction[1] - (0.5 * h))), 
                                (int(prediction[0] + (0.5 * w)), 
                                int(prediction[1] + (0.5 * h))), 
                                (0, 255, 0), 
                                2)
        
    # show field mask and tracking window
    cv2.imshow("background subtraction", field_mask)
    cv2.imshow("Tracking Window", frame)
    
    return rois, players


def transform_and_draw_points_on_court(players, H, tennis_court):
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    radar = tennis_court.copy()
    for x, y in new_points[0]:
        cv2.circle(radar, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow('Radar', radar)

    return radar

def main():
    fps = 30.0
    dt = 1 / fps

    cap = cv2.VideoCapture("assets/doubles_clip3.mp4") 
    
    _, frame = cap.read()
    print(frame.shape)
    width, height = 960, 540
    frame = cv2.resize(frame, (width, height))

    rois = select_user_rois(frame)

    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        return

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)

    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorKNN()
    back_sub.apply(frame)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 2)
    
    kalman_filters = initialize_kalman_filters(rois, dt)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputs/output.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        
        rois, players = tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub)
        
        radar = transform_and_draw_points_on_court(players, H, tennis_court)
        
        if cv2.waitKey(10) == 27:  # Esc key to quit
            break
        
        # minimize the radar to fit the video
        radar = cv2.resize(radar, (radar.shape[1]//2, radar.shape[0]//2))
        # overlay the radar on the video
        frame[0:radar.shape[0], 0:radar.shape[1]] = radar

        out.write(frame)
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

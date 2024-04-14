import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography
from utils.KalmanFilter import AdaptiveKalmanFilter 
from utils.homography import get_line


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
        
        process_noise_cov = np.array([[0.01, 0, 0, 0], 
                                      [0, 0.01, 0, 0], 
                                      [0, 0, 0.05, 0], 
                                      [0, 0, 0, 0.05]], np.float32)
        
        measurement_noise_cov = np.array([[0.1, 0], 
                                          [0, 0.1]], np.float32)
        
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


def tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub, mid_point, team_1, team_2):
    players = []
    frame_dimensions = (frame.shape[1], frame.shape[0])
    
    # Apply background subtraction
    field_mask = back_sub.apply(frame)
    field_mask = cv2.GaussianBlur(field_mask, (5, 5), 0)
    _, field_mask = cv2.threshold(field_mask, 230, 255, 0)
    field_mask = cv2.dilate(field_mask, None, iterations=2) 

    for i, roi in enumerate(rois):
        prev_x, prev_y, prev_w, prev_h = roi
        if i == 0 or i == 1:
            prev_y -= 5
        else:
            prev_y += 5

        prev_center = np.array([prev_x + (prev_w / 2), prev_y + (prev_h / 2)], np.float32)

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
        player_location = (int(center[0]), int(center[1]) + (h / 2))
        player_prev_location = (int(prev_center[0]), int(prev_center[1]) + (prev_h / 2))
        
        
        if mid_point is not None:
            if i in team_1 and player_location[1] < mid_point[1]:
                players.append(player_location)
            elif i in team_2 and player_location[1] > mid_point[1]:
                players.append(player_location)
            else:
                rois[i] = (prev_x, prev_y, prev_w, prev_h)
                new_roi = roi
                x, y, w, h = new_roi
                players.append(player_prev_location)

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
        
        # For Debugging: draw net line
        # if mid_point is not None:
        #     length = 300
        #     cv2.line(frame, (int(mid_point[0] - length), int(mid_point[1])), (int(mid_point[0] + length), int(mid_point[1])), (0, 0, 255), 2)
        
    # show field mask and tracking window
    cv2.imshow("background subtraction", field_mask)
    cv2.imshow("Tracking Window", frame)
    
    return rois, players


def transform_and_draw_points_on_court(players, H, tennis_court):
    # Define a list of distinct colors for the players
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, player in enumerate(players):
        # Apply the homography to transform the player's position to the court's perspective
        transformed_pos = cv2.perspectiveTransform(np.array([[player]], dtype=np.float32), H)
        # Choose a color for the player based on their index
        color = colors[i % len(colors)]
        # Draw the player's current position on the tennis court image
        cv2.circle(tennis_court, (int(transformed_pos[0][0][0]), int(transformed_pos[0][0][1])), 2, color, -1)
    
    cv2.imshow('Radar', tennis_court)

    return tennis_court

def main():
    fps = 30.0
    dt = 1 / fps

    cap = cv2.VideoCapture("assets/doubles_clip3.mp4") 
    
    _, frame = cap.read()
    width, height = 960, 540
    frame = cv2.resize(frame, (width, height))

    rois = select_user_rois(frame)

    # Decide teams based on the number of ROIs selected
    if len(rois) == 4:
        team_1 = [0, 1]
        team_2 = [2, 3]
    elif len(rois) == 2:
        team_1 = [0]
        team_2 = [1]
    else:
        team_1, team_2 = [], []
    
    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    scale_factor = height / tennis_court.shape[0]
    new_width = int(tennis_court.shape[1] * scale_factor)
    tennis_court = cv2.resize(tennis_court, (new_width, height))

    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)
    
    homo_pts = np.concatenate((src_pts, np.ones([len(src_pts), 1])), axis=1) 
    
    l1 = get_line(homo_pts[0], homo_pts[2])
    l2 = get_line(homo_pts[1], homo_pts[3])
    
    mid_point = np.cross(l1, l2)
    mid_point = (mid_point / mid_point[2]).astype(np.int32)


    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorKNN()
    back_sub.apply(frame)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 2)
    
    kalman_filters = initialize_kalman_filters(rois, dt)


    # initialize our predictions with the chosen rois
    prev_predicts = rois

    
    # initialize our predictions with the chosen rois
    prev_predicts = rois

    # Prepare for combined video output
    output_width = width + tennis_court.shape[1]  # combine the frame and court widths
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputs/output.mp4', fourcc, fps, (output_width, height))
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        
        rois, players = tracking_with_meanshift_and_kalman(rois, frame, termination, kalman_filters, back_sub, mid_point, team_1, team_2)
        
        transform_and_draw_points_on_court(players, H, tennis_court)
        
        # Combine frame and tennis_court_resized for side-by-side output
        combined_frame = np.hstack((frame, tennis_court))
        
        out.write(combined_frame)

        if cv2.waitKey(10) == 27:  # Esc key to quit
            break
    
    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
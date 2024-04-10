import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography, apply_homography
from utils.MeanShiftTracker import MeanShiftTracker  # Ensure this import points to your implemented class

def select_user_rois(frame):
    # User selects the ROIs in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')
    return rois

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

def tracking_with_meanshift(trackers, frame):
    rois = []
    bp_imgs = []
    players = []
    for tracker in trackers:
        # Update tracker with the current frame
        tracker.mean_shift(frame)
        x, y, w, h = tracker.roi
        
        # Draw tracking ROI on the frame
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        
        # Extract player position
        player = np.array([x + w / 2, y + h], np.float32)
        players.append(player)

        bp_imgs.append(tracker.likelihood_img)

    combined_bp = combine_backprojections_in_grid(bp_imgs, frame)

    cv2.imshow("Combined Backprojections", combined_bp)
    cv2.imshow("Tracking Window", frame)
    return players

def transform_and_draw_points_on_court(players, H, tennis_court):
    # Transform and draw points on the tennis court
    new_points = cv2.perspectiveTransform(np.array([players], dtype=np.float32), H)
    for x, y in new_points[0]:
        cv2.circle(tennis_court, (int(x), int(y)), 1, (0, 0, 255), -1)
    cv2.imshow('Radar', tennis_court)

def main():
    cap = cv2.VideoCapture("./assets/doubles_clip2.mp4")  # Adjust path as necessary
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame. Exiting.")
        exit()

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    # Initialize trackers for each ROI
    trackers = [MeanShiftTracker(frame, roi) for roi in rois]

    # Setup for court and homography
    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))

    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        players = tracking_with_meanshift(trackers, frame)
        transform_and_draw_points_on_court(players, H, tennis_court)

        if cv2.waitKey(10) == 27:  # Esc key to stop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

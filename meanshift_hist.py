import cv2
import numpy as np
from utils.DLT import select_points, estimate_homography, apply_homography


def calc_histogram_rois(frame, rois, foreground_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hists = []
    for roi in rois:
        x, y, w, h = roi  

        # Use ROI coordinates to crop the relevant region from the HSV image and the foreground mask
        hsv_roi = hsv[y:y+h, x:x+w]
        mask_roi = foreground_mask[y:y+h, x:x+w]

        # Construct a histogram of hue and saturation values using the foreground mask to focus on the moving object
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask_roi, [30, 40], [0, 180, 0, 256])
        
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        roi_hists.append(roi_hist)

    return roi_hists


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


def select_user_rois(frame):
    # user selects the ROIs in the first frame
    rois = cv2.selectROIs('Select ROIs', frame, fromCenter=False, showCrosshair=False)
    cv2.destroyWindow('Select ROIs')
    return rois


def tracking_with_meanshift(rois, frame, termination, roi_hists):
    players = []
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bp_images = []

    for i, roi in enumerate(rois):
        roi_hist = roi_hists[i]
        # Backproject the histogram to find pixels with similar húes
        img_bp = cv2.calcBackProject(
                    [img_hsv], 
                    [0, 1], 
                    roi_hist, 
                    [0, 180, 0, 255],
                    1)

        # Apply MeanShift
        ret, new_roi = cv2.meanShift(img_bp, roi, termination)
        rois[i] = new_roi
        x, y, w, h = new_roi

        # draw new_roi on image - in BLUE
        frame = cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)
        
        # extract player position
        player = np.array([x + w / 2, y + h], np.float32)
        players.append(player)
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

    return radar


def main():
    fps = 30.0
    dt = 1 / fps
    cap = cv2.VideoCapture("./assets/doubles_clip2.mp4") 
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame. Exiting.")
        exit()

      
    # Initialize the KNN background subtractor with a warm-up phase
    background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    
    # Warm-up phase
    warm_up_frames = 30
    for _ in range(warm_up_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame during warm-up. Exiting.")
            return
        background_subtractor.apply(frame)
    
    # Restart the video for the actual tracking
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read the first frame to start tracking
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab initial frame for tracking. Exiting.")
        return
    
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Apply the background subtractor to get the foreground mask
    foreground_mask = background_subtractor.apply(frame)

    # Display the foreground mask for debugging
    cv2.imshow('Foreground Mask', foreground_mask)
    
    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    # Calculate histograms for ROIs using the foreground mask
    roi_hists = calc_histogram_rois(frame, rois, foreground_mask)
    
    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)
    
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame.shape[1], frame.shape[0]))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        
        rois, players = tracking_with_meanshift(rois, frame, termination, roi_hists)
        
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
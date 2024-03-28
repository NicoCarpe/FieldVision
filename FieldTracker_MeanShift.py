import cv2
import numpy as np
from DLT import select_points, estimate_homography, apply_homography


def adjust_hsv_mask(frame):
    def nothing(x):
        pass

    cv2.namedWindow('Adjust HSV Mask')
    cv2.createTrackbar('S Lower', 'Adjust HSV Mask', 0, 255, nothing)
    cv2.createTrackbar('S Upper', 'Adjust HSV Mask', 255, 255, nothing)
    cv2.createTrackbar('V Lower', 'Adjust HSV Mask', 0, 255, nothing)
    cv2.createTrackbar('V Upper', 'Adjust HSV Mask', 255, 255, nothing)

    while True:
        s_lower = cv2.getTrackbarPos('S Lower', 'Adjust HSV Mask')
        s_upper = cv2.getTrackbarPos('S Upper', 'Adjust HSV Mask')
        v_lower = cv2.getTrackbarPos('V Lower', 'Adjust HSV Mask')
        v_upper = cv2.getTrackbarPos('V Upper', 'Adjust HSV Mask')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, s_lower, v_lower), (180, s_upper, v_upper))
        cv2.imshow('Adjust HSV Mask', mask)

        if cv2.waitKey(10) == 27:  # Esc key to quit
            break

    cv2.destroyWindow('Adjust HSV Mask')
    return s_lower, s_upper, v_lower, v_upper


def calc_histogram_rois(frame, rois, s_lower, s_upper, v_lower, v_upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hists = []
    for roi in rois:
        # extract the ROI coordinates
        x, y, w, h = roi  

        # use ROI coordinates to crop the relevant region from the HSV image
        hsv_roi = hsv[y:y+h, x:x+w]

        # apply mask with all h values and user-defined s and v thresholds
        mask = cv2.inRange(hsv_roi,
                           np.array((0., float(s_lower), float(v_lower))),
                           np.array((180., float(s_upper), float(v_upper))))

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

def tracking_with_meanshift(rois, frame, termination, roi_hists):
    players = []
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
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
        
        # debuging
        img_bp_show = cv2.rectangle(img_bp,
                              (x, y),
                              (x + w, y + h),
                              (255, 0, 0),
                              2)
        # show backprojection image
        cv2.imshow("Backprojection", img_bp_show)
        
        # extract player position
        player = np.array([x + w / 2, y + h], np.float32)
        players.append(player)

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
    cap = cv2.VideoCapture("./assets/singles_1.mp4") # ""
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


    midpoint = (field[0] + field[2]) // 2
    field_color = frame[midpoint[1]-10, midpoint[0]-10]
    # Fill the frame outside the field with black
    frame[outside_field_mask == 255] = 0
    indices = np.all(frame == [0, 0, 0], axis=-1)
    frame[indices] = field_color

    # interactive HSV mask adjustment
    s_lower, s_upper, v_lower, v_upper = adjust_hsv_mask(frame)

    rois = select_user_rois(frame)
    if len(rois) == 0:
        print("No ROI selected. Exiting.")
        exit()

    tennis_court = cv2.imread('./assets/tennis_court_background.png')
    tennis_court = cv2.resize(tennis_court, (tennis_court.shape[1] // 6, tennis_court.shape[0] // 6))
    
    src_pts = select_points(frame, 4)
    dst_pts = select_points(tennis_court.copy(), 4)
    H = estimate_homography(src_pts, dst_pts)
    
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 5)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    out_debug = cv2.VideoWriter('output_debug.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    out_radar = cv2.VideoWriter('output_radar.mp4', fourcc, 30.0, (tennis_court.shape[1], tennis_court.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame[indices] = field_color

        # calculate histograms for ROI
        roi_hists = calc_histogram_rois(frame, rois, s_lower, s_upper, v_lower, v_upper)
    
        rois, players, debug_show = tracking_with_meanshift(rois, frame, termination, roi_hists)
        
        radar = transform_and_draw_points_on_court(players, H, tennis_court)
        
        if cv2.waitKey(10) == 27:  # Esc key to quit
            break

        # minimize the radar to fit the video
        radar = cv2.resize(radar, (radar.shape[1]//2, radar.shape[0]//2))
        # overlay the radar on the video
        frame[0:radar.shape[0], 0:radar.shape[1]] = radar

        # minimize the debug screen to fit the video
        debug_show = cv2.resize(debug_show, (debug_show.shape[1]//5, debug_show.shape[0]//5))
        # overlay the debug screen on the video
        frame[0:debug_show.shape[0], debug_show.shape[1]:2*debug_show.shape[1]] = debug_show

        out.write(frame)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# FieldVision

## Overview

FieldVision is a player-tracking system tailored for tennis analytics. It monitors player movements, while handling occlusions, and maps player positions onto a top-down view of a tennis court. The system employs the Meanshift algorithm for initial player detection, an Adaptive Kalman Filter for dynamic adjustments during tracking, and homography transformations for mapping onto our top-down view.

<p align="center">
  <img src="https://github.com/NicoCarpe/FieldVision/assets/output_3.gif" alt="animated" />
</p>


### Key Features

- **Player Tracking with Meanshift:** Utilizes the Meanshift algorithm for robust, feature-based tracking, focusing reliably on players by exploiting the color distribution within video frames.
- **Adaptive Occlusion Handling with Kalman Filter:** Features an Adaptive Kalman Filter that dynamically adjusts its parameters based on detection conditions, such as occlusions and variable detection confidence, to ensure consistent and reliable tracking.
- **Standardized Court Mapping via Homography:** Implements homography transformations to map the 2D player positions onto a standardized model of a tennis court, facilitating accurate and consistent analysis across different matches and camera angles.


## Implementation Details

The system starts by applying the Meanshift algorithm to detect initial player positions from video frames. These positions are refined using the Adaptive Kalman Filter, which adjusts to changes in player motion and occlusion dynamically. Finally, player coordinates are transformed to a standard tennis court using homography, allowing for precise strategic analysis.


## Sources
Comaniciu, D., Ramesh, V., & Meer, P. (2000). Real-time tracking of non-rigid objects
using mean shift. In IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), (pp. –). Hilton Head Island, SC: IEEE.

Fang, M.-Y., Chang, C.-K., Yang, N.-C., Kuo, C.-M., & Guang, S.-K. (2014). Robust
player tracking for broadcast tennis videos with adaptive kalman filtering. Journal of
Information Hiding and Multimedia Signal Processing, 5 (2), 242–262.

Hartley, R., & Zisserman, A. (2000). Multiple view geometry in computer vision. Cambridge
University Press.

Jernb ̈acker, A. (2022). Kalman filters as an enhancement to object tracking using yolov7. 
Technical Report 99999-99, KTH Royal Institute of Technology. TRITA – SCI-GRU 2022:345.

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal
of basic Engineering, 82 (1), 35–45.

Salhi, A., & Jammoussi, A. Y. (2012). Object tracking system using camshift, meanshift
and kalman filter. World Academy of Science, Engineering and Technology, International
Journal of Electronics and Communication Engineering, 6 (4), 421–426.

Tennis TV (2020). 10 minutes of incredible doubles tennis. YouTube video.
URL https://www.youtube.com/watch?v=G2klD115vM

"""
Constants
"""

from typing import Final

#: Window name for the OpenCV display.
WINDOW_NAME: Final[str] = "Tracking"

#: Filename of the predictions CSV file.
PREDICTIONS_FILENAME: Final[str] = "predictions.csv"

#: Filename of the video output with detections drawn on it.
ANNOTATED_VIDEO_FILENAME: Final[str] = "video_with_predicitons.mp4"

#: Tile size
TILE_SIZE: Final[int] = 640

#: Crop size
CROP_SIZE: Final[int] = 512

#: Overlap between tiles (as a fraction of tile size)
OVERLAP: Final[float] = 0.1

#: Confidence threshold
CONFIDENCE_THRESHOLD: Final[float] = 0.1

#: IoU threshold
IOU_THRESHOLD: Final[float] = 0.45

#: Maximum of consecutive frames where the ball was not detected
MAX_LOST_FRAMES: Final[int] = 10

#: Number of frames to skip between detections when tracking is active (0 means no skipping)
NUM_FRAMES_TO_SKIP: Final[int] = 0

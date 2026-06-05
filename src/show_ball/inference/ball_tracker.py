"""
Module that implements a tracking pipeline using a detection model. It maintains the position of
the detected object across frames and handles cases where the object is lost for a few frames,
attempting to reacquire it using the full frame detection.
"""

from enum import StrEnum

import numpy as np
import numpy.typing as npt

from show_ball.inference.ball_detector import BallDetector, Detection


def center_position(det: Detection) -> tuple[int, int]:
    """
    Gets the center position of a detection.

    Args:
        det: Detection.

    Returns:
        Tuple of x and y coordinates of the center position.
    """

    x = (det.x1 + det.x2) / 2
    y = (det.y1 + det.y2) / 2

    return int(x), int(y)


class TrackerMode(StrEnum):
    """
    Tracker mode
    """

    TRACK = "track"
    FULL = "full"
    LOST = "lost"


class BallTracker:
    """
    Class responsible for the tracking pipeline.
    """

    def __init__(
        self,
        predictor: BallDetector,
        crop_size: int,
        max_lost_frames: int,
        num_frames_to_skip: int,
    ):
        """
        Args:
            predictor: TorchPredictor object.
            crop_size: Crop size for tracking.
            max_lost_frames: Maximum number of frames allowed per lost frame.
            num_frames_to_skip: Number of frames to skip between detections when tracking is
            active.
        """

        # Torch predictor for running inference.
        self._predictor: BallDetector = predictor

        # Crop size around the ball when detected.
        self._crop_size: int = crop_size

        # Maximum number of frames allowed per lost frame.
        self._max_lost_frames: int = max_lost_frames

        # Number of frames to skip between detections when tracking is active.
        self._num_frames_to_skip: int = num_frames_to_skip

        # Counter of number of lost frames
        self._lost_frames_counter: int = 0

        # Number of frames processed since the start of the video
        self._frame_counter: int = 0

        # Previous detection of the ball. Used for frame skipping when tracking is active.
        self._last_detection: Detection | None = None

    @property
    def is_tracking(self) -> bool:
        """
        Whether the tracker is currently tracking the ball (i.e. has a valid detection to track).
        """
        return self._last_detection is not None

    def _crop(self, frame: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], int, int]:
        """
        Crops a square region around the current position.

        Args:
            frame: Original frame.

        Returns:
            Cropped frame and x and y coordinates of the crop's top-left corner in the
            original frame.

        Raises:
            ValueError: If there is no previous detection to crop around.
        """

        if self._last_detection is None:
            raise ValueError("There is no previous detection.")

        prev_position_x, prev_position_y = center_position(self._last_detection)

        x0 = int(prev_position_x - self._crop_size / 2)
        y0 = int(prev_position_y - self._crop_size / 2)

        x0 = max(0, min(x0, frame.shape[1] - self._crop_size))
        y0 = max(0, min(y0, frame.shape[0] - self._crop_size))

        crop = frame[y0 : y0 + self._crop_size, x0 : x0 + self._crop_size]
        return crop, x0, y0

    def _update_detection(self, det: Detection) -> None:
        """
        Updates tracker state from a detection.

        Args:
            det: Detection.
        """

        self._lost_frames_counter = 0
        self._last_detection = det

    def _should_skip_detection(self) -> bool:
        """
        Whether the tracker should skip detection on the current frame, based on the frame
        skipping configuration and the current tracking state.
        """

        return (
            self.is_tracking
            and self._num_frames_to_skip > 0
            and self._frame_counter % (self._num_frames_to_skip + 1) != 1
        )

    def _run_crop_detection(self, frame: npt.NDArray[np.uint8]) -> Detection | None:
        """
        Runs predictor on a cropped region around the last known position to get the best detection.

        Args:
            frame: Original frame.

        Returns:
            Detection in the original frame. None if there is no prediction.
        """

        crop, x0, y0 = self._crop(frame)

        det = self._predictor.best_detection(
            crop,
            use_tiles=False,
        )

        if det is None:
            return None

        det.x1 += x0
        det.x2 += x0
        det.y1 += y0
        det.y2 += y0

        return det

    def _run_full_frame_detection(self, frame: npt.NDArray[np.uint8]) -> Detection | None:
        """
        Runs predictor on the full frame to get the best detection.

        Args:
            frame: Original frame.

        Returns:
            Detection in the original frame. None if there is no prediction.
        """

        return self._predictor.best_detection(
            frame,
            use_tiles=True,
        )

    def _should_run_full_frame_detection(self) -> bool:
        """
        Whether the tracker should reacquire the full frame to try to find the ball again, based on
        the current tracking state and the number of frames where the ball was lost.
        """

        return not self.is_tracking or self._lost_frames_counter > self._max_lost_frames

    def _mark_lost_frame(self) -> None:
        """
        Increases the number of frames where the ball was lost.
        """

        self._lost_frames_counter += 1

    def _reset_tracking(self) -> None:
        """
        Resets the tracking state, clearing the last detection and resetting the lost frames
        counter.
        """

        self._last_detection = None
        self._lost_frames_counter = 0

    def process_frame(
        self,
        frame: npt.NDArray[np.uint8],
    ) -> tuple[Detection | None, TrackerMode]:
        """
        Processes a frame.

        Returns:
            Detection and tracker mode.
        """

        self._frame_counter += 1

        if self._should_skip_detection():
            return self._last_detection, TrackerMode.TRACK

        if self.is_tracking or not self._should_run_full_frame_detection():
            det = self._run_crop_detection(frame)

            if det is not None:
                self._update_detection(det)
                return det, TrackerMode.TRACK

            self._mark_lost_frame()

        if self._should_run_full_frame_detection():
            det = self._run_full_frame_detection(frame)

            if det is not None:
                self._update_detection(det)
                return det, TrackerMode.FULL

            self._reset_tracking()

        return None, TrackerMode.LOST

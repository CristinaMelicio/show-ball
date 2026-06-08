"""
Tests for ball_tracker.py
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from show_ball.inference.ball_detector import BallDetector
from show_ball.inference.ball_tracker import TrackerMode, BallTracker
from show_ball.utils.constants import (
    TILE_SIZE,
    OVERLAP,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    CROP_SIZE,
)
from show_ball.utils.helpers import choose_device

TEST_RESOURCES = Path(__file__).parent / "resources"
WEIGHTS_PATH = "run/weights/best.pt"


def make_detection(x1=10, y1=20, x2=30, y2=40):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.fixture
def sample_frame():
    return np.zeros((1920, 1080, 3), dtype=np.uint8)


@pytest.fixture
def full_frame():
    return cv2.imread(str(TEST_RESOURCES / "images" / "full_image.jpg"))


@pytest.fixture
def tile_frame():
    return cv2.imread(str(TEST_RESOURCES / "images" / "tile.jpg"))


@pytest.fixture
def detector():
    return BallDetector(
        weights=WEIGHTS_PATH,
        device=choose_device("auto"),
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )


@pytest.fixture
def tracker(detector):
    return BallTracker(
        predictor=detector,
        crop_size=CROP_SIZE,
        max_lost_frames=1,
        num_frames_to_skip=1,
    )


def test_is_tracking_false_when_no_last_detection(tracker):
    assert tracker.is_tracking is False


def test_is_tracking_true_when_last_detection_exists(tracker):
    tracker._last_detection = make_detection()

    assert tracker.is_tracking is True


def test_should_not_skip_when_num_frames_to_skip_is_zero(tracker):

    tracker._last_detection = make_detection()
    tracker._frame_counter = 100
    tracker._num_frames_to_skip = 0
    assert tracker._should_skip_detection() is False


def test_should_skip_detection_when_tracking_and_skip_frame(tracker):
    tracker._last_detection = make_detection()

    tracker._frame_counter = 1
    assert tracker._should_skip_detection() is False

    tracker._frame_counter = 2
    assert tracker._should_skip_detection() is True

    tracker._frame_counter = 3
    assert tracker._should_skip_detection() is False


def test_should_not_skip_detection_when_not_tracking(tracker):
    tracker._frame_counter = 2
    tracker._last_detection = None

    assert tracker._should_skip_detection() is False


def test_crop_around_last_detection(tracker, sample_frame):

    tracker._last_detection = make_detection(
        x1=600,
        y1=600,
        x2=610,
        y2=610,
    )

    crop, x0, y0 = tracker._crop(sample_frame)
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)

    # Detection center is (605, 605)
    # If x0/y0 are the crop top-left coordinates:

    assert x0 == 605 - CROP_SIZE // 2
    assert y0 == 605 - CROP_SIZE // 2


def test_crop_clamps_to_frame_boundaries(tracker, sample_frame):
    tracker._last_detection = make_detection(x1=0, y1=0, x2=10, y2=10)

    crop, x0, y0 = tracker._crop(sample_frame)

    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    assert x0 == 0
    assert y0 == 0


def test_crop_raises_when_no_last_detection(tracker, sample_frame):
    with pytest.raises(ValueError, match="There is no previous detection"):
        tracker._crop(sample_frame)


def test_update_detection_sets_last_detection_and_resets_lost_counter(tracker):
    det = make_detection()
    tracker._lost_frames_counter = 3

    tracker._update_detection(det)

    assert tracker._last_detection is det
    assert tracker._lost_frames_counter == 0


def test_should_reacquire_when_not_tracking(tracker):
    tracker._last_detection = None

    assert tracker._should_run_full_frame_detection() is True


def test_should_not_reacquire_when_tracking_and_lost_counter_within_limit(tracker):
    tracker._last_detection = make_detection()
    tracker._lost_frames_counter = 2
    tracker._max_lost_frames = 2
    tracker._frame_counter = 2

    assert tracker._should_run_full_frame_detection() is False


def test_should_reacquire_when_lost_counter_exceeds_limit(tracker):
    tracker._last_detection = make_detection()
    tracker._lost_frames_counter = 3
    tracker._max_lost_frames = 2

    assert tracker._should_run_full_frame_detection() is True


def test_mark_lost_frame_increments_counter(tracker):
    tracker._lost_frames_counter = 1

    tracker._mark_lost_frame()

    assert tracker._lost_frames_counter == 2


def test_reset_tracking_clears_state(tracker):
    tracker._last_detection = make_detection()
    tracker._lost_frames_counter = 3

    tracker._reset_tracking()

    assert tracker._last_detection is None
    assert tracker._lost_frames_counter == 0


def test_process_frame_marks_lost_when_crop_detection_fails_but_does_not_reacquire_yet(
    tracker, sample_frame
):

    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)
    det, mode = tracker.process_frame(sample_frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._lost_frames_counter == 1
    assert tracker.is_tracking is True


def test_process_frame_reacquires_with_full_frame_after_too_many_lost_frames(tracker, full_frame):

    tracker._last_detection = None
    tracker._lost_frames_counter = 1

    det, mode = tracker.process_frame(full_frame)

    assert mode == TrackerMode.FULL
    assert tracker._last_detection is not None
    assert tracker._lost_frames_counter == 0


def test_process_frame_resets_tracking_when_full_frame_reacquisition_fails(tracker, sample_frame):

    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)
    tracker._lost_frames_counter = 2

    det, mode = tracker.process_frame(sample_frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._last_detection is None
    assert tracker._lost_frames_counter == 0


def test_process_frame_runs_full_frame_detection_when_not_tracking(tracker, full_frame, detector):

    det, mode = tracker.process_frame(full_frame)

    assert mode == TrackerMode.FULL
    assert tracker._last_detection is not None
    assert det.cls == 0
    assert det.conf >= 0.1


def test_process_frame_returns_lost_when_not_tracking_and_full_frame_detection_fails(
    tracker, sample_frame
):
    det, mode = tracker.process_frame(sample_frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._last_detection is None
    assert tracker._lost_frames_counter == 0

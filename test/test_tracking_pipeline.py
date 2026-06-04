"""
Tests for ball_tracker.py
"""
from types import SimpleNamespace

import numpy as np
import pytest

from show_ball.inference.ball_tracker import TrackerMode, BallTracker


def make_detection(x1=10, y1=20, x2=30, y2=40):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)


class FakePredictor:
    def __init__(self, detections=None):
        self.detections = list(detections or [])
        self.calls = []

    def best_detection(self, frame, use_tiles):
        self.calls.append({"frame": frame, "use_tiles": use_tiles})
        if not self.detections:
            return None
        return self.detections.pop(0)


@pytest.fixture
def frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def tracker():
    return BallTracker(
        predictor=FakePredictor(),
        crop_size=20,
        max_lost_frames=2,
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


def test_crop_around_last_detection(tracker, frame):
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)

    crop, x0, y0 = tracker._crop(frame)

    assert crop.shape == (20, 20, 3)
    assert x0 == 40
    assert y0 == 40


def test_crop_clamps_to_frame_boundaries(tracker, frame):
    tracker._last_detection = make_detection(x1=0, y1=0, x2=10, y2=10)

    crop, x0, y0 = tracker._crop(frame)

    assert crop.shape == (20, 20, 3)
    assert x0 == 0
    assert y0 == 0


def test_crop_raises_when_no_last_detection(tracker, frame):
    with pytest.raises(ValueError, match="There is no previous detection"):
        tracker._crop(frame)


def test_update_detection_sets_last_detection_and_resets_lost_counter(tracker):
    det = make_detection()
    tracker._lost_frames_counter = 3

    tracker._update_detection(det)

    assert tracker._last_detection is det
    assert tracker._lost_frames_counter == 0


def test_run_crop_detection_offsets_detection_coordinates(frame):
    det = make_detection(x1=1, y1=2, x2=3, y2=4)
    predictor = FakePredictor(detections=[det])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)

    result = tracker._run_crop_detection(frame)

    assert result.x1 == 41
    assert result.y1 == 42
    assert result.x2 == 43
    assert result.y2 == 44
    assert predictor.calls[0]["use_tiles"] is False
    assert predictor.calls[0]["frame"].shape == (20, 20, 3)


def test_run_crop_detection_returns_none_when_no_detection(frame):
    predictor = FakePredictor(detections=[None])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)

    result = tracker._run_crop_detection(frame)

    assert result is None


def test_run_full_frame_detection_uses_tiles(frame):
    det = make_detection()
    predictor = FakePredictor(detections=[det])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )

    result = tracker._run_full_frame_detection(frame)

    assert result is det
    assert predictor.calls[0]["use_tiles"] is True
    assert predictor.calls[0]["frame"] is frame


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


def test_process_frame_returns_last_detection_when_skipping(frame):
    predictor = FakePredictor()

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    last_det = make_detection()
    tracker._last_detection = last_det
    tracker._frame_counter = 1

    det, mode = tracker.process_frame(frame)

    assert det is last_det
    assert mode == TrackerMode.TRACK
    assert predictor.calls == []


def test_process_frame_tracks_with_crop_detection(frame):
    crop_det = make_detection(x1=1, y1=2, x2=3, y2=4)
    predictor = FakePredictor(detections=[crop_det])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)

    det, mode = tracker.process_frame(frame)

    assert mode == TrackerMode.TRACK
    assert det.x1 == 41
    assert det.y1 == 42
    assert det.x2 == 43
    assert det.y2 == 44
    assert tracker._last_detection is det
    assert tracker._lost_frames_counter == 0


def test_process_frame_marks_lost_when_crop_detection_fails_but_does_not_reacquire_yet(frame):
    predictor = FakePredictor(detections=[None])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)

    det, mode = tracker.process_frame(frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._lost_frames_counter == 1
    assert tracker.is_tracking is True
    assert len(predictor.calls) == 1
    assert predictor.calls[0]["use_tiles"] is False


def test_process_frame_reacquires_with_full_frame_after_too_many_lost_frames(frame):
    full_det = make_detection(x1=5, y1=6, x2=7, y2=8)
    predictor = FakePredictor(detections=[None, full_det])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)
    tracker._lost_frames_counter = 2

    det, mode = tracker.process_frame(frame)

    assert det is full_det
    assert mode == TrackerMode.FULL
    assert tracker._last_detection is full_det
    assert tracker._lost_frames_counter == 0
    assert predictor.calls[0]["use_tiles"] is False
    assert predictor.calls[1]["use_tiles"] is True


def test_process_frame_resets_tracking_when_full_frame_reacquisition_fails(frame):
    predictor = FakePredictor(detections=[None, None])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )
    tracker._last_detection = make_detection(x1=40, y1=40, x2=60, y2=60)
    tracker._lost_frames_counter = 2

    det, mode = tracker.process_frame(frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._last_detection is None
    assert tracker._lost_frames_counter == 0
    assert predictor.calls[0]["use_tiles"] is False
    assert predictor.calls[1]["use_tiles"] is True


def test_process_frame_runs_full_frame_detection_when_not_tracking(frame):
    full_det = make_detection()
    predictor = FakePredictor(detections=[full_det])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )

    det, mode = tracker.process_frame(frame)

    assert det is full_det
    assert mode == TrackerMode.FULL
    assert tracker._last_detection is full_det
    assert predictor.calls[0]["use_tiles"] is True


def test_process_frame_returns_lost_when_not_tracking_and_full_frame_detection_fails(frame):
    predictor = FakePredictor(detections=[None])

    tracker = BallTracker(
        predictor=predictor,
        crop_size=20,
        max_lost_frames=2,
        num_frames_to_skip=1,
    )

    det, mode = tracker.process_frame(frame)

    assert det is None
    assert mode == TrackerMode.LOST
    assert tracker._last_detection is None
    assert tracker._lost_frames_counter == 0
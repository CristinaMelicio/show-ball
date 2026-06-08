"""
Module with functions for calculating tracking and runtime metrics for evaluating the performance of
a ball tracking pipeline against ground truth annotations.
"""

from dataclasses import dataclass
from math import hypot
from statistics import mean, median


@dataclass
class RuntimeMetrics:
    """
    Dataclass for the runtime metrics
    """

    #: Total frames in the video.
    total_frames: int

    #: Amount of frames processed.
    processed_frames: int

    #: Frames per second of the video.
    video_fps: float

    #: Processing frames per second by the tracking pipeline.
    processing_fps: float

    #: Factor by which the processing is faster (greater than 1) or slower (less than 1) than the
    # video FPS.
    realtime_factor: float

    #: Elapsed time in seconds for processing the video.
    elapsed_seconds: float

    #: Average CPU percentage usage.
    mean_cpu_percent: float

    #: Maximum memory used.
    max_memory_mb: float

    def __str__(self) -> str:
        return (
            f"Runtime Metrics\n"
            f"----------------\n"
            f"Total frames      : {self.total_frames}\n"
            f"Processed frames  : {self.processed_frames}\n"
            f"Video FPS         : {self.video_fps:.2f}\n"
            f"Processing FPS    : {self.processing_fps:.2f}\n"
            f"Realtime factor   : {self.realtime_factor:.2f}x\n"
            f"Elapsed time      : {self.elapsed_seconds:.2f} s\n"
            f"Mean CPU          : {self.mean_cpu_percent:.1f}%\n"
            f"Peak memory       : {self.max_memory_mb:.1f} MB"
        )


@dataclass
class TrackingMetrics:
    """
    Tracking metrics for evaluating the performance of a ball tracking pipeline against
    ground truth annotations.
    """

    #: Fraction of predicted ball positions that are correct within the distance threshold.
    precision: float

    #: Fraction of ground-truth ball positions that are successfully detected within the
    #: distance threshold.
    recall: float

    #: Harmonic mean of precision and recall.
    f1: float

    #: Fraction of ground-truth frames with a prediction located within 5 pixels of the
    #: annotated ball position.
    recall_5px: float

    #: Fraction of ground-truth frames with a prediction located within 10 pixels of the
    #: annotated ball position.
    recall_10px: float

    #: Fraction of ground-truth frames with a prediction located within 20 pixels of the
    #: annotated ball position.
    recall_20px: float

    #: Mean Euclidean distance in pixels between predicted and ground-truth ball positions.
    mean_error_px: float

    #: Median Euclidean distance in pixels between predicted and ground-truth ball positions.
    median_error_px: float

    #: 95th percentile of the Euclidean distance error in pixels.
    p95_error_px: float

    #: Fraction of ground-truth frames where no valid ball prediction was produced.
    missing_detection_rate: float

    def __str__(self) -> str:
        return (
            "Tracking Metrics\n"
            "----------------\n"
            f"Precision              : {self.precision:.4f}\n"
            f"Recall                 : {self.recall:.4f}\n"
            f"F1 Score               : {self.f1:.4f}\n"
            f"Recall @ 5 px          : {self.recall_5px:.4f}\n"
            f"Recall @ 10 px         : {self.recall_10px:.4f}\n"
            f"Recall @ 20 px         : {self.recall_20px:.4f}\n"
            f"Mean Error             : {self.mean_error_px:.2f} px\n"
            f"Median Error           : {self.median_error_px:.2f} px\n"
            f"95th Percentile Error  : {self.p95_error_px:.2f} px\n"
            f"Missing Detection Rate : {self.missing_detection_rate:.4f}"
        )


def calculate_metrics(
    predicted: list[dict],
    ground_truth: list[dict],
    distance_threshold_px: float = 10.0,
) -> TrackingMetrics:
    """ """
    predicted_by_frame = {int(row["frame_no"]): row for row in predicted}
    ground_truth_by_frame = {int(row["frame_no"]): row for row in ground_truth}

    tp = 0
    fp = 0
    fn = 0

    missing_detections = 0
    errors: list[float] = []

    for frame_no, gt in ground_truth_by_frame.items():
        gt_x = float(gt["ball_x"])
        gt_y = float(gt["ball_y"])
        pred = predicted_by_frame.get(frame_no, None)

        if pred is None:
            fn += 1
            missing_detections += 1
            continue

        pred_x = float(pred["ball_x"])
        pred_y = float(pred["ball_y"])

        error = hypot(pred_x - gt_x, pred_y - gt_y)
        errors.append(error)

        if error <= distance_threshold_px:
            tp += 1

        else:
            fp += 1
            fn += 1

    for frame_no, pred in predicted_by_frame.items():
        if frame_no in ground_truth_by_frame:
            continue

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    total_gt_frames = len(ground_truth_by_frame)

    return TrackingMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        recall_5px=recall_at_threshold(errors, total_gt_frames, 5.0),
        recall_10px=recall_at_threshold(errors, total_gt_frames, 10.0),
        recall_20px=recall_at_threshold(errors, total_gt_frames, 20.0),
        mean_error_px=mean(errors) if errors else 0.0,
        median_error_px=median(errors) if errors else 0.0,
        p95_error_px=(sorted(errors)[int(0.95 * (len(errors) - 1))] if errors else 0.0),
        missing_detection_rate=(
            missing_detections / total_gt_frames * 100 if total_gt_frames > 0 else 0.0
        ),
    )


def recall_at_threshold(
    errors: list[float],
    total_gt_frames: int,
    threshold_px: float,
) -> float:
    """
    Calculate recall given a certain threshold in pixels.

    Args:
        errors: List of Euclidean distance errors in pixels between predicted and ground-truth ball
        positions for each frame.
        total_gt_frames: Total number of ground-truth frames.
        threshold_px: Distance in pixels.

    Return:
        Recall @px_threshold.
    """

    if total_gt_frames == 0:
        return 0.0

    return sum(error <= threshold_px for error in errors) / total_gt_frames

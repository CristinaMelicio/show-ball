"""
Utilities for representing detections and computing geometric relationships
between them.
"""

from dataclasses import dataclass
from math import hypot


@dataclass
class Detection:
    """
    Detection dataclass.
    """

    # Coordinates of the bounding box (top-left and bottom-right corners)
    x1: float
    y1: float
    x2: float
    y2: float

    # Confidence score of the detection
    conf: float

    # Class index of the detected object. For this is always 0
    cls: int


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


def detection_iou(a: Detection, b: Detection) -> float:
    """
    Compute the Intersection over Union (IoU) between two detections.

    Args:
        a: First detection.
        b: Second detection.

    Returns:
        The IoU score between the two bounding boxes.
    """

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def center_distance(a: Detection, b: Detection) -> float:
    """
    Compute the Euclidean distance between the centers of two detections.

    Args:

        a: First detection.
        b: Second detection.

    Returns:
        The Euclidean distance, in pixels, between the centers of the two
        bounding boxes.
    """

    ax, ay = center_position(a)
    bx, by = center_position(b)

    return hypot(ax - bx, ay - by)

import cv2
import numpy as np
import numpy.typing as npt

from show_ball.inference.ball_detector import Detection
from show_ball.inference.ball_tracker import TrackerMode, center_position


def draw_detection(
    frame: npt.NDArray[np.uint8], det: Detection | None, mode: TrackerMode, fps: float
):
    if det is not None:
        x1 = int(det.x1)
        y1 = int(det.y1)
        x2 = int(det.x2)
        y2 = int(det.y2)

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cx, cy = center_position(det)
        # Center point
        cv2.circle(
            frame,
            (cx, cy),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

        cv2.putText(
            frame,
            f"ball {det.conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    text = f"mode={mode}"
    if fps is not None:
        text += f" | FPS={fps:.2f}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame

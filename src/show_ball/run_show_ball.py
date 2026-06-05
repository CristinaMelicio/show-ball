""" """

import csv
import time
from pathlib import Path

import click
import cv2
import numpy as np
import psutil

from show_ball.inference.ball_detector import BallDetector
from show_ball.inference.ball_tracker import BallTracker, center_position
from show_ball.utils.constants import (
    ANNOTATED_VIDEO_FILENAME,
    CONFIDENCE_THRESHOLD,
    CROP_SIZE,
    IOU_THRESHOLD,
    MAX_LOST_FRAMES,
    NUM_FRAMES_TO_SKIP,
    OVERLAP,
    PREDICTIONS_FILENAME,
    TILE_SIZE,
    WINDOW_NAME,
)
from show_ball.utils.draw import draw_detection
from show_ball.utils.helpers import choose_device
from show_ball.utils.metrics import RuntimeMetrics


def run_tracking_pipeline_on_video(
    tracker: BallTracker,
    video_path: str | Path,
    output_path: str | Path,
    show: bool,
    window_name: str,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> RuntimeMetrics:
    """
    Run the tracker on the specified video and save the predicted ball positions to a CSV file.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_output_file = output_path / PREDICTIONS_FILENAME
    video_output_file = output_path / ANNOTATED_VIDEO_FILENAME

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        video_output_file,
        fourcc,
        video_fps,
        (width, height),
    )

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {video_output_file}")

    process = psutil.Process()
    process.cpu_percent(interval=None)

    cpu_samples: list[float] = []
    memory_samples_mb: list[float] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    processed_frames = 0
    frame_no = start_frame
    elapsed_seconds = 0

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with csv_output_file.open("w", newline="") as f:
            csv_writer = csv.DictWriter(
                f,
                fieldnames=["frame_no", "ball_x", "ball_y"],
            )
            csv_writer.writeheader()

            while True:
                if end_frame is not None and frame_no >= end_frame:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.perf_counter()
                det, mode = tracker.process_frame(frame)
                end_time = time.perf_counter()

                if det is not None:
                    ball_x, ball_y = center_position(det)
                    csv_writer.writerow(
                        {
                            "frame_no": frame_no,
                            "ball_x": ball_x,
                            "ball_y": ball_y,
                        }
                    )

                elapsed = end_time - start_time
                elapsed_seconds += elapsed
                current_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0

                annotated_frame = draw_detection(frame, det, mode, current_fps)
                video_writer.write(annotated_frame)

                cpu_samples.append(process.cpu_percent(interval=None))
                memory_samples_mb.append(process.memory_info().rss / (1024 * 1024))

                if show:
                    cv2.imshow(window_name, annotated_frame)

                    # Press q or ESC to stop early.
                    key = cv2.waitKey(17) & 0xFF
                    if key in (ord("q"), 27):
                        break

                processed_frames += 1
                frame_no += 1

    finally:
        cap.release()
        if show:
            cv2.destroyWindow(window_name)

    processing_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
    realtime_factor = processing_fps / video_fps if video_fps > 0 else 0.0

    return RuntimeMetrics(
        total_frames=total_frames,
        processed_frames=processed_frames,
        video_fps=video_fps,
        processing_fps=processing_fps,
        realtime_factor=realtime_factor,
        elapsed_seconds=elapsed_seconds,
        mean_cpu_percent=float(np.mean(cpu_samples)) if cpu_samples else 0.0,
        max_memory_mb=float(np.max(memory_samples_mb)) if memory_samples_mb else 0.0,
    )


@click.command("Run Tracking Pipeline")
@click.option("--video-path", required=True, type=click.Path(exists=True))
@click.option("--output-path", default="output/", type=click.Path(exists=False))
@click.option("--weights-path", default="run/weights/best.pt", type=click.Path(exists=True))
@click.option("--device", default="auto", type=click.Choice(["auto", "cpu", "cuda", "mps"]))
@click.option("--start-frame", default=0, type=int)
@click.option("--end-frame", default=None, type=int)
@click.option("--show", is_flag=True)
def main(
    video_path: str,
    output_path: str,
    weights_path: str,
    device: str,
    start_frame: int,
    end_frame: int | None,
    show: bool,
) -> None:
    """
    Run the tracking pipeline on the specified video and save the predicted ball positions to a CSV
    file.

    Args:
        video_path: path to the video.
        output_path: path to save the predicted ball positions.
        weights_path: path to the weights file.
        device: device to run the tracking pipeline on.
        start_frame: start frame of the video.
        end_frame: end frame of the video.
        show: show the video flag.
    """

    predictor = BallDetector(
        weights=weights_path,
        device=choose_device(device),
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )

    tracker = BallTracker(
        predictor,
        crop_size=CROP_SIZE,
        max_lost_frames=MAX_LOST_FRAMES,
        num_frames_to_skip=NUM_FRAMES_TO_SKIP,
    )

    runtime_metrics = run_tracking_pipeline_on_video(
        tracker=tracker,
        video_path=video_path,
        output_path=output_path,
        window_name=WINDOW_NAME,
        show=show,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    click.echo(runtime_metrics)


if __name__ == "__main__":
    main()

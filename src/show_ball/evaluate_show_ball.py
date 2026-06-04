"""
Module for evaluating the tracking performance of a ball tracking pipeline against ground truth
data.
"""

import csv
from pathlib import Path
from typing import Final

import click

from show_ball.utils.metrics import TrackingMetrics, calculate_metrics

#: Maximum distance to consider a predicted ball position as correct (in pixels).
DISTANCE_THRESHOLD: Final[int] = 10


def evaluate_tracking_predictions(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    distance_threshold_px: float,
    start_frame: int,
    end_frame: int | None = None,
) -> TrackingMetrics:
    """
    Evaluate the tracking performance of a ball tracking pipeline against ground truth.

    Args:
        predictions_path: Path to the CSV file with predictions format <frame_no, ball_x, ball_y>.
        ground_truth_path: Path to the CSV file with ground truth format <frame_no, ball_x, ball_y>.
        distance_threshold_px: Maximum distance to consider a predicted ball position as correct (in
            pixels).
        start_frame: start frame of the video.
        end_frame: end frame of the video.
    """

    def _read_csv(path: str | Path) -> list[dict]:
        with open(path, newline="") as f:
            rows = []

            for row in csv.DictReader(f):
                rows.append(
                    {
                        "frame_no": int(row["frame_no"]),
                        "ball_x": float(row["ball_x"]),
                        "ball_y": float(row["ball_y"]),
                    }
                )

            return rows

    predicted = _read_csv(predictions_path)
    ground_truth = _read_csv(ground_truth_path)

    predicted = [r for r in predicted if r["frame_no"] >= start_frame]
    ground_truth = [r for r in ground_truth if r["frame_no"] >= start_frame]

    if end_frame is not None:
        predicted = [r for r in predicted if r["frame_no"] < end_frame]
        ground_truth = [r for r in ground_truth if r["frame_no"] < end_frame]

    return calculate_metrics(
        predicted=predicted,
        ground_truth=ground_truth,
        distance_threshold_px=distance_threshold_px,
    )


@click.command("Run Evaluation on the ball tracking pipeline")
@click.option("--predictions-path", required=True, type=click.Path(exists=True))
@click.option("--ground-truth-path", required=True, type=click.Path(exists=True))
@click.option("--start-frame", default=0, type=int)
@click.option("--end-frame", default=None, type=int)
def main(
    predictions_path: str,
    ground_truth_path: str,
    start_frame: int,
    end_frame: int | None,
) -> None:
    """
    Args:
        predictions_path: Path to the CSV file with predictions format <frame_no, ball_x, ball_y>.
        ground_truth_path: Path to the CSV file with ground truth format <frame_no, ball_x, ball_y>.
        start_frame: start frame of the video.
        end_frame: end frame of the video.
    """

    tracking_metrics = evaluate_tracking_predictions(
        predictions_path=predictions_path,
        ground_truth_path=ground_truth_path,
        distance_threshold_px=DISTANCE_THRESHOLD,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    click.echo(tracking_metrics)


if __name__ == "__main__":
    main()

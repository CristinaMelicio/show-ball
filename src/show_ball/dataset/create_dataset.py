"""
Convert a video and CSV labels into a YOLO tiled dataset.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from show_ball.dataset.helpers import (
    estimate_missing_ball_position,
    find_neighbor_labels,
    make_hard_negative_tile_near_position,
    make_negative_tile,
    make_positive_tile,
)


def read_labels(csv_path: str) -> dict[int, tuple[float, float]]:
    """
    Reads the CSV file containing ball position labels and returns a dictionary mapping frame
    numbers to (x, y) coordinates.

    Args:
        csv_path: Path to the CSV file containing labels with columns: frame_no, ball_x, ball_y.

    Returns:
        A dictionary mapping frame numbers to (x, y) coordinates of the ball.
    """

    labels = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame_no = int(row["frame_no"])

            if row["ball_x"] == "" or row["ball_y"] == "":
                continue

            labels[frame_no] = (float(row["ball_x"]), float(row["ball_y"]))

    return labels


def make_segments(total_frames: int, segment_len: int) -> list[tuple[int, int]]:
    """
    Divides the total number of frames into segments of a specified length.

    Args:
        total_frames: Total number of frames in the video.
        segment_len: Desired length of each segment in frames.

    Returns:
        A list of tuples where each tuple contains the start and end frame indices for a segment.
    """

    return [
        (start, min(start + segment_len, total_frames))
        for start in range(0, total_frames, segment_len)
    ]


def split_segments(
    segments: list[tuple[int, int]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[tuple[int, int]]]:
    """
    Randomly shuffles and splits the list of segments into training, validation, and test sets
    based on specified ratios.

    Args:
        segments: List of tuples where each tuple contains the start and end indices for a segment.
        train_ratio: Proportion of segments to include in the training set.
        val_ratio: Proportion of segments to include in the validation set.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with keys "train", "val", and "test" mapping to lists of segment tuples for
        each split.
    """

    random.seed(seed)
    random.shuffle(segments)

    n = len(segments)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": segments[:n_train],
        "val": segments[n_train : n_train + n_val],
        "test": segments[n_train + n_val :],
    }


def get_split_for_frame(frame_idx: int, split_map: dict[str, list[tuple[int, int]]]) -> str | None:
    """
    Determines which split (train, val, test) a given frame index belongs to based on the provided
    split map.

    Args:
        frame_idx: The index of the frame to check.
        split_map: A dictionary with keys "train", "val", and "test" mapping to lists of segment
        tuples for each split.
    Returns:
        The name of the split ("train", "val", or "test") that the frame belongs to, or None if it
        does not belong to any split.
    """

    for split_name, segments in split_map.items():
        for start, end in segments:
            if start <= frame_idx < end:
                return split_name

    return None


def yolo_box_from_xy(
    x: float, y: float, img_w: int, img_h: int, box_w: int, box_h: int
) -> tuple[float, float, float, float]:
    """
    Converts (x, y) coordinates of the ball into YOLO format (x_center, y_center, width, height)
    normalized to [0, 1].

    Args:
        x: The x-coordinate of the ball in pixels.
        y: The y-coordinate of the ball in pixels.
        img_w: The width of the image in pixels.
        img_h: The height of the image in pixels.
        box_w: The width of the bounding box around the ball in pixels.
        box_h: The height of the bounding box around the ball in pixels.

    Returns:
        Coordinates of the ball into YOLO format (x_center, y_center, width, height) normalized to
        [0, 1].
    """

    x_center = max(0.0, min(1.0, x / img_w))
    y_center = max(0.0, min(1.0, y / img_h))
    w = min(box_w / img_w, 1.0)
    h = min(box_h / img_h, 1.0)

    return x_center, y_center, w, h


def ball_box_xyxy(
    x: float, y: float, box_w: float, box_h: float
) -> tuple[float, float, float, float]:
    """
    Given the (x, y) coordinates of the ball and the desired width and height of the bounding box,
    calculates the top-left (x0, y0) and bottom-right (x1, y1) coordinates of the bounding box in
    (x0, y0, x1, y1) format.

    Args:
        x: The x-coordinate of the ball in pixels.
        y: The y-coordinate of the ball in pixels.
        box_w: The width of the bounding box in pixels.
        box_h: The height of the bounding box in pixels.

    Returns:
        Bounding box coordinates of the bounding box in (x0, y0, x1, y1) format.
    """

    return (
        x - box_w / 2,
        y - box_h / 2,
        x + box_w / 2,
        y + box_h / 2,
    )


def yolo_label_for_tile(
    ball_x: float,
    ball_y: float,
    tile_x0: int,
    tile_y0: int,
    tile_size: int,
    box_w: float,
    box_h: float,
) -> tuple[float, float, float, float] | None:
    """
    Calculates the YOLO label (x_center, y_center, width, height) for the ball relative to a given
    tile.
    """

    local_x = ball_x - tile_x0
    local_y = ball_y - tile_y0

    if not (0 <= local_x < tile_size and 0 <= local_y < tile_size):
        return None

    x_c = local_x / tile_size
    y_c = local_y / tile_size

    w = box_w / tile_size
    h = box_h / tile_size

    # Keep values valid. In case of jitter or rounding issues, the ball might be slightly outside
    # the tile or the box might be slightly larger than the tile, so we clamp the values to ensure
    # they are valid.
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))

    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_c, y_c, w, h


def save_tile(
    frame: npt.NDArray[np.uint8],
    out_dir: Path,
    split_name: str,
    stem: str,
    tile_idx: int,
    x0: int,
    y0: int,
    tile_size: int,
    label=None,
) -> None:
    """
    Saves a tile of the frame as an image and its corresponding label in YOLO format.
    """

    tile = frame[y0 : y0 + tile_size, x0 : x0 + tile_size]
    image_out = out_dir / "images" / split_name / f"{stem}_tile_{tile_idx:03d}.jpg"
    label_out = out_dir / "labels" / split_name / f"{stem}_tile_{tile_idx:03d}.txt"

    cv2.imwrite(str(image_out), tile)
    if label is None:
        label_out.touch()

    else:
        x_c, y_c, w, h = label
        with open(label_out, "w") as f:
            f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--box-w", type=float, default=16)
    parser.add_argument("--box-h", type=float, default=16)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--pos-tiles-per-frame", type=int, default=2)
    parser.add_argument("--neg-tiles-per-frame", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--hard-neg-neighbour-distance", type=int, default=15)
    parser.add_argument("--hard-neg-tiles-for-missing-label", type=int, default=2)
    parser.add_argument("--hard-neg-offset-min", type=int, default=80)
    parser.add_argument("--hard-neg-offset-max", type=int, default=220)
    parser.add_argument("--jitter", type=float, default=0.5)
    parser.add_argument("--segment-len", type=int, default=1800)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frame-start", type=int, default=160, help="Ignore all frames before this frame index"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_map = read_labels(args.labels_csv)
    sorted_label_frames = sorted(label_map.keys())
    label_frames = set(sorted_label_frames)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.tile_size > img_w or args.tile_size > img_h:
        raise ValueError(f"tile-size={args.tile_size} is larger than image size {img_w}x{img_h}")

    segments = make_segments(total_frames, args.segment_len)

    split_map = split_segments(
        segments,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    frame_idx = 0
    saved_tiles = 0
    saved_positive = 0
    saved_negative = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx < args.frame_start:
            frame_idx += 1
            continue

        split_name = get_split_for_frame(frame_idx, split_map)
        if split_name is None:
            frame_idx += 1
            continue

        stem = f"{video_path.stem}_frame_{frame_idx:06d}"

        tile_idx = 0

        # --------------------------------------
        # Case 1: visible ball / positive frame
        # --------------------------------------
        if frame_idx in label_frames:
            if frame_idx % args.frame_stride != 0:
                frame_idx += 1
                continue
            ball_x, ball_y = label_map[frame_idx]

            expected_ball_box = ball_box_xyxy(
                ball_x,
                ball_y,
                args.box_w,
                args.box_h,
            )

            # Positive tiles
            for _ in range(args.pos_tiles_per_frame):
                x0, y0 = make_positive_tile(
                    ball_x,
                    ball_y,
                    img_w,
                    img_h,
                    args.tile_size,
                    args.jitter,
                )

                label = yolo_label_for_tile(
                    ball_x,
                    ball_y,
                    x0,
                    y0,
                    args.tile_size,
                    args.box_w,
                    args.box_h,
                )

                if label is not None:
                    save_tile(
                        frame,
                        out_dir,
                        split_name,
                        stem,
                        tile_idx,
                        x0,
                        y0,
                        args.tile_size,
                        label=label,
                    )

                    tile_idx += 1
                    saved_tiles += 1
                    saved_positive += 1

            # Negative tiles away from visible ball
            for _ in range(args.neg_tiles_per_frame):
                neg_xy = make_negative_tile(
                    img_w,
                    img_h,
                    args.tile_size,
                    forbidden_box=expected_ball_box,
                )

                if neg_xy is None:
                    continue

                x0, y0 = neg_xy

                save_tile(
                    frame,
                    out_dir,
                    split_name,
                    stem,
                    tile_idx,
                    x0,
                    y0,
                    args.tile_size,
                    label=None,
                )

                tile_idx += 1
                saved_tiles += 1
                saved_negative += 1

        # -------------------------------------------------
        # Case 2: missing label / ball hidden by the player
        # -------------------------------------------------
        else:
            previous_label, next_label = find_neighbor_labels(
                frame_idx,
                sorted_label_frames,
                label_map,
                args.hard_neg_neighbour_distance,
            )

            estimated_position = estimate_missing_ball_position(
                frame_idx,
                previous_label,
                next_label,
            )

            for _ in range(args.hard_neg_tiles_for_missing_label):
                neg_xy = None

                # Prefer hard negatives near the expected hidden-ball location
                if estimated_position is not None:
                    est_x, est_y = estimated_position

                    expected_ball_box = ball_box_xyxy(
                        est_x,
                        est_y,
                        args.box_w,
                        args.box_h,
                    )

                    neg_xy = make_hard_negative_tile_near_position(
                        est_x,
                        est_y,
                        img_w,
                        img_h,
                        args.tile_size,
                        forbidden_box=expected_ball_box,
                        offset_min=args.hard_neg_offset_min,
                        offset_max=args.hard_neg_offset_max,
                    )

                # Fallback: random negative tile
                if neg_xy is None:
                    neg_xy = make_negative_tile(
                        img_w,
                        img_h,
                        args.tile_size,
                        forbidden_box=None,
                    )

                if neg_xy is None:
                    continue

                x0, y0 = neg_xy

                save_tile(
                    frame,
                    out_dir,
                    split_name,
                    stem,
                    tile_idx,
                    x0,
                    y0,
                    args.tile_size,
                    label=None,
                )

                tile_idx += 1
                saved_tiles += 1
                saved_negative += 1

        frame_idx += 1

    cap.release()

    with open(out_dir / "split_record.json", "w") as f:
        json.dump(split_map, f, indent=2)

    print(f"Done. Dataset saved to: {out_dir}")
    print(f"Frames processed: {frame_idx}")
    print(f"Tiles saved: {saved_tiles}")
    print(f"Positive tiles: {saved_positive}")
    print(f"Negative tiles: {saved_negative}")


if __name__ == "__main__":
    main()

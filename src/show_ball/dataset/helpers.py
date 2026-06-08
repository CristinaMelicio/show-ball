"""
Helper functions for finding neighbor labels and estimating missing positions
"""

import random


def clamp_tile_xy(
    x0: int, y0: int, tile_w: int, tile_h: int, img_w: int, img_h: int
) -> tuple[int, int]:
    """
    Clamps the top-left corner (x0, y0) of a tile to ensure that the entire tile fits within the
    image boundaries.
    """
    x0 = max(0, min(x0, img_w - tile_w))
    y0 = max(0, min(y0, img_h - tile_h))

    return x0, y0


def intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """
    Checks if two boxes in (x0, y0, x1, y1) format intersect.

    Args:
        a: The first box in (x0, y0, x1, y1) format.
        b: The second box in (x0, y0, x1, y1) format.

    Returns:
        True if boxes intersects, false otherwise.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def find_neighbor_labels(
    frame_idx: int,
    sorted_label_frames: list[int],
    labels: dict[int, tuple[float, float]],
    max_distance: int,
) -> tuple[tuple[int, tuple[float, float]] | None, tuple[int, tuple[float, float]] | None]:
    """
    Finds the closest labeled frame before and after a given frame.
    """

    previous_label = None
    next_label = None

    for labelled_frame_idx in reversed(sorted_label_frames):
        if labelled_frame_idx < frame_idx:
            distance = frame_idx - labelled_frame_idx
            if distance <= max_distance:
                previous_label = (labelled_frame_idx, labels[labelled_frame_idx])
            break

    for labelled_frame_idx in sorted_label_frames:
        if labelled_frame_idx > frame_idx:
            distance = labelled_frame_idx - frame_idx
            if distance <= max_distance:
                next_label = (labelled_frame_idx, labels[labelled_frame_idx])
            break

    return previous_label, next_label


def estimate_missing_ball_position(
    frame_idx: int,
    previous_label: tuple[int, tuple[float, float]] | None,
    next_label: tuple[int, tuple[float, float]] | None,
) -> tuple[float, float] | None:
    """
    Estimates where the ball would be in an unlabelled frame based on neighboring labels.
    This is only used to create hard negatives near the expected ball trajectory.
    """

    if previous_label is not None and next_label is not None:
        previous_frame, (previous_x, previous_y) = previous_label
        next_frame, (next_x, next_y) = next_label

        if next_frame == previous_frame:
            return previous_x, previous_y

        alpha = (frame_idx - previous_frame) / (next_frame - previous_frame)
        x = previous_x + alpha * (next_x - previous_x)
        y = previous_y + alpha * (next_y - previous_y)
        return x, y

    if previous_label is not None:
        return previous_label[1]

    if next_label is not None:
        return next_label[1]

    return None


def make_hard_negative_tile_near_position(
    x: float,
    y: float,
    img_w: int,
    img_h: int,
    tile_size: int,
    offset_min: int,
    offset_max: int,
    max_tries: int = 20,
) -> tuple[int, int] | None:
    """
    Creates a hard negative tile near the estimated ball position.
    The tile should contain/intersect the expected ball area,
    because the ball is hidden/occluded there.
    """

    for _ in range(max_tries):
        offset_x = random.choice([-1, 1]) * random.randint(offset_min, offset_max)
        offset_y = random.choice([-1, 1]) * random.randint(offset_min, offset_max)

        x0 = int(x - tile_size / 2 + offset_x)
        y0 = int(y - tile_size / 2 + offset_y)
        x0, y0 = clamp_tile_xy(x0, y0, tile_size, tile_size, img_w, img_h)

        if x0 <= x < x0 + tile_size and y0 <= y < y0 + tile_size:
            return x0, y0

    return None


def make_negative_tile(
    img_w: int,
    img_h: int,
    tile_size: int,
    forbidden_box=None,
    max_tries: int = 3,
) -> tuple[int, int] | None:
    """
    Creates a tile that does not contain the ball, ensuring it does not intersect with the forbidden
    box (the ball's bounding box) if provided.

    Args:
        img_w: The width of the image in pixels.
        img_h: The height of the image in pixels.
        tile_size: The size of the tile in pixels.
        forbidden_box: The forbidden box in pixels.
        max_tries: The maximum number of times to try to make a tile, if it fails.

    Returns:
        The top-left corner (x0, y0) of the tile that does not contain the ball or None if it fails
        to find a valid tile after max_tries.
    """

    for _ in range(max_tries):
        x0 = random.randint(0, max(0, img_w - tile_size))
        y0 = random.randint(0, max(0, img_h - tile_size))

        tile_box = (x0, y0, x0 + tile_size, y0 + tile_size)
        if forbidden_box is None or not intersects(tile_box, forbidden_box):
            return x0, y0

    return None


def make_positive_tile(
    ball_x: float,
    ball_y: float,
    img_w: int,
    img_h: int,
    tile_size: int,
    jitter: float,
) -> tuple[int, int]:
    """
    Creates a tile containing the ball, with optional jitter to add variability to the ball's
    position.
    Example: jitter=0.0 ball centered, jitter=0.5 ball can move up to half tile.

    Args:
        ball_x: The x-coordinate of the ball in pixels.
        ball_y: The y-coordinate of the ball in pixels.
        img_w: The width of the image in pixels.
        img_h: The height of the image in pixels.
        tile_size: The size of the tile in pixels.
        jitter: The jitter to add variability to the ball's position.

    Returns:
        The top-left corner (x0, y0) of the tile containing the ball, clamped to ensure it fits
        within the image boundaries.
    """

    max_offset = int(tile_size * jitter)

    dx = random.randint(-max_offset, max_offset)
    dy = random.randint(-max_offset, max_offset)

    x0 = int(ball_x - tile_size / 2 + dx)
    y0 = int(ball_y - tile_size / 2 + dy)

    return clamp_tile_xy(x0, y0, tile_size, tile_size, img_w, img_h)

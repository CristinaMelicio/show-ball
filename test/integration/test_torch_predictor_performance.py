import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from show_ball.inference.ball_detector import BallDetector


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_torch_predictor_inference_time():
    weights_path = Path("test/resources/best.pt")

    if not weights_path.exists():
        pytest.skip(f"Missing weights file: {weights_path}")

    predictor = BallDetector(
        weights=weights_path,
        device="mps",
        tile_size=640,
        overlap=0.25,
        confidence_threshold=0.1,
        iou_threshold=0.5,
    )

    real_image = cv2.imread("test/resources/images/full_image.jpg")

    predictor.inference([real_image], use_tiles=True)

    num_runs = 10
    times = []

    for _ in range(num_runs):
        t0 = time.time()
        predictor.inference([real_image], use_tiles=True)
        t1 = time.time()

        times.append(t1-t0)

    mean_time = np.mean(times)
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    print(f"\nMean inference time: {mean_time}")
    print(f"FPS: {fps:.2f}")

    # Adjust this threshold for your machine/model.
    assert mean_time < 1.0
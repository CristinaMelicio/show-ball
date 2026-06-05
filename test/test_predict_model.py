from types import SimpleNamespace

import numpy as np
import pytest
import torch

from show_ball.inference.ball_detector import Detection, BallDetector


@pytest.fixture
def sample_image():
    return np.zeros((640, 640, 3), dtype=np.uint8)


def make_predictor(
    tile_size=512,
    overlap=0.25,
    confidence_threshold=0.1,
    iou_threshold=0.5,
):
    predictor = BallDetector.__new__(BallDetector)
    predictor._device = "cpu"
    predictor._input_size = tile_size
    predictor._overlap = overlap
    predictor._confidence_threshold = confidence_threshold
    predictor._iou_threshold = iou_threshold
    predictor._model = None
    return predictor


def make_result(
    boxes,
    confs,
    classes,
):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=torch.tensor(boxes, dtype=torch.float32),
            conf=torch.tensor(confs, dtype=torch.float32),
            cls=torch.tensor(classes, dtype=torch.float32),
        )
    )


def test_tile_positions_for_640x640_image():
    predictor = make_predictor(tile_size=512, overlap=0.25)

    positions = predictor._tile_positions(width=640, height=640)

    assert positions == [
        (0, 0),
        (128, 0),
        (0, 128),
        (128, 128),
    ]


def test_pre_processing_without_tiles_returns_original_image(sample_image):
    predictor = make_predictor()

    tiles = predictor.pre_processing([sample_image], use_tiles=False)

    assert len(tiles) == 1

    tile, image_idx, x0, y0 = tiles[0]

    assert tile is sample_image
    assert image_idx == 0
    assert x0 == 0
    assert y0 == 0


def test_pre_processing_with_tiles_returns_512_tiles(sample_image):
    predictor = make_predictor(tile_size=512, overlap=0.25)

    tiles = predictor.pre_processing([sample_image], use_tiles=True)

    assert len(tiles) == 4

    positions = [(x0, y0) for _, _, x0, y0 in tiles]
    assert positions == [
        (0, 0),
        (128, 0),
        (0, 128),
        (128, 128),
    ]

    for tile, image_idx, _, _ in tiles:
        assert tile.shape == (512, 512, 3)
        assert image_idx == 0


def test_pre_processing_pads_small_image():
    small_image = np.zeros((300, 300, 3), dtype=np.uint8)
    predictor = make_predictor(tile_size=512, overlap=0.25)

    tiles = predictor.pre_processing([small_image], use_tiles=True)

    assert len(tiles) == 1

    tile, image_idx, x0, y0 = tiles[0]

    assert tile.shape == (300, 300, 3)
    assert image_idx == 0
    assert x0 == 0
    assert y0 == 0

    assert np.array_equal(tile[:300, :300], small_image)
    assert np.all(tile[300:, :] == 0)
    assert np.all(tile[:, 300:] == 0)


def test_inference_without_tiles_maps_detection_to_original_image(sample_image):
    predictor = make_predictor()

    predictor._model = SimpleNamespace(
        predict=lambda **kwargs: [
            make_result(
                boxes=[[10, 20, 30, 40]],
                confs=[0.9],
                classes=[0],
            )
        ]
    )

    detections_per_image = predictor.inference(
        [sample_image],
        use_tiles=False,
    )

    assert len(detections_per_image) == 1
    assert len(detections_per_image[0]) == 1

    det = detections_per_image[0][0]

    assert det == Detection(
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
        conf=pytest.approx(0.9),
        cls=0,
    )


def test_inference_with_tiles_offsets_detection_coordinates(sample_image):
    predictor = make_predictor(tile_size=512, overlap=0.25)

    # 4 tiles for a 640x640 image:
    # (0,0), (128,0), (0,128), (128,128)
    predictor._model = SimpleNamespace(
        predict=lambda **kwargs: [
            make_result([], [], []),
            make_result([[10, 20, 30, 40]], [0.9], [0]),
            make_result([], [], []),
            make_result([], [], []),
        ]
    )

    detections_per_image = predictor.inference(
        [sample_image],
        use_tiles=True,
    )

    assert len(detections_per_image[0]) == 1

    det = detections_per_image[0][0]

    assert det.x1 == 138.0
    assert det.y1 == 20.0
    assert det.x2 == 158.0
    assert det.y2 == 40.0
    assert det.conf == pytest.approx(0.9)
    assert det.cls == 0


def test_inference_clips_detection_to_image_bounds(sample_image):
    predictor = make_predictor(tile_size=512, overlap=0.25)

    predictor._model = SimpleNamespace(
        predict=lambda **kwargs: [
            make_result([], [], []),
            make_result([], [], []),
            make_result([], [], []),
            make_result([[400, 400, 600, 600]], [0.9], [0]),
        ]
    )

    detections_per_image = predictor.inference(
        [sample_image],
        use_tiles=True,
    )

    det = detections_per_image[0][0]

    assert det.x1 == 528.0
    assert det.y1 == 528.0
    assert det.x2 == 640.0
    assert det.y2 == 640.0


def test_post_processing_removes_overlapping_lower_confidence_detection():
    predictor = make_predictor(iou_threshold=0.5)

    detections = [
        Detection(x1=10, y1=10, x2=100, y2=100, conf=0.9, cls=0),
        Detection(x1=12, y1=12, x2=102, y2=102, conf=0.8, cls=0),
        Detection(x1=300, y1=300, x2=350, y2=350, conf=0.7, cls=0),
    ]

    final_results = predictor.post_processing([detections])

    assert len(final_results) == 1
    assert len(final_results[0]) == 2

    kept_confs = [det.conf for det in final_results[0]]

    assert 0.9 in kept_confs
    assert 0.7 in kept_confs
    assert 0.8 not in kept_confs


def test_best_detection_returns_none_when_no_detections(sample_image):
    predictor = make_predictor()

    predictor.inference = lambda images, use_tiles: [[]]

    det = predictor.best_detection(sample_image, use_tiles=True)

    assert det is None


def test_best_detection_returns_highest_confidence_detection(sample_image):
    predictor = make_predictor()

    low_conf = Detection(x1=1, y1=1, x2=2, y2=2, conf=0.2, cls=0)
    high_conf = Detection(x1=3, y1=3, x2=4, y2=4, conf=0.9, cls=0)

    predictor.inference = lambda images, use_tiles: [[low_conf, high_conf]]

    det = predictor.best_detection(sample_image, use_tiles=True)

    assert det is high_conf
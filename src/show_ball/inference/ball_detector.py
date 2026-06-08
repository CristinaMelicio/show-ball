"""
YOLO tiled inference predictor.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from ultralytics import YOLO

from show_ball.inference.utils import Detection


class BallDetector:
    """
    Predictor that performs tiled YOLO inference on input images and applies global NMS to
    return final detections per original image.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str,
        tile_size: int,
        overlap: float,
        confidence_threshold: float,
        iou_threshold: float,
    ):
        """
        Args:
            weights: Path to YOLO weights.
            device: Device to run inference on: "mps" for Mac with Apple Silicon, "cuda" for Nvidia
            GPUs, or "cpu".
            tile_size: Size of tiles to use in the inference.
            overlap: Overlap between tiles.
            confidence_threshold: Confidence threshold to use in the inference.
            iou_threshold: IoU threshold to use in the inference.
        """

        # Device string.
        self._device: str = device

        # Input size of the model.
        self._input_size: int = tile_size

        # Amount of overlap between tiles.
        self._overlap: float = overlap

        # Confidence threshold that filters out detections.
        self._confidence_threshold: float = confidence_threshold

        # IoU threshold for the NMS
        self._iou_threshold: float = iou_threshold

        # Yolo model
        self._model = YOLO(str(weights), task="detect")

    def _tile_positions(self, width: int, height: int) -> list[tuple[int, int]]:
        """
        Computes top-left positions for tiling an image of given width and height with specified
        tile size and overlap.

        Args:
            width: Width of the original image.
            height: Height of the original image.

        Returns:
            Top-left positions for tiling an image of given width and height.
        """

        stride = int(self._input_size * (1.0 - self._overlap))
        stride = max(1, stride)

        xs = list(range(0, max(width - self._input_size, 0) + 1, stride))
        ys = list(range(0, max(height - self._input_size, 0) + 1, stride))

        if not xs or xs[-1] != width - self._input_size:
            xs.append(max(width - self._input_size, 0))

        if not ys or ys[-1] != height - self._input_size:
            ys.append(max(height - self._input_size, 0))

        return [(x, y) for y in ys for x in xs]

    def pre_processing(
        self,
        images: list[npt.NDArray[np.uint8]],
        use_tiles: bool,
    ) -> list[tuple[npt.NDArray[np.uint8], int, int, int]]:
        """
        Splits input images into overlapping tiles.

        Args:
            images: Input images to preprocess.
            use_tiles: Whether the input images should be tiled. If False, we return the
            original images with (0, 0) tile positions.

        Returns:
            List of tuples containing the tile image, original image index, and tile's top-left
            position (x0, y0) in the original image.
        """

        if not use_tiles:
            return [(image, image_idx, 0, 0) for image_idx, image in enumerate(images)]

        tiles = []
        for image_idx, image in enumerate(images):
            h, w = image.shape[:2]

            if h <= self._input_size and w <= self._input_size:
                tiles.append((image, image_idx, 0, 0))
                continue

            for x0, y0 in self._tile_positions(w, h):
                tile = image[y0 : y0 + self._input_size, x0 : x0 + self._input_size]

                if tile.shape[0] != self._input_size or tile.shape[1] != self._input_size:
                    padded = np.zeros(
                        (self._input_size, self._input_size, image.shape[2]),
                        dtype=image.dtype,
                    )
                    padded[: tile.shape[0], : tile.shape[1]] = tile
                    tile = padded

                tiles.append((tile, image_idx, x0, y0))

        return tiles

    def inference(
        self,
        images: list[npt.NDArray[np.uint8]],
        use_tiles: bool = True,
    ) -> list[list[Detection]]:
        """
        Runs tiled YOLO inference and returns detections per original image.

        Args:
            images: Input images to preprocess.
            use_tiles: Whether to tile the input images. If True, we perform the inference on the
            tiles otherwise we perform the inference a single image.

        Returns:
            Detections per original image.
        """

        tiles = self.pre_processing(images, use_tiles)

        if not tiles:
            return [[] for _ in images]

        input_images = [t[0] for t in tiles]

        results_per_tile = self._model.predict(
            source=input_images,
            device=self._device,
            imgsz=self._input_size,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        detections_per_image: list[list[Detection]] = [[] for _ in images]
        for result, (_, image_idx, x0, y0) in zip(results_per_tile, tiles):
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            img_h, img_w = images[image_idx].shape[:2]

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box

                det = Detection(
                    x1=float(np.clip(x1 + x0, 0, img_w)),
                    y1=float(np.clip(y1 + y0, 0, img_h)),
                    x2=float(np.clip(x2 + x0, 0, img_w)),
                    y2=float(np.clip(y2 + y0, 0, img_h)),
                    conf=float(conf),
                    cls=int(cls),
                )

                detections_per_image[image_idx].append(det)

        return detections_per_image

    def post_processing(
        self,
        detections_per_image: list[list[Detection]],
    ) -> list[list[Detection]]:
        """
        Applies global NMS per original image.

        Args:
            detections_per_image: List of detections per original image.

        Returns:
            Detections after NMS.
        """

        final_results = []

        for detections in detections_per_image:
            if not detections:
                final_results.append([])
                continue

            boxes = torch.tensor(
                [[d.x1, d.y1, d.x2, d.y2] for d in detections],
                dtype=torch.float32,
            )
            scores = torch.tensor([d.conf for d in detections], dtype=torch.float32)

            keep = torch.ops.torchvision.nms(boxes, scores, self._iou_threshold)

            final_results.append([detections[i] for i in keep.tolist()])

        return final_results

    def best_detection(
        self,
        image: npt.NDArray[np.uint8],
        use_tiles: bool,
    ) -> Detection | None:
        """
        Runs inference on the input image and returns the detection with the highest confidence.

        Args:
            image: Input image to preprocess.
            use_tiles: Whether to tile the input images. If True, we perform the inference on the
            tiles otherwise we perform the inference a single image.

        Returns:
            Best detection per original image.
        """

        detections = self.inference([image], use_tiles=use_tiles)[0]

        if not detections:
            return None

        return max(detections, key=lambda d: d.conf)

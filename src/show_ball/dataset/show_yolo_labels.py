import argparse
import random
from pathlib import Path

import cv2


def draw_yolo_label(image, label_path):
    h, w = image.shape[:2]

    if not label_path.exists() or label_path.stat().st_size == 0:
        return image, False

    with open(label_path, "r") as f:
        lines = f.readlines()

    has_label = False

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, x_c, y_c, box_w, box_h = parts

        x_c = float(x_c) * w
        y_c = float(y_c) * h
        box_w = float(box_w) * w
        box_h = float(box_h) * h

        x1 = int(x_c - box_w / 2)
        y1 = int(y_c - box_h / 2)
        x2 = int(x_c + box_w / 2)
        y2 = int(y_c + box_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (int(x_c), int(y_c)), 3, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        has_label = True

    return image, has_label


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--only-labeled", action="store_true")
    parser.add_argument("--out-dir", default="label_preview")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images" / args.split
    labels_dir = dataset_dir / "labels" / args.split
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    if args.only_labeled:
        image_paths = [
            p
            for p in image_paths
            if (labels_dir / f"{p.stem}.txt").exists()
            and (labels_dir / f"{p.stem}.txt").stat().st_size > 0
        ]

    if not image_paths:
        raise RuntimeError("No images found.")

    sample = random.sample(image_paths, min(args.num, len(image_paths)))

    for image_path in sample:
        label_path = labels_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read {image_path}")
            continue

        image, has_label = draw_yolo_label(image, label_path)

        status = "labeled" if has_label else "empty_label"
        out_path = out_dir / f"{image_path.stem}_{status}.jpg"

        cv2.imwrite(str(out_path), image)
        print(f"Saved {out_path}")

    print(f"\nPreview images saved in: {out_dir}")


if __name__ == "__main__":
    main()

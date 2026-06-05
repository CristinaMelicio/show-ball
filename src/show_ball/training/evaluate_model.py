"""
Evaluate a trained YOLO model on the test set and save visual predictions.
"""

import random
from pathlib import Path

import click
import yaml
from ultralytics import YOLO

from show_ball.utils import choose_device


def get_test_images(data_yaml: str) -> list[Path]:
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg["path"])
    test_dir = dataset_root / data_cfg["test"]

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(test_dir.glob(ext))

    return sorted(images)


@click.command("Evaluate YOLO model on test set.")
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--weights", required=True, type=click.Path(exists=True))
@click.option("--device", default="mps", type=str)
@click.option("--samples", default=50, type=int)
@click.option("--conf", default=0.1, type=float)
@click.option("--out-dir", default="eval/ball_yolo26_test", type=str)
def main(
    config: str,
    weights: str,
    device: str,
    samples: int,
    conf: float,
    out_dir: str,
):
    device = choose_device(device)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    data_yaml = cfg["data"]

    train_cfg = cfg.get("train", {})
    imgsz = train_cfg.get("imgsz", 640)

    model = YOLO(weights)

    click.echo("Running validation on test split...")
    metrics = model.val(
        data=data_yaml,
        split="test",
        device=device,
        imgsz=imgsz,
        batch=train_cfg.get("batch", 4),
        conf=conf,
        plots=True,
        project=out_dir,
        fraction=0.1,  # 10% do test set
        name="metrics",
    )

    click.echo("\nTest metrics:")
    click.echo(f"mAP50:     {metrics.box.map50:.4f}")
    click.echo(f"mAP50-95:  {metrics.box.map:.4f}")
    click.echo(f"Precision: {metrics.box.mp:.4f}")
    click.echo(f"Recall:    {metrics.box.mr:.4f}")

    test_images = get_test_images(data_yaml)

    if not test_images:
        raise RuntimeError(f"No test images found in {data_yaml}")

    selected = random.sample(test_images, min(samples, len(test_images)))

    click.echo(f"\nRunning predictions on {len(selected)} test images...")

    model.predict(
        source=[str(p) for p in selected],
        device=device,
        imgsz=imgsz,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=out_dir,
        name="samples",
    )

    click.echo(f"\nDone. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

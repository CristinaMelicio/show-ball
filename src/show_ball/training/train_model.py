"""
This script is designed to train a YOLOv26 model for object detection using the Ultralytics library.
"""

import click
import yaml
from ultralytics import YOLO

from show_ball.utils.helpers import choose_device


@click.command("Train a YOLOv26 model for object detection.")
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML configuration file",
)
@click.option("--device", default="mps", help="Desired device to use", type=str)
def main(device: str, config: str):
    device = choose_device(device)
    click.echo(f"Using device: {device}")

    with open(config) as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"])

    model.train(
        data=cfg["data"],
        device=device,
        **cfg["train"],
    )


if __name__ == "__main__":
    main()

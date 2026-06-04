from ultralytics import YOLO


def main():

    model = YOLO("run/weights/best.pt")

    engine_path = model.export(
        format="engine",
        imgsz=640,
        batch=8,
        dynamic=True,
        half=True,
        nms=True,
    )

    print(f"Exported engine: {engine_path}")


if __name__ == "__main__":
    main()

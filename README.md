# ⚽ Soccer Ball Position Prediction in Video Frames

### Prerequisites
- Python 3.12
- Git
- Make (Linux/macOS)

### Setup

1. Clone the repository 
   ```
   git clone https://github.com/CristinaMelicio/show-ball.git
   cd show-ball
   ```
2. Create the environment and install dependencies.
   ```
   make install 
   ```
3. Activate the virtual environment.
   ```
   source .venv/bin/activate
   ```
4. Check the installation.
   ```
   make check
   make test
   ```
   
5. Export the model to TensorRT format for faster inference (optional, requires NVIDIA GPU).
   ```
   make export-tensorrt
   ```


## Usage
Run the ball tracker on a video:
```
python -m tools.run_pipeline \     
--weights path/to/weights \    
--video path/to/video \     
--output path/to/prediction \     
--show 
```

### Arguments

| Argument | Description |
|-----------|-------------|
| --weights | Path to the trained model weights (for example best.pt). |
| --video | Path to the input video file. |
| --output | Path to the output CSV file containing ball position predictions. |
| --show | Optional flag to display detections while processing the video. |


# Tracking Pipeline

![Project Image](resources/docs/image.png)

The pipeline consists of two main stages:

1. **Ball Detection** – A YOLO-based detector (`BallDetector`) is used to locate the ball in each frame.
2. **Ball Tracking** – The `BallTracker` combines detector outputs with temporal information from previous frames to maintain a stable estimate of the ball position over time.

For every frame, the pipeline:

- Reads the frame from the input video.
- Runs ball detection and tracking.
- Stores the predicted ball position in a CSV file.
- Draws the detection and tracking status on the frame.
- Writes the annotated frame to an output video.
- Collects runtime statistics such as processing speed, CPU usage, and memory consumption.


## Tracking pipeline

```bash
python show_ball.run_show_ball \
    --video-path resources/video.mp4 \
    --weights-path run/weights/best.pt \
    --output-path outputs \
    --device auto
```

| Argument | Description                                                       |
|-----------|-------------------------------------------------------------------|
| `--video-path` | Path to the input video.                                          |
| `--weights-path` | Path to the trained YOLO weights. Weights can be .pt or .engine.  |
| `--output-path` | Directory where outputs will be saved.                            |
| `--device` | Execution device (`auto`, `cpu`, `cuda`, or `mps`).               |
| `--start-frame` | First frame to process.                                           |
| `--end-frame` | Last frame to process. By default processes until the last frame. |
| `--show` | Display the annotated video while processing.                     |

### Outputs

The pipeline generates:
#### Predictions CSV
This file contains the predicted ball center coordinates for each frame like

  ```text
  frame_no,ball_x,ball_y
  0,523,312
  1,526,309
  ...
  ```

#### Annotated Video
The video contains the original frames with the predicted ball position
and tracking information overlaid.

#### Runtime Metrics
At the end of execution, the following metrics are printed to the console:

- **Total Frames** – Total number of frames in the input video.
- **Processed Frames** – Number of frames processed by the pipeline.
- **Video FPS** – Frame rate of the input video.
- **Processing FPS** – Average processing speed of the pipeline.
- **Realtime Factor** – Ratio between processing speed and video speed.
- **Elapsed Time** – Total processing time in seconds.
- **Mean CPU Usage** – Average CPU utilization during execution.
- **Peak Memory Usage** – Maximum memory consumption during execution.


## Evaluate tracking pipeline
This module evaluates the performance of a ball tracking pipeline against ground-truth annotations.

The evaluation compares predicted ball positions with annotated ground-truth positions on a frame-by-frame basis.
Each prediction is considered **correct** if its Euclidean distance from the ground-truth ball position is less than or equal to a configurable distance threshold (10 pixels by default).
The evaluation computes standard detection metrics as well as localization accuracy metrics.

For each ground-truth frame:

1. Find the corresponding prediction.
2. Compute the Euclidean distance between predicted and ground-truth positions.
3. If the distance is within the threshold:
   - True Positive (TP)
4. Otherwise:
   - False Positive (FP)
   - False Negative (FN)

Frames with no prediction are counted as:
- False Negative (FN)
- Missing Detection

Predictions that exist for frames without ground-truth annotations are counted as:
- False Positive (FP)


```bash
python show_ball.evaluate_show_ball \
    --predictions-path outputs/predictions.csv \
    --ground-truth-path resources/ground_truth.csv
```

| Argument | Description                                                               |
|-----------|---------------------------------------------------------------------------|
| `--predictions-path` | Path to the CSV file with predictions format <frame_no, ball_x, ball_y>.  |
| `--ground-truth-path` | Path to the CSV file with ground truth format <frame_no, ball_x, ball_y>. |
| `--start-frame` | First frame to process. By default 0.                                     |
| `--end-frame` | Last frame to process. By default processes until the last frame.         |


### Input Format

Both prediction and ground-truth CSV files must have the following format:

```csv
frame_no,ball_x,ball_y
0,123.4,456.7
1,125.1,454.2
2,127.0,451.8
```


| Column | Description |
|----------|-------------|
| `frame_no` | Frame number |
| `ball_x` | Ball x-coordinate in pixels |
| `ball_y` | Ball y-coordinate in pixels |


### Metrics

A ``DISTANCE_THRESHOLD = 10`` pixels is used to determine whether a predicted ball position
is considered a correct detection (True Positive).
This threshold can be adjusted depending on image resolution and annotation accuracy.

#### Precision

Fraction of predicted ball positions that are correct.

```text
Precision = TP / (TP + FP)
```

#### Recall

Fraction of ground-truth ball positions that are successfully detected.

```text
Recall = TP / (TP + FN)
```

#### F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

#### Recall @ N Pixels

Fraction of ground-truth frames whose prediction falls within a given localization error threshold.

Reported values:

- Recall @ 5 px
- Recall @ 10 px
- Recall @ 20 px

#### Mean Error

Average Euclidean localization error across matched detections.

#### Median Error

Median Euclidean localization error.

#### P95 Error

95th percentile of localization error. This metric highlights worst-case tracking performance while being less sensitive to outliers than the maximum error.

#### Missing Detection Rate

Fraction of ground-truth frames for which no valid prediction was produced.

```text
Missing Detection Rate = Missing Detections / Ground Truth Frames
```

### Dataset 
The dataset was generated from a single 8-minute video containing a ball under 
varying positions and motion conditions. The video was recorded at 60 frames per 
second (FPS), resulting in approximately 28,800 frames. Video frames were extracted
and annotated using the provided ball center coordinates (ball_x, ball_y). 
For object detection training, each point annotation was converted into a fixed-size
bounding box centered on the ball and stored in YOLO format. 
Frames without an annotation were included in the dataset with empty label files, 
allowing the model to learn both the presence and absence of the target object.

To create representative training, validation, and test sets while reducing temporal
bias, the video was divided into contiguous segments. The segment length was set to 
1,800 frames, corresponding to 30 seconds of video at 60 FPS. Segment length defines 
the number of consecutive frames grouped together before assignment to a dataset split. 
Using larger segments helps prevent highly similar neighboring frames from being 
distributed across different splits, thereby reducing information leakage and 
providing a more realistic evaluation of model performance. The resulting segments 
were randomly assigned to the training, validation, and test sets in a 70/15/15 ratio, 
ensuring that samples from the entire video were represented across all splits.

### Dataset structure

The dataset follows the standard YOLO object detection format. Images and labels 
are organized into separate directories for the training, validation, and test 
splits. Each image has a corresponding label file with the same filename. 
Label files contain the object class and normalized bounding box coordinates in 
the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

where all coordinates are normalized to the image dimensions. Since the dataset 
contains a single object class (the ball), all annotations use class ID 0. Frames 
in which the ball is not present are represented by an empty label file, indicating
that no target objects are visible in the image.
```
ball_yolo_dataset/
├── data.yaml
├── split_record.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
* data.yaml: YOLO dataset configuration file specifying dataset paths and class names.
* split_record.json: Record of the video segments assigned to each dataset split.
* images/train, images/val, images/test: Extracted video frames belonging to the training, validation, and test sets.
* labels/train, labels/val, labels/test: YOLO annotation files corresponding to each image.

# Further Improvements
- Improve the tensorRT conversion, currently the half precision in not working, I could also have implemented a 
EnginePredictor with cupy.
- 

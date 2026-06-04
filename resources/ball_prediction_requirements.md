# ⚽ Soccer Ball Position Prediction in Video Frames

The objective of this assignment is to design and implement a system in Python, that **predicts the position of a soccer ball** (its coordinates) in each frame of a given video.

The primary goal is to develop a **robust and modern detection approach**, using appropriate tools and methods suited for this task. Submissions will be evaluated not only on detection performance and accuracy, but also on **simplicity**, **engineering quality**, and **maintainability** of the code.

> 🔍 **Note**: The soccer ball is often small, fast-moving, and partially occluded. As such, we recommend approaches that can learn from data — for example, training a lightweight detection model or adapting an existing one.  
> Classical computer vision methods (e.g., Hough transforms, simple color filtering) are unlikely to be reliable for this task on their own.

---

## ✅ Final Evaluation Script

Your submission should include a script that, given the path to a 1080p video, performs the following:

1. **Writes predictions to CSV**  
   Generate a file called `part1.csv` with the following structure:
   - frame_no,ball_x,ball_y  
Each row should represent the predicted ball coordinates for frames where the ball has been detected.

2. **Visualizes predictions (optional)**  
Add an option to **visualize detections frame by frame**, similar to `show_ball_dataset.py`, but for all consecutive frames. This should be controlled by a command-line argument (e.g., `--show`).

---

## 📁 Dataset

- Video: `part1.mp4`  
- Labels: `part1.csv` (ground truth coordinates)  
- Use `show_ball_dataset.py` to visualize labeled frames for inspection and debugging.

---

## 💡 Implementation Guidance

- You **do not** need to run real-time detection, but faster inference is appreciated.
- It's acceptable to **downscale** frames before detection to speed up processing.
- You may skip detection in frames where the ball is not visible.
- The test video is from the **same game as the training set**, so you can focus on optimizing for this specific domain.

---

## 🛠 Suggested Tools

- PyTorch / TensorFlow (for model training/inference)  
- OpenCV (for video and image handling)
- Pandas

---

## 📦 Submission Recommendation (Optional)

- Push your project to a public GitHub repository or send it over email.  
- Include clear instructions for setup and usage in a `README.md`.


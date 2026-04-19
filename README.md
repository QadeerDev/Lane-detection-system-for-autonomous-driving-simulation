# Lane Detection for Autonomous Driving Simulation

> Draw your own ROI. Lock it. Watch the pipeline run.

A real-time lane detection system built with pure Computer Vision — no deep learning, no pretrained weights, no dataset required to get started. You draw the region of interest on the first frame, and the system focuses only on that area from that point on.

---

## What it does

- Opens the first frame (or a synthetic road if you have no video)
- Lets you draw a polygon ROI by clicking points on screen
- Locks the Canny + Hough pipeline to that exact region
- Detects left and right lane lines with temporal smoothing
- Estimates a steering angle in real-time and shows it on a HUD
- Press `R` at any time to redraw the ROI on a live frame

---

## Demo

### ROI Drawing
Click to place points → green fill shows your selected region → press ENTER to confirm.

```
Left Click   → Add vertex
Right Click  → Remove last vertex
ENTER        → Confirm ROI, start detection
R            → Reset / redraw ROI
ESC          → Quit
```

### Detection HUD
Once ROI is confirmed, you get:
- Lane fill overlay (orange-blue)
- Steering angle in degrees with a visual needle
- Direction indicator (STRAIGHT / TURN LEFT / TURN RIGHT)
- Live FPS counter

---

## Pipeline

```
Frame
  └── Grayscale + Gaussian Blur (5×5)
        └── Canny Edge Detection (50 / 150)
              └── ROI Mask (your drawn polygon)
                    └── Hough Line Transform (probabilistic)
                          └── Slope-based classification → left / right
                                └── Weighted average per side
                                      └── Temporal smoothing (8-frame history)
                                            └── Steering angle from lane center offset
```

| Stage | Method | Why |
|---|---|---|
| Noise removal | Gaussian blur (5×5) | Reduces false edges from camera grain |
| Edge detection | Canny (50/150) | Standard, fast, tunable |
| Region focus | Custom polygon mask | You decide what counts as road |
| Line detection | Probabilistic Hough | Faster than standard Hough for line segments |
| Lane classification | Slope + x-position filter | Separates left (negative slope) from right |
| Stability | 8-frame rolling average | Removes jitter between frames |
| Steering | Lane center vs frame center | Maps pixel offset to ±40° angle |

---

## Requirements

```
opencv-python==4.9.0.80
numpy==1.26.4
matplotlib==3.8.4
moviepy==1.0.3
```

Install:

```bash
pip install opencv-python numpy matplotlib moviepy
```

No GPU needed. Runs on any laptop with a webcam or a video file.

---

## Quickstart

```bash
git clone https://github.com/QadeerDev/lane-detection-sim
cd lane-detection-sim
pip install -r requirements.txt
python lane_detection.py
```

Choose a mode:

```
1 → Synthetic road demo   (no video needed — good for testing)
2 → Webcam live feed
3 → Video file (any dashcam .mp4)
4 → Static pipeline debug (matplotlib grid)
```

---

## Usage Notes

**Using your own video:**
Any front-facing dashcam footage works. When the ROI window opens, draw a trapezoid that covers the road ahead — bottom-wide, top-narrow, stopping before the horizon. Four points is usually enough.

**Redrawing mid-session:**
Press `R` while detection is running. The current frame freezes, the drawing window opens, and you can place a new polygon. Detection resumes with the new ROI and a reset smoothing history.

**Synthetic mode:**
Generates a perspective road with dashed center lines and solid lane markers. Useful for demoing without any footage. The road curves gently using a sine wave so you can see the steering angle respond.

---

## Project Structure

```
lane-detection-sim/
├── lane_detection.py      ← single-file implementation
├── requirements.txt
├── output/                ← saved .mp4 and .png results
└── README.md
```

---

## Output

Results are saved automatically to `output/`:

- `lane_detection_roi.mp4` — full video with overlay and HUD
- `roi_pipeline.png` — static debug view (edges / masked / result)

---

## Limitations

- Works best on roads with clearly painted lane markings
- Struggles in heavy rain, glare, or very worn roads
- Hough-based detection does not handle sharp curves well — a polynomial or sliding-window approach handles those better
- ROI needs to be redrawn if the camera angle changes significantly

---

## Roadmap

- [ ] Curved lane detection using sliding window + polynomial fit
- [ ] YOLOv8 integration for vehicle + pedestrian detection in the same frame
- [ ] Lane departure warning with audio alert
- [ ] Kalman filter to replace simple rolling average
- [ ] CULane / TuSimple benchmark evaluation

---

## Part of

**30-Day Computer Vision Portfolio Challenge** — building and shipping one CV project per day.

Follow the series on [LinkedIn](https://www.linkedin.com/in/qadeerjutt) | [GitHub](https://github.com/QadeerDev)

---

## License

MIT

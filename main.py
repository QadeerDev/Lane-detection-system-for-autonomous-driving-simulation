"""
╔══════════════════════════════════════════════════════════════════╗
║     LANE DETECTION — INTERACTIVE ROI + AUTONOMOUS DRIVING SIM   ║
║     Draw your own ROI → pipeline locks to it                    ║
╚══════════════════════════════════════════════════════════════════╝

CONTROLS (ROI Drawing):
  Left Click  → Add point
  Right Click → Remove last point
  ENTER       → Confirm ROI and start detection
  R           → Reset / redraw ROI
  ESC         → Quit

CONTROLS (Detection):
  R           → Re-draw ROI on current frame
  Q           → Quit
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
class Config:
    BLUR_KERNEL       = (5, 5)
    CANNY_LOW         = 50
    CANNY_HIGH        = 150
    HOUGH_RHO         = 1
    HOUGH_THETA       = np.pi / 180
    HOUGH_THRESHOLD   = 30
    HOUGH_MIN_LINE_LEN = 50
    HOUGH_MAX_LINE_GAP = 200
    MIN_SLOPE         = 0.3
    MAX_SLOPE         = 10.0
    LANE_COLOR        = (0, 255, 120)
    OVERLAY_COLOR     = (0, 180, 255)
    STEERING_COLOR    = (255, 80, 0)
    LINE_THICKNESS    = 8
    OVERLAY_ALPHA     = 0.25
    HISTORY_LEN       = 8


# ─────────────────────────────────────────────────────────────────
#  INTERACTIVE ROI SELECTOR
# ─────────────────────────────────────────────────────────────────
class ROISelector:
    """
    Opens a window on a snapshot frame.
    User clicks to place polygon vertices.
    ENTER confirms. R resets. ESC cancels.
    """

    def __init__(self):
        self.points   = []
        self.done     = False
        self.snapshot = None

    # ── Mouse callback ─────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
        self._redraw()

    # ── Redraw the instruction frame ──────────────────────────
    def _redraw(self):
        display = self.snapshot.copy()
        h, w    = display.shape[:2]

        # ── dim overlay ──
        overlay = np.zeros_like(display)

        # ── draw filled polygon preview ──
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 120))
            display = cv2.addWeighted(display, 1.0, overlay, 0.18, 0)
            cv2.polylines(display, [pts], isClosed=True,
                          color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

        # ── draw dots + connecting lines ──
        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 7, (0, 255, 120), -1)
            cv2.circle(display, pt, 9, (255, 255, 255), 1)
            if i > 0:
                cv2.line(display, self.points[i-1], pt,
                         (0, 230, 100), 2, cv2.LINE_AA)
            # index label
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── instruction panel ──
        panel_h = 110
        panel   = np.zeros((panel_h, w, 3), dtype=np.uint8)
        panel[:] = (18, 18, 18)
        cv2.line(panel, (0, 0), (w, 0), (0, 255, 120), 2)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        lines_txt = [
            (f"Points placed: {len(self.points)}  (need ≥ 3)",
             (0, 255, 120) if len(self.points) >= 3 else (100, 100, 100)),
            ("LEFT CLICK = add point   |   RIGHT CLICK = undo last",
             (180, 180, 180)),
            ("ENTER = confirm ROI   |   R = reset   |   ESC = quit",
             (180, 180, 180)),
        ]
        for i, (txt, col) in enumerate(lines_txt):
            cv2.putText(panel, txt, (20, 28 + i * 30), font, 0.58, col, 1)

        combined = np.vstack([display, panel])
        cv2.imshow("Draw ROI  —  Lane Detection", combined)

    # ── Public: run the selector on a frame ───────────────────
    def select(self, frame: np.ndarray):
        """
        Show frame, let user draw polygon.
        Returns np.array of points or None if cancelled.
        """
        self.points   = []
        self.done     = False
        self.snapshot = frame.copy()

        cv2.namedWindow("Draw ROI  —  Lane Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Draw ROI  —  Lane Detection",
                         min(frame.shape[1], 1400),
                         min(frame.shape[0] + 120, 900))
        cv2.setMouseCallback("Draw ROI  —  Lane Detection", self._mouse_cb)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 13:   # ENTER
                if len(self.points) >= 3:
                    self.done = True
                    cv2.destroyWindow("Draw ROI  —  Lane Detection")
                    return np.array(self.points, dtype=np.int32)
                else:
                    # flash warning
                    tmp = self.snapshot.copy()
                    cv2.putText(tmp, "Need at least 3 points!",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)
                    cv2.imshow("Draw ROI  —  Lane Detection", tmp)

            elif key == ord('r') or key == ord('R'):
                self.points = []
                self._redraw()

            elif key == 27:  # ESC
                cv2.destroyWindow("Draw ROI  —  Lane Detection")
                return None


# ─────────────────────────────────────────────────────────────────
#  CORE LANE DETECTOR  (ROI comes from outside now)
# ─────────────────────────────────────────────────────────────────
class LaneDetector:
    def __init__(self, cfg: Config = Config()):
        self.cfg               = cfg
        self.roi_points        = None   # np.array of polygon pts set externally
        self.left_fit_history  = []
        self.right_fit_history = []

    def set_roi(self, points: np.ndarray):
        """Receive drawn polygon from ROISelector."""
        self.roi_points = points
        print(f"[ROI] Locked to {len(points)} points: {points.tolist()}")

    # ── Preprocessing ──────────────────────────────────────────
    def preprocess(self, frame):
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.cfg.BLUR_KERNEL, 0)
        return cv2.Canny(blurred, self.cfg.CANNY_LOW, self.cfg.CANNY_HIGH)

    # ── ROI Mask using drawn polygon ───────────────────────────
    def apply_roi(self, edges):
        mask = np.zeros_like(edges)
        if self.roi_points is not None and len(self.roi_points) >= 3:
            cv2.fillPoly(mask, [self.roi_points], 255)
        else:
            # Fallback: full frame
            mask[:] = 255
        return cv2.bitwise_and(edges, mask)

    # ── Hough Lines ───────────────────────────────────────────
    def detect_lines(self, masked):
        return cv2.HoughLinesP(
            masked,
            rho=self.cfg.HOUGH_RHO,
            theta=self.cfg.HOUGH_THETA,
            threshold=self.cfg.HOUGH_THRESHOLD,
            minLineLength=self.cfg.HOUGH_MIN_LINE_LEN,
            maxLineGap=self.cfg.HOUGH_MAX_LINE_GAP
        )

    # ── Classify lines into left / right ──────────────────────
    def classify_lines(self, lines, shape):
        h, w = shape[:2]
        left_lines, right_lines = [], []
        if lines is None:
            return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.cfg.MIN_SLOPE or abs(slope) > self.cfg.MAX_SLOPE:
                continue
            intercept = y1 - slope * x1
            length    = np.hypot(x2 - x1, y2 - y1)
            if slope < 0 and x1 < w * 0.6 and x2 < w * 0.6:
                left_lines.append((slope, intercept, length))
            elif slope > 0 and x1 > w * 0.4 and x2 > w * 0.4:
                right_lines.append((slope, intercept, length))
        return self._wavg(left_lines), self._wavg(right_lines)

    def _wavg(self, lines):
        if not lines:
            return None
        slopes, intercepts, lengths = zip(*lines)
        total = sum(lengths)
        return (sum(s*l for s,l in zip(slopes,lengths))/total,
                sum(i*l for i,l in zip(intercepts,lengths))/total)

    # ── Temporal smoothing ────────────────────────────────────
    def smooth(self, lf, rf):
        if lf:
            self.left_fit_history.append(lf)
            if len(self.left_fit_history) > self.cfg.HISTORY_LEN:
                self.left_fit_history.pop(0)
        if rf:
            self.right_fit_history.append(rf)
            if len(self.right_fit_history) > self.cfg.HISTORY_LEN:
                self.right_fit_history.pop(0)
        sl = np.mean(self.left_fit_history,  axis=0) if self.left_fit_history  else None
        sr = np.mean(self.right_fit_history, axis=0) if self.right_fit_history else None
        return sl, sr

    # ── Fit → pixel coords ────────────────────────────────────
    def _roi_y_range(self, h):
        """Derive y_top / y_bottom from the drawn ROI polygon."""
        if self.roi_points is not None:
            ys = self.roi_points[:, 1]
            return int(ys.min()), int(ys.max())
        return int(h * 0.60), h - 5

    def fit_to_coords(self, fit, y_top, y_bottom):
        if fit is None:
            return None
        slope, intercept = fit
        try:
            return (int((y_bottom - intercept) / slope), y_bottom), \
                   (int((y_top    - intercept) / slope), y_top)
        except ZeroDivisionError:
            return None

    # ── Steering angle ────────────────────────────────────────
    def compute_steering(self, lc, rc, w):
        cx = w // 2
        pts = []
        if lc: pts.append(lc[1][0])
        if rc: pts.append(rc[1][0])
        if len(pts) == 2:
            lane_center = (pts[0] + pts[1]) // 2
        elif len(pts) == 1:
            lane_center = pts[0]
        else:
            return 0.0, cx
        angle = np.clip(((lane_center - cx) / (w * 0.5)) * 40, -40, 40)
        return angle, lane_center

    # ── Draw ROI boundary (always visible) ────────────────────
    def draw_roi_boundary(self, frame):
        if self.roi_points is not None and len(self.roi_points) >= 3:
            cv2.polylines(frame, [self.roi_points], isClosed=True,
                          color=(0, 200, 255), thickness=1,
                          lineType=cv2.LINE_AA)
            # Corner dots
            for pt in self.roi_points:
                cv2.circle(frame, tuple(pt), 4, (0, 200, 255), -1)
        return frame

    # ── Draw lane overlay ─────────────────────────────────────
    def draw_overlay(self, frame, lc, rc):
        overlay = frame.copy()
        if lc and rc:
            pts = np.array([lc[0], lc[1], rc[1], rc[0]], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], self.cfg.OVERLAY_COLOR)
            frame = cv2.addWeighted(overlay, self.cfg.OVERLAY_ALPHA,
                                    frame, 1 - self.cfg.OVERLAY_ALPHA, 0)
        for coords in [lc, rc]:
            if coords:
                cv2.line(frame, coords[0], coords[1],
                         self.cfg.LANE_COLOR, self.cfg.LINE_THICKNESS,
                         cv2.LINE_AA)
        return frame

    # ── HUD ───────────────────────────────────────────────────
    def draw_hud(self, frame, angle, lane_center, fps=0):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (360, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (360, 150), (0, 255, 120), 1)

        font      = cv2.FONT_HERSHEY_SIMPLEX
        direction = "STRAIGHT" if abs(angle) < 5 else \
                    ("← TURN LEFT" if angle < 0 else "TURN RIGHT →")
        col_dir   = (0, 255, 120) if abs(angle) < 5 else (0, 120, 255)

        cv2.putText(frame, f"Steering : {angle:+.1f} deg", (20, 45),  font, 0.62, (255,255,255), 2)
        cv2.putText(frame, f"Direction: {direction}",      (20, 80),  font, 0.62, col_dir, 2)
        cv2.putText(frame, f"Lane Ctr : {lane_center}px",  (20, 115), font, 0.58, (180,180,180), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}",              (290,145), font, 0.50, (100,200,100), 1)
        cv2.putText(frame, "R=redraw ROI  Q=quit",         (10, h-10),font, 0.45, (80, 80, 80), 1)

        # Steering wheel
        cx, cy, r = w - 80, 80, 50
        cv2.circle(frame, (cx, cy), r,   (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), r,   (0, 255, 120), 2)
        nx = int(cx + r * 0.7 * np.sin(np.radians(angle)))
        ny = int(cy - r * 0.7 * np.cos(np.radians(angle)))
        cv2.line(frame, (cx, cy), (nx, ny), self.cfg.STEERING_COLOR, 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        return frame

    # ── MAIN process_frame ────────────────────────────────────
    def process_frame(self, frame, fps=0):
        h, w     = frame.shape[:2]
        y_top, y_bottom = self._roi_y_range(h)

        edges    = self.preprocess(frame)
        masked   = self.apply_roi(edges)
        lines    = self.detect_lines(masked)
        lf, rf   = self.classify_lines(lines, frame.shape)
        lf, rf   = self.smooth(lf, rf)
        lc       = self.fit_to_coords(lf, y_top, y_bottom)
        rc       = self.fit_to_coords(rf, y_top, y_bottom)
        angle, lane_center = self.compute_steering(lc, rc, w)

        result = self.draw_roi_boundary(frame.copy())
        result = self.draw_overlay(result, lc, rc)
        result = self.draw_hud(result, angle, lane_center, fps)
        return result, edges, masked


# ─────────────────────────────────────────────────────────────────
#  SYNTHETIC ROAD GENERATOR  (zero-dataset demo)
# ─────────────────────────────────────────────────────────────────
def generate_synthetic_road(width=1280, height=720, curve=0.0):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height // 2):
        v = int(60 + (y / (height // 2)) * 40)
        img[y] = [v + 20, v + 10, v]
    road_color = np.array([60, 65, 70])
    for y in range(height // 2, height):
        f = (y - height // 2) / (height // 2)
        img[y] = (road_color * (0.6 + 0.4 * f)).astype(np.uint8)

    vx = width // 2 + int(curve * 100)
    vy = height // 2 + 10
    lb = (int(width * 0.20), height - 10)
    rb = (int(width * 0.80), height - 10)
    lt = (vx - 30, vy)
    rt = (vx + 30, vy)

    for i in range(12):
        t0 = i / 12;  t1 = (i + 0.4) / 12
        if i % 2 == 0:
            cx0 = (int(lb[0]+(vx-lb[0])*t0) + int(rb[0]+(vx-rb[0])*t0)) // 2
            cx1 = (int(lb[0]+(vx-lb[0])*t1) + int(rb[0]+(vx-rb[0])*t1)) // 2
            ly0 = int(lb[1] + (vy - lb[1]) * t0)
            ly1 = int(lb[1] + (vy - lb[1]) * t1)
            cv2.line(img, (cx0, ly0), (cx1, ly1), (200, 200, 200), 3)
    cv2.line(img, lb, lt, (240, 240, 240), 5)
    cv2.line(img, rb, rt, (240, 240, 240), 5)
    return img


# ─────────────────────────────────────────────────────────────────
#  RUNNER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def run(source=None, synthetic=False):
    """
    Universal runner.
    source = 0 (webcam) | "path/to/video.mp4" | None (synthetic)
    """
    selector = ROISelector()
    detector = LaneDetector()

    # ── Grab first frame for ROI drawing ──────────────────────
    if synthetic or source is None:
        snapshot = generate_synthetic_road()
        use_synthetic = True
    else:
        cap_tmp = cv2.VideoCapture(source)
        ret, snapshot = cap_tmp.read()
        cap_tmp.release()
        if not ret:
            print(f"[ERROR] Cannot read from: {source}")
            return
        use_synthetic = False

    # ── ROI Selection ─────────────────────────────────────────
    print("\n[ROI] Draw your region of interest on the frame.")
    print("      Left-click to add points, Right-click to undo.")
    print("      Press ENTER when done, R to reset, ESC to cancel.\n")

    roi_pts = selector.select(snapshot)
    if roi_pts is None:
        print("[INFO] ROI selection cancelled.")
        return

    detector.set_roi(roi_pts)
    os.makedirs("output", exist_ok=True)

    # ── Detection loop ────────────────────────────────────────
    cap = None
    if not use_synthetic:
        cap = cv2.VideoCapture(source)

    writer = cv2.VideoWriter(
        "output/lane_detection_roi.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (snapshot.shape[1], snapshot.shape[0])
    )

    print("[INFO] Detection running — R=redraw ROI, Q=quit\n")
    frame_idx = 0
    prev_t    = time.time()

    while True:
        # ── Get next frame ────────────────────────────────────
        if use_synthetic:
            curve = np.sin(frame_idx / 50) * 0.35
            frame = generate_synthetic_road(curve=curve)
            frame_idx += 1
            if frame_idx > 900:   # ~30s
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break

        cur_t = time.time()
        fps   = 1.0 / max(cur_t - prev_t, 1e-6)
        prev_t = cur_t

        result, edges, masked = detector.process_frame(frame, fps)
        writer.write(result)
        cv2.imshow("Lane Detection — Interactive ROI", result)

        key = cv2.waitKey(1 if not use_synthetic else 33) & 0xFF

        if key == ord('q') or key == 27:
            print("[INFO] Quit.")
            break

        elif key == ord('r') or key == ord('R'):
            # ── Re-draw ROI on current frame ──────────────────
            print("[ROI] Re-drawing ROI...")
            cv2.destroyWindow("Lane Detection — Interactive ROI")
            new_pts = selector.select(frame)
            if new_pts is not None:
                detector.set_roi(new_pts)
                # Reset history so old fits don't bleed through
                detector.left_fit_history  = []
                detector.right_fit_history = []
                print("[ROI] ROI updated.")

    if cap:
        cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Saved → output/lane_detection_roi.mp4")


def static_test():
    """Single-frame debug view with drawn ROI."""
    selector = ROISelector()
    detector = LaneDetector()

    snapshot = generate_synthetic_road(curve=0.15)
    print("[ROI] Draw ROI on the test frame, then press ENTER.")
    roi_pts  = selector.select(snapshot)
    if roi_pts is None:
        return
    detector.set_roi(roi_pts)

    result, edges, masked = detector.process_frame(snapshot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0d0d0d')
    titles = ["Canny Edges", "ROI Masked Edges", "Final Output"]
    imgs   = [edges, masked, cv2.cvtColor(result, cv2.COLOR_BGR2RGB)]
    cmaps  = ['gray', 'gray', None]
    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11)
        ax.axis('off')
    plt.suptitle("Lane Detection — Interactive ROI Pipeline", color='white', fontsize=13)
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/roi_pipeline.png", dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    plt.show()
    print("[INFO] Saved → output/roi_pipeline.png")


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  LANE DETECTION — INTERACTIVE ROI EDITION")
    print("=" * 60)
    print("\n  1 → Synthetic road (no dataset needed)")
    print("  2 → Webcam live feed")
    print("  3 → Video file")
    print("  4 → Static pipeline debug (matplotlib)")

    choice = input("\nEnter [1-4]: ").strip()

    if choice == "1":
        run(synthetic=True)
    elif choice == "2":
        run(source=0)
    elif choice == "3":
        path = input("Video path: ").strip()
        run(source=path)
    elif choice == "4":
        static_test()
    else:
        run(synthetic=True)
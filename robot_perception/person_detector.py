"""Person detector node using YOLO11.

Publishes detected target position as geometry_msgs/Point:
  x : horizontal offset from image centre, normalised to [-1, 1]
        positive → target is to the right
  y : distance proxy  =  (bbox_height / frame_height) - TARGET_HEIGHT_RATIO
        negative → target is far (small bbox)
        positive → target is close (large bbox)
  z : detection confidence [0, 1]

Topic interface
---------------
Subscribed : /camera/image_raw  (sensor_msgs/Image)
Published  : /person_position   (geometry_msgs/Point)   ← consumed by tracking_controller
             /camera/annotated  (sensor_msgs/Image)      ← debug visualisation

Architecture handling
---------------------
* ARM (aarch64 / armv7l) — Raspberry Pi
    ONNX 파일을 cv2.dnn으로 직접 추론 (ultralytics/torch/onnxruntime 완전 우회).
    → SIGILL 없음, OpenCV는 Pi용으로 제대로 컴파일돼 있음.
* x86_64 — development workstation
    Tries .pt first (GPU / CPU, fast for dev), then NCNN as fallback.

Debug mode
----------
TARGET_CLASSES contains both *person* (COCO 0) and *sports ball* (COCO 32).
All detections are treated as a single "person" target for the controller.
The highest-confidence detection is forwarded each frame.
"""

import platform
import os
from collections import namedtuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

# ── COCO class IDs treated as trackable targets ─────────────────────────────
TARGET_CLASSES = {
    0: 'person',
    32: 'sports ball',
}

# Desired bounding-box height as a fraction of frame height.
# Deviation from this ratio becomes error_y.
TARGET_HEIGHT_RATIO = 0.40

# Inference input resolution — smaller = faster (try 256 if Pi is still slow).
INFER_SIZE = 320

# Run inference once every N frames; annotated image is published every frame.
# Raise this if the model is too slow (e.g. 4 on Pi).
INFER_EVERY = 3

Detection = namedtuple('Detection', ['x1', 'y1', 'x2', 'y2', 'cls_id', 'conf'])

NUM_CLASSES = 80  # COCO


class _CvDnnDetector:
    """YOLOv8 ONNX inference via cv2.dnn — no torch/onnxruntime required.

    Works on any platform where OpenCV is available (incl. Raspberry Pi ARM).
    Expected ONNX output shape: (1, 84, N) where 84 = 4 bbox + 80 classes.
    """

    def __init__(self, onnx_path: str, infer_size: int):
        self._infer_size = infer_size
        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self, frame: np.ndarray, conf_thresh: float,
                iou_thresh: float, target_classes: list) -> list:
        """Return list of Detection(x1,y1,x2,y2,cls_id,conf) in frame coords."""
        orig_h, orig_w = frame.shape[:2]
        sz = self._infer_size

        # Letterbox resize to keep aspect ratio
        scale = min(sz / orig_w, sz / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (sz - new_w) // 2
        pad_y = (sz - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1.0 / 255.0,
            size=(sz, sz), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        raw = self._net.forward()  # (1, 84, N)

        # Transpose to (N, 84)
        preds = raw[0].T  # (N, 84)

        dets = []
        boxes_raw, scores_raw, cls_ids_raw = [], [], []

        for row in preds:
            cx, cy, bw, bh = row[:4]
            class_scores = row[4:4 + NUM_CLASSES]
            cls_id = int(np.argmax(class_scores))
            if cls_id not in target_classes:
                continue
            conf = float(class_scores[cls_id])
            if conf < conf_thresh:
                continue
            boxes_raw.append([float(cx), float(cy), float(bw), float(bh)])
            scores_raw.append(conf)
            cls_ids_raw.append(cls_id)

        if not boxes_raw:
            return []

        # NMS (cv2 wants x,y,w,h format)
        boxes_xywh = [[b[0] - b[2] / 2, b[1] - b[3] / 2, b[2], b[3]]
                      for b in boxes_raw]
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores_raw, conf_thresh, iou_thresh
        )
        if len(indices) == 0:
            return []

        for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
            cx, cy, bw, bh = boxes_raw[i]
            # Convert from letterboxed input coords back to original frame coords
            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            # Clamp
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))
            dets.append(Detection(x1, y1, x2, y2, cls_ids_raw[i], scores_raw[i]))

        return dets


def _detect_model_path(declared_path: str, force_ncnn: bool = False) -> tuple:
    """Return (resolved_path, mode) where mode is 'ncnn', 'onnx', or 'pt'.

    Priority on ARM        : onnx  → ncnn dir  → .pt file
    Priority on x86        : .pt file  → ncnn dir  → onnx
    force_ncnn=True (any)  : ncnn dir only (for cross-arch testing)
    """
    arch = platform.machine()
    is_arm = arch in ('aarch64', 'armv7l')

    if declared_path.endswith('_ncnn_model') or os.path.isdir(declared_path):
        ncnn_dir = declared_path
        base = declared_path.rstrip('/').removesuffix('_ncnn_model')
    else:
        base = os.path.splitext(declared_path)[0]
        ncnn_dir = base + '_ncnn_model'

    pt_file = base + '.pt'
    onnx_file = base + '.onnx'
    ncnn_ok = os.path.isdir(ncnn_dir)
    pt_ok = os.path.isfile(pt_file)
    onnx_ok = os.path.isfile(onnx_file)

    if force_ncnn:
        if ncnn_ok:
            return ncnn_dir, 'ncnn'
        raise FileNotFoundError(
            f"force_ncnn=True but NCNN dir not found: {ncnn_dir}"
        )

    if is_arm:
        # ARM (Pi): onnx most stable, then ncnn, then pt
        if onnx_ok:
            return onnx_file, 'onnx'
        if ncnn_ok:
            return ncnn_dir, 'ncnn'
        if pt_ok:
            return pt_file, 'pt'
    else:
        # x86: pt fastest with GPU/CPU, then ncnn, then onnx
        if pt_ok:
            return pt_file, 'pt'
        if ncnn_ok:
            return ncnn_dir, 'ncnn'
        if onnx_ok:
            return onnx_file, 'onnx'

    raise FileNotFoundError(
        f"No usable model found.\n"
        f"  ONNX file : {onnx_file} (exists={onnx_ok})\n"
        f"  NCNN dir  : {ncnn_dir} (exists={ncnn_ok})\n"
        f"  .pt file  : {pt_file}  (exists={pt_ok})"
    )


class PersonDetector(Node):
    """ROS 2 node that detects persons (and balls in debug mode) via YOLO11."""

    def __init__(self):
        super().__init__('person_detector')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('model_path', 'yolov8n_ncnn_model')
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('target_height_ratio', TARGET_HEIGHT_RATIO)
        self.declare_parameter('infer_size', INFER_SIZE)
        self.declare_parameter('infer_every', INFER_EVERY)
        self.declare_parameter('debug_track_all_targets', True)
        self.declare_parameter('force_ncnn', False)

        raw_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self._conf = self.get_parameter('conf_threshold').get_parameter_value().double_value
        self._iou = self.get_parameter('iou_threshold').get_parameter_value().double_value
        self._target_h_ratio = self.get_parameter(
            'target_height_ratio').get_parameter_value().double_value
        self._infer_size = self.get_parameter('infer_size').get_parameter_value().integer_value
        self._infer_every = self.get_parameter('infer_every').get_parameter_value().integer_value
        debug_mode = self.get_parameter(
            'debug_track_all_targets').get_parameter_value().bool_value
        force_ncnn = self.get_parameter('force_ncnn').get_parameter_value().bool_value

        self._target_classes = TARGET_CLASSES if debug_mode else {0: 'person'}

        self.get_logger().info(f'Host architecture: {platform.machine()}')

        # ── Load model ───────────────────────────────────────────────────────
        try:
            model_path, mode = _detect_model_path(raw_model_path, force_ncnn=force_ncnn)
        except FileNotFoundError as exc:
            self.get_logger().fatal(str(exc))
            raise SystemExit(1) from exc

        self.get_logger().info(f'Loading YOLO model [{mode.upper()}]: {model_path}')

        arch = platform.machine()
        is_arm = arch in ('aarch64', 'armv7l')

        if mode == 'onnx' and is_arm:
            # ARM: cv2.dnn으로 직접 추론 (ultralytics/torch 완전 우회)
            self._cv_detector = _CvDnnDetector(model_path, self._infer_size)
            self._model = None
            self.get_logger().info(
                f'Using cv2.dnn ONNX backend (ARM) | '
                f'targets={list(self._target_classes.values())} '
                f'| conf>={self._conf} | imgsz={self._infer_size}'
            )
        else:
            # x86 or non-ONNX: ultralytics
            self._cv_detector = None
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self.get_logger().info('Warming up model...')
            self._model.predict(
                source=np.zeros((self._infer_size, self._infer_size, 3), dtype=np.uint8),
                imgsz=self._infer_size,
                verbose=False,
            )
            self.get_logger().info(
                f'YOLO ready | targets={list(self._target_classes.values())} '
                f'| conf>={self._conf} | imgsz={self._infer_size}'
            )

        # ── State ────────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._frame_count = 0
        self._infer_count = 0
        self._det_count = 0
        self._last_dets: list = []   # cached Detection namedtuples
        self._last_best = None       # cached (error_x, error_y, conf) or None

        # ── ROS interfaces ───────────────────────────────────────────────────
        self._img_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)
        self._pos_pub = self.create_publisher(Point, 'person_position', 10)
        self._ann_pub = self.create_publisher(Image, '/camera/annotated', 10)

    # ── Image callback ────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # Run inference every N frames; use cached results otherwise.
        self._frame_count += 1
        if self._frame_count % self._infer_every == 0:
            self._infer_count += 1
            try:
                if self._cv_detector is not None:
                    # ARM cv2.dnn path
                    dets = self._cv_detector.predict(
                        frame, self._conf, self._iou,
                        list(self._target_classes.keys())
                    )
                    self._last_dets, self._last_best = \
                        self._dets_to_best(dets, w, h)
                else:
                    # ultralytics path
                    results = self._model.predict(
                        source=frame,
                        imgsz=self._infer_size,
                        conf=self._conf,
                        iou=self._iou,
                        classes=list(self._target_classes.keys()),
                        verbose=False,
                    )
                    self._last_dets, self._last_best = \
                        self._parse_results(results, w, h)
                if self._last_dets:
                    self._det_count += 1
                    best = self._last_best
                    self.get_logger().info(
                        f'[{self._frame_count}] Detected {len(self._last_dets)} target(s) | '
                        f'best: ex={best[0]:+.2f} ey={best[1]:+.2f} conf={best[2]:.2f}'
                    )
                else:
                    self.get_logger().debug(
                        f'[{self._frame_count}] No targets detected'
                    )
            except Exception as exc:
                self.get_logger().warn(f'Inference error: {exc}')

            if self._last_best is not None:
                pt = Point()
                ex, ey, ec = self._last_best
                pt.x, pt.y, pt.z = float(ex), float(ey), float(ec)
                self._pos_pub.publish(pt)

        # Log stats every 30 frames
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Stats | frames={self._frame_count} '
                f'infers={self._infer_count} '
                f'frames_with_det={self._det_count} '
                f'resolution={w}x{h}'
            )

        # Always publish annotated image (with cached boxes when skipping)
        ann = self._annotate(frame, self._last_dets, w, h, self._last_best)
        ann_msg = self._bridge.cv2_to_imgmsg(ann, encoding='bgr8')
        ann_msg.header = msg.header
        self._ann_pub.publish(ann_msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dets_to_best(self, dets: list, frame_w: int, frame_h: int):
        """Convert Detection list to (dets, best_tuple) — shared by both backends."""
        best_conf = -1.0
        best = None
        for d in dets:
            cx = (d.x1 + d.x2) / 2.0
            bh = d.y2 - d.y1
            error_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
            error_y = (bh / frame_h) - self._target_h_ratio
            if d.conf > best_conf:
                best_conf = d.conf
                best = (error_x, error_y, d.conf)
        return dets, best

    def _parse_results(self, results, frame_w: int, frame_h: int):
        """Return (list[Detection], best_tuple) from ultralytics results."""
        dets = []
        best_conf = -1.0
        best = None

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in self._target_classes:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append(Detection(x1, y1, x2, y2, cls_id, conf))

                cx = (x1 + x2) / 2.0
                bh = y2 - y1
                error_x = (cx - frame_w / 2.0) / (frame_w / 2.0)
                error_y = (bh / frame_h) - self._target_h_ratio

                if conf > best_conf:
                    best_conf = conf
                    best = (error_x, error_y, conf)

        return dets, best

    def _annotate(
        self, frame: np.ndarray, dets: list, frame_w: int, frame_h: int, best
    ) -> np.ndarray:
        ann = frame.copy()

        for d in dets:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            label = f"{self._target_classes.get(d.cls_id, str(d.cls_id))} {d.conf:.2f}"
            colour = (0, 255, 0) if d.cls_id == 0 else (0, 165, 255)
            cv2.rectangle(ann, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(ann, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        # Cross-hair at frame centre
        cx, cy = frame_w // 2, frame_h // 2
        cv2.line(ann, (cx - 14, cy), (cx + 14, cy), (180, 180, 180), 1)
        cv2.line(ann, (cx, cy - 14), (cx, cy + 14), (180, 180, 180), 1)

        if best is not None:
            ex, ey, ec = best
            txt = f"ex={ex:+.2f}  ey={ey:+.2f}  conf={ec:.2f}"
            cv2.putText(ann, txt, (6, frame_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(ann, "SEARCHING...", (6, frame_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        return ann


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

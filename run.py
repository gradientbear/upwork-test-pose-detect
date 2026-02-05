#!/usr/bin/env python3
"""
ML Technical Interview Task (Python)

Command:
  python run.py --input video.mp4 --output out.mp4 --json out.json

Outputs:
  1) Annotated video with COCO-17 skeleton overlay
  2) JSON with 17 COCO-style keypoints per frame (x,y in original pixel coords, score in [0,1])

Implementation notes:
- MediaPipe 0.10.30+ removed the legacy `mp.solutions` API from the default PyPI build on some platforms.
  This script uses the MediaPipe **Tasks** API (`mp.tasks.vision.PoseLandmarker`) instead.
- Outputs are still COCO-17: we map a subset of the 33 BlazePose landmarks to COCO-17.
- For speed, frames are optionally downscaled for inference.

See README.md for details.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# MediaPipe import can be slow; keep it after stdlib imports.
import mediapipe as mp


# Public MediaPipe model bundle URLs (".task" files) for PoseLandmarker.
# These are referenced in official demos and samples.
MODEL_URLS: Dict[str, str] = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


COCO_17 = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# MediaPipe Pose landmark indices for the subset that matches COCO-17.
# Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_IDX = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# A standard COCO-ish skeleton; any consistent edge set is acceptable.
# Each edge refers to COCO_17 indices.
SKELETON_EDGES = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# torso
    (11, 12),        # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _default_model_path(variant: str) -> str:
    # Keep models in-repo so reruns are fast.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"pose_landmarker_{variant}.task")


def _download_model_if_needed(model_path: str, variant: str) -> str:
    """Ensures the PoseLandmarker .task file exists at model_path.

    If missing, downloads it from the public MediaPipe model bucket.
    """
    if os.path.isfile(model_path):
        return model_path
    if variant not in MODEL_URLS:
        raise ValueError(f"Unknown model variant '{variant}'. Choose from: {sorted(MODEL_URLS.keys())}")

    _ensure_parent_dir(model_path)
    url = MODEL_URLS[variant]
    try:
        print(f"[info] Downloading model '{variant}' -> {model_path}")
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download the MediaPipe PoseLandmarker model. "
            "Either ensure you have internet access for the first run, "
            "or download the .task file manually and pass --model-path. "
            f"(url={url})"
        ) from e
    return model_path


def _resize_for_inference(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    """
    Returns resized frame for inference.
    If max_side <= 0, returns original.
    """
    if max_side is None or max_side <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr
    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def _extract_coco17_from_pose_landmarker(
    results,
    orig_w: int,
    orig_h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map MediaPipe Pose landmarks -> COCO-17.
    (x,y) returned in original pixel coordinates.
    score returned in [0,1]. In the Tasks API, per-landmark confidence may appear in
    `visibility` and/or `presence`; we use max(visibility, presence).
    """
    xy = np.zeros((17, 2), dtype=np.float32)
    sc = np.zeros((17,), dtype=np.float32)

    # PoseLandmarkerResult.pose_landmarks is a list[ list[NormalizedLandmark] ].
    if not getattr(results, "pose_landmarks", None):
        return xy, sc

    # Single person: take the first pose.
    lms = results.pose_landmarks[0]
    for i, name in enumerate(COCO_17):
        lm = lms[MP_IDX[name]]
        # MediaPipe gives normalized coordinates (can be slightly outside [0,1] at times).
        x_orig = float(lm.x) * orig_w
        y_orig = float(lm.y) * orig_h

        # Clamp to image bounds for output stability
        x_orig = max(0.0, min(orig_w - 1.0, x_orig))
        y_orig = max(0.0, min(orig_h - 1.0, y_orig))

        xy[i, 0] = x_orig
        xy[i, 1] = y_orig
        # In the Tasks API, per-landmark confidence may appear in `visibility` and/or `presence`.
        v = getattr(lm, "visibility", None)
        p = getattr(lm, "presence", None)
        v = 0.0 if v is None else float(v)
        p = 0.0 if p is None else float(p)
        sc[i] = max(v, p)

    return xy, sc


def _draw_overlay(
    frame_bgr: np.ndarray,
    xy: np.ndarray,
    sc: np.ndarray,
    min_score: float,
    show_text: Optional[str] = None,
) -> np.ndarray:
    out = frame_bgr.copy()

    # Draw skeleton
    for a, b in SKELETON_EDGES:
        if sc[a] >= min_score and sc[b] >= min_score:
            ax, ay = int(round(xy[a, 0])), int(round(xy[a, 1]))
            bx, by = int(round(xy[b, 0])), int(round(xy[b, 1]))
            cv2.line(out, (ax, ay), (bx, by), (0, 255, 0), 2, lineType=cv2.LINE_AA)

    # Draw points
    for i in range(17):
        if sc[i] >= min_score:
            x, y = int(round(xy[i, 0])), int(round(xy[i, 1]))
            cv2.circle(out, (x, y), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    if show_text:
        cv2.putText(
            out,
            show_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            show_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def process_video(
    input_path: str,
    output_video_path: str,
    output_json_path: str,
    max_side: int = 640,
    min_score: float = 0.2,
    model_complexity: int = 1,
    smooth_landmarks: bool = True,
    show_perf_text: bool = True,
    model_path: Optional[str] = None,
    download_model: bool = True,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0  # reasonable fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        # Try to read one frame to infer dimensions
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to read any frames from the input video.")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _ensure_parent_dir(output_video_path)
    _ensure_parent_dir(output_json_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_video_path}")

    # Map the old --model-complexity flag to the Tasks API model variants.
    variant = {0: "lite", 1: "full", 2: "heavy"}.get(int(model_complexity), "full")
    if model_path is None:
        model_path = _default_model_path(variant)
    if download_model:
        model_path = _download_model_if_needed(model_path, variant)
    elif not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Either enable auto-download or pass --model-path to an existing .task."
        )

    # MediaPipe Tasks setup (VIDEO mode uses tracking across frames).
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Note: The legacy `smooth_landmarks` flag doesn't directly exist in Tasks.
    # VIDEO mode tracking provides temporal stability; keep the flag for CLI compatibility.
    _ = smooth_landmarks

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    frames_json: List[dict] = []

    frame_idx = 0
    ema_ms: Optional[float] = None
    ema_alpha = 0.1

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # Tasks VIDEO mode requires a monotonically increasing timestamp (ms).
                timestamp_ms = int(frame_idx * 1000.0 / fps)

                t0 = time.perf_counter()
                infer_bgr = _resize_for_inference(frame_bgr, max_side=max_side)

                # MediaPipe expects SRGB/RGB
                infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
                infer_rgb = np.ascontiguousarray(infer_rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)

                res = landmarker.detect_for_video(mp_image, timestamp_ms)

                xy, sc = _extract_coco17_from_pose_landmarker(res, orig_w=width, orig_h=height)
                pose_score = float(np.mean(sc)) if sc.size else 0.0

                t1 = time.perf_counter()
                infer_ms = (t1 - t0) * 1000.0

                ema_ms = infer_ms if ema_ms is None else (ema_alpha * infer_ms + (1.0 - ema_alpha) * ema_ms)
                perf_text = None
                if show_perf_text:
                    est_fps = 1000.0 / max(1e-6, (ema_ms or infer_ms))
                    perf_text = f"{est_fps:5.1f} FPS | {ema_ms:5.1f} ms/frame"

                annotated = _draw_overlay(frame_bgr, xy, sc, min_score=min_score, show_text=perf_text)
                writer.write(annotated)

                frames_json.append({
                    "frame_index": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "pose_score": round(pose_score, 6),
                    "keypoints": [
                        {"name": COCO_17[i], "x": float(xy[i, 0]), "y": float(xy[i, 1]), "score": float(sc[i])}
                        for i in range(17)
                    ],
                })

                frame_idx += 1

    finally:
        cap.release()
        writer.release()

    # Write JSON (pretty-printed for readability; can be removed for speed)
    out = {
        "meta": {
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "frame_count": int(frame_idx),
        },
        "keypoint_format": COCO_17,
        "confidence_convention": "Per-keypoint 'score' is max(landmark.visibility, landmark.presence) in [0,1] when available. 0 indicates missing/low-confidence keypoints.",
        "frames": frames_json,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pose keypoint extractor: COCO-17 skeleton overlay + JSON.")
    p.add_argument("--input", required=True, help="Input video path (e.g. video.mp4)")
    p.add_argument("--output", required=True, help="Output annotated video path (e.g. out.mp4)")
    p.add_argument("--json", required=True, dest="json_path", help="Output JSON path (e.g. out.json)")
    p.add_argument("--max-side", type=int, default=640,
                   help="Downscale frames so max(H,W)<=max-side before inference for speed. Use 0 to disable.")
    p.add_argument("--min-score", type=float, default=0.2,
                   help="Only draw points/lines for keypoints with score >= min-score.")
    p.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2],
                   help="PoseLandmarker model variant (0=lite, 1=full, 2=heavy).")
    p.add_argument("--model-path", default=None,
                   help="Path to a PoseLandmarker .task file. If omitted, a default model is downloaded into ./models/.")
    p.add_argument("--no-download-model", action="store_true",
                   help="Disable auto-downloading the .task model on first run (requires --model-path pointing to an existing file).")
    p.add_argument("--no-smooth", action="store_true",
                   help="(Legacy flag) Kept for CLI compatibility. Tasks VIDEO mode already uses tracking for temporal stability.")
    p.add_argument("--no-perf-text", action="store_true",
                   help="Disable FPS/latency overlay text.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    process_video(
        input_path=args.input,
        output_video_path=args.output,
        output_json_path=args.json_path,
        max_side=args.max_side,
        min_score=args.min_score,
        model_complexity=args.model_complexity,
        smooth_landmarks=not args.no_smooth,
        show_perf_text=not args.no_perf_text,
        model_path=args.model_path,
        download_model=not args.no_download_model,
    )


if __name__ == "__main__":
    main()

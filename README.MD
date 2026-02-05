# ML Technical Interview Task (Python) — Solution

This repo implements the CLI described in the provided spec:

```bash
python run.py --input video.mp4 --output out.mp4 --json out.json
```

It outputs:

1) an annotated video with a skeleton overlay (COCO-17 keypoints + consistent skeleton edges)  
2) a JSON file containing **17 COCO-style body keypoints per frame** (with confidence)

## Approach

I used **MediaPipe Tasks: PoseLandmarker** for single-person pose estimation and mapped a subset of its 33 BlazePose landmarks to **COCO-17**.

Why this choice:

- **Practical + shippable**: a single pip dependency (`mediapipe`) with CPU-optimized inference.
- **Fast**: typically tens of milliseconds per frame on a developer laptop.
- **Robust**: handles real-world phone video well.

Notes on MediaPipe versions:

- On MediaPipe **0.10.30+**, many platforms no longer expose the legacy `mp.solutions` API by default.
- This repo uses the **Tasks** API (PoseLandmarker), so it works with newer MediaPipe releases.

Keypoint output order matches the required COCO-17 schema:

`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run.py --input video.mp4 --output out.mp4 --json out.json
```

Helpful flags:

- `--max-side 640` (default): downscales frames so `max(H,W) <= 640` before inference to reduce latency. Set `0` to disable.
- `--min-score 0.2` (default): draw only keypoints/edges above threshold.
- `--model-complexity {0,1,2}`: PoseLandmarker model variant (0=lite, 1=full, 2=heavy).
- `--model-path /path/to/pose_landmarker_*.task`: use a locally downloaded model bundle.
- `--no-download-model`: disable auto-download (requires `--model-path`).
- `--no-perf-text`: disables FPS / ms-per-frame overlay text.

## Output formats

### Annotated video

- Same duration as input
- Same resolution as input
- Overlay:
  - keypoints (dots)
  - skeleton edges (lines)
  - optional FPS / latency text

### JSON

Top-level structure:

```json
{
  "meta": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 900},
  "keypoint_format": ["nose", "...", "right_ankle"],
  "confidence_convention": "Per-keypoint 'score' is max(landmark.visibility, landmark.presence) in [0,1] when available. ...",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "pose_score": 0.92,
      "keypoints": [
        {"name": "nose", "x": 960.2, "y": 210.8, "score": 0.99}
      ]
    }
  ]
}
```

Notes:

- **(x, y)** are pixel coordinates in the **original** input frame.
- `score` is `max(landmark.visibility, landmark.presence)` when available (both are **[0,1]**).
- If pose tracking fails on a frame, we output `(x,y)=(0,0)` and `score=0` for all keypoints.
- `pose_score` is the mean of the 17 per-keypoint scores.

## How to realistically push toward ~500ms (or better)

This implementation is already typically well under 500ms/frame on CPU. If you needed to guarantee ~500ms across machines and videos:

1. **Downscale for inference** (already supported via `--max-side`):
   - e.g. `--max-side 512` or `384` can cut latency significantly.
2. **Use the lite model** (`--model-complexity 0`).
3. **Pipeline/parallelism**:
   - decode frames on one thread/process, run inference on another (producer/consumer queue).
4. **Hardware acceleration**:
   - On macOS/Apple Silicon: use MediaPipe’s GPU delegate / Metal pipelines where supported.
   - On Android: NNAPI delegate.
5. **Avoid per-frame allocations**:
   - reuse buffers; avoid repeated conversions when possible.
6. **Batching (when permissible)**:
   - some pose runtimes support multi-frame batching; if available, batch 2–4 frames for throughput.

## Project structure

- `run.py` — CLI entrypoint (required)
- `requirements.txt`
- `README.md`

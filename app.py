import os
import sys
import io
import json
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

# Add yolov5 to path so its internal imports resolve correctly
YOLOV5_ROOT = Path(__file__).resolve().parent / "yolov5"
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

import cv2
import torch
import numpy as np
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64

app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------------- #
#  Model loading
# --------------------------------------------------------------------------- #
WEIGHTS = os.environ.get("MODEL_WEIGHTS", "yolov5s.pt")   # replace with your
                                                            # custom .pt path
DEVICE = "cpu"   # change to "cuda" if GPU is available

print(f"[INFO] Loading model weights: {WEIGHTS}")
model = torch.hub.load(
    str(YOLOV5_ROOT),
    "custom",
    path=WEIGHTS,
    source="local",
    force_reload=False,
    verbose=False,
)
model.to(DEVICE)
model.eval()
print("[INFO] Model loaded successfully.")

# --------------------------------------------------------------------------- #
#  Webcam streaming helpers
# --------------------------------------------------------------------------- #
camera = None   # lazily opened


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def generate_frames():
    """Yield annotated MJPEG frames from the webcam."""
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break

        # Run inference
        results = model(frame)
        annotated = np.squeeze(results.render())   # renders bboxes onto frame

        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream from webcam with detections overlaid."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_image():
    """Accept a base64-encoded image, run detection, and return annotated image."""
    try:
        data = request.get_json()
        image_b64 = data.get("image", "")

        # Decode base64 → OpenCV frame
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        results = model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        annotated = np.squeeze(results.render())
        _, buffer = cv2.imencode(".jpg", annotated)
        result_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "image": result_b64,
                "detections": detections,
                "count": len(detections),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": WEIGHTS})


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

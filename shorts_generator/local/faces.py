"""Face detection for framing.

OpenCV's YuNet (a 230 KB model in assets/models, MIT licence) finds small, turned and partly covered faces, which is what a
stage show, a podcast or a panel is made of. The old Haar cascade is only the fallback when the model file is missing.
"""
from pathlib import Path
from typing import List, Tuple

import cv2  # type: ignore

MODEL = Path(__file__).resolve().parents[2] / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
Box = Tuple[int, int, int, int]                       # x, y, width, height in source pixels
MIN_FACE_FRAC = 0.045                                 # smaller than this share of the picture height: audience, posters, noise


class FaceDetector:
    def __init__(self, width: int, height: int, max_side: int = 640, score: float = 0.6):
        self.scale = min(1.0, max_side / float(max(width, 1)))
        self.size = (max(1, int(round(width * self.scale))), max(1, int(round(height * self.scale))))
        self.min_height = MIN_FACE_FRAC * height
        self.net = None
        self.cascade = None
        if MODEL.is_file() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.net = cv2.FaceDetectorYN.create(str(MODEL), "", self.size, score, 0.3, 200)
            except cv2.error:
                self.net = None
        if self.net is None:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    @property
    def kind(self) -> str:
        return "yunet" if self.net is not None else "haar"

    def detect(self, frame) -> List[Box]:
        small = frame if self.scale == 1.0 else cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        k = 1.0 / self.scale
        if self.net is not None:
            _, found = self.net.detect(small)
            boxes = [] if found is None else [(row[0], row[1], row[2], row[3]) for row in found]
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            side = max(20, int(self.min_height * self.scale))
            boxes = list(self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(side, side)))
        out = [(int(x * k), int(y * k), int(w * k), int(h * k)) for x, y, w, h in boxes]
        return [b for b in out if b[3] >= self.min_height]

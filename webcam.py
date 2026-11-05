# pip install mediapipe opencv-python


from pathlib import Path
import json
import math
import time
from urllib.request import urlopen

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from calibration_data import POINT_BEHIND_SAMPLES


MODEL_PATH = Path(__file__).resolve().parent / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MIN_GESTURE_CONFIDENCE = 0.6
CALIBRATION_PATH = Path(__file__).resolve().parent / "point_behind_calibration.json"
IMAGE_DIRECTORY = Path(__file__).resolve().parent / "images"


class GestureImageWindow:
    """Cache gesture images and switch after three consistent video frames."""

    WINDOW_NAME = "Image"
    GESTURE_FILES = {
        "Both Palms Up": "both_palms_up",
        "Point Behind": "point_behind",
        "Pointing Up": "point_up",
    }

    def __init__(self, directory=IMAGE_DIRECTORY):
        self.images = {}
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                if path.stem.lower() not in self.GESTURE_FILES.values():
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    print(f"Could not load gesture image: {path}")
                    continue
                self.images[path.stem.lower()] = image
        for stem in self.GESTURE_FILES.values():
            if stem not in self.images:
                print(f"No image loaded for {stem} in {directory}")
        self.candidate = None
        self.candidate_frames = 0
        self.current = None

    def update(self, labels, combined):
        # A two-hand pose wins; for different one-hand poses prefer Point Behind.
        gestures = {label.split(" (")[0] for label in labels}
        if combined:
            gestures.add(combined)
        selected = next((stem for gesture, stem in self.GESTURE_FILES.items()
                         if gesture in gestures and stem in self.images), None)
        if selected is None:
            if self.current is not None:
                try:
                    cv2.destroyWindow(self.WINDOW_NAME)
                except cv2.error:
                    pass  # The user may have already closed the image window.
            self.current = None
            self.candidate = None
            self.candidate_frames = 0
            return
        if selected != self.candidate:
            self.candidate = selected
            self.candidate_frames = 0
        self.candidate_frames += 1
        if self.candidate_frames >= 3:
            if self.current is None:
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self.current = selected
        # AUTOSIZE follows each image's original dimensions when gestures change.
        if self.current is not None:
            cv2.imshow(self.WINDOW_NAME, self.images[self.current])


def download_model():
    """Download the pretrained model once, beside this script."""
    if MODEL_PATH.is_file():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the gesture recognition model...")
    # Save only a completed download so interruptions do not leave a partial model.
    with urlopen(MODEL_URL, timeout=60) as response:
        model_data = response.read()
    MODEL_PATH.write_bytes(model_data)


def pose_features(landmarks, width, height):
    """Visible hand shape, relative to the wrist and normalized by palm size.

    Deliberately omit estimated depth: it missed the user's foreshortened thumb.
    Keep the screen orientation because it helps distinguish an upright thumb.
    """
    points = [(p.x * width, p.y * height) for p in landmarks]
    scale = math.dist(points[0], points[9])
    if scale < 1:
        return None
    return [((x - points[0][0]) / scale, (y - points[0][1]) / scale)
            for x, y in points]


class BehindCalibration:
    """Learn the user's visible Point Behind pose from short calibration clips."""

    def __init__(self, path=CALIBRATION_PATH, default_samples=POINT_BEHIND_SAMPLES):
        self.path = path
        self.default_samples = [[list(point) for point in sample]
                                for sample in default_samples]
        self.samples = self.default_samples.copy()
        self.pending = []
        self.start_at = None
        self.message = "Point Behind ready | B: add examples | R: restore | Q: quit"
        if path.is_file():
            try:
                samples = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(samples, list) or len(samples) > 100
                        or any(len(s) != 21 or any(len(p) != 2 or
                            any(not isinstance(v, (int, float)) or not math.isfinite(v)
                                for v in p) for p in s) for s in samples)):
                    raise ValueError("Invalid calibration data")
                self.samples = samples
            except (ValueError, TypeError, OSError):
                self.message = "Using built-in calibration | B: add examples | Q: quit"

    def start(self):
        self.pending = []
        self.start_at = time.monotonic() + 2

    def reset(self):
        self.samples = self.default_samples.copy()
        self.pending = []
        self.start_at = None
        self.path.unlink(missing_ok=True)
        self.message = "Built-in calibration restored | B: add examples | Q: quit"

    def update(self, result, width, height):
        if self.start_at is None:
            return
        remaining = self.start_at - time.monotonic()
        if remaining > 0:
            self.message = f"Make Point Behind with ONE hand: {math.ceil(remaining)}"
            return
        if len(result.hand_landmarks) != 1:
            self.message = "Calibration: show exactly ONE hand (B: restart)"
            return
        features = pose_features(result.hand_landmarks[0], width, height)
        if features is None:
            self.message = "Calibration: move hand into view (B: restart)"
            return
        self.pending.append(features)
        self.message = f"Hold Point Behind: {len(self.pending)}/20"
        if len(self.pending) == 20:
            # Repeat B to add a wrist angle or the other hand, without losing
            # earlier examples. Keep the calibration small and local.
            self.samples = (self.samples + self.pending)[-100:]
            self.path.write_text(json.dumps(self.samples), encoding="utf-8")
            self.start_at = None
            self.pending = []
            self.message = "Saved! B: add angle/other hand | R: reset | Q: quit"

    def matches(self, landmarks, width, height):
        features = pose_features(landmarks, width, height)
        if features is None:
            return False
        for sample in self.samples:
            for mirror in (1, -1):
                distances = [math.dist((mirror * p[0], p[1]), q)
                             for p, q in zip(features, sample)]
                # Separate thumb/fingertip checks prevent the palm's many stable
                # points from hiding a changed thumb or an extended finger.
                if (math.sqrt(sum(d*d for d in distances) / 21) < 0.16
                        and max(distances[3:5]) < 0.22
                        and max(distances[i] for i in (8, 12, 16, 20)) < 0.25):
                    return True
        return False


def palm_up(world, handedness):
    """An open hand with its palm normal pointing mainly toward screen-up.

    Uses 3D geometry so a foreshortened open hand need not receive Open_Palm
    from the canned classifier. Allows a relaxed bend and a forward tilt.
    """
    if len(world) != 21 or handedness not in ("Left", "Right"):
        return False
    xyz = [(p.x, p.y, p.z) for p in world]
    if not all(math.isfinite(v) for point in xyz for v in point):
        return False

    def vector(a, b):
        return tuple(xyz[b][i] - xyz[a][i] for i in range(3))

    # Require all four fingers to be mostly extended. The endpoint/path ratio
    # tolerates naturally cupped fingers but rejects a fist and pointing finger.
    for base in (5, 9, 13, 17):
        path = sum(math.dist(xyz[i], xyz[i + 1]) for i in range(base, base + 3))
        if path < 1e-6 or math.dist(xyz[base], xyz[base + 3]) / path < 0.8:
            return False

    a, b = vector(0, 5), vector(0, 17)
    normal = (a[1]*b[2] - a[2]*b[1],
              a[2]*b[0] - a[0]*b[2],
              a[0]*b[1] - a[1]*b[0])
    length = math.sqrt(sum(v*v for v in normal))
    if length < 1e-8:
        return False
    # Handedness from the model paired with this script's mirrored selfie input.
    # Reverse the winding for opposite hands; negative y points upward.
    sign = -1 if handedness == "Right" else 1
    upward = -sign * normal[1] / length
    return upward >= 0.55


def offering_palm(landmarks, width, height):
    """Recognize the visible, cupped offering pose from the user's screenshot.

    From a front-facing camera the wrist sits above the foreshortened fingers,
    with a thumb splayed to the side. This fallback avoids unreliable depth
    and straight-finger estimates when the hand faces partly toward the lens.
    """
    if len(landmarks) != 21:
        return False
    points = [(p.x * width, p.y * height) for p in landmarks]
    if not all(math.isfinite(v) for point in points for v in point):
        return False
    wrist = points[0]
    palm_size = math.dist(wrist, points[9])
    if palm_size < 5:
        return False
    # A spread thumb separates the open, cupped hand from a hanging fist.
    thumb = points[4]
    if (math.dist(wrist, thumb) < 1.25 * palm_size
            or abs(thumb[0] - wrist[0]) < 0.65 * palm_size
            or thumb[1] > wrist[1] - 0.15 * palm_size):
        return False
    if math.dist(points[5], points[17]) < 0.45 * palm_size:
        return False
    # Allow one obscured/curled finger and relaxed cupping of the others.
    offered = sum(points[i][1] > wrist[1] + 0.15 * palm_size
                  and math.dist(wrist, points[i]) > 0.9 * palm_size
                  for i in (8, 12, 16, 20))
    return offered >= 3


def result_palm_up(result, index, width, height):
    if offering_palm(result.hand_landmarks[index], width, height):
        return True
    worlds = getattr(result, "hand_world_landmarks", [])
    handedness = getattr(result, "handedness", [])
    if index >= len(worlds) or index >= len(handedness) or not handedness[index]:
        return False
    hand = handedness[index][0]
    return hand.score >= 0.6 and palm_up(worlds[index], hand.category_name)


def classify_hands(result, width, height, calibration=None):
    """Return individual gestures and the two-hand, palms-up offering pose."""
    labels = []
    upward_palms = []
    for index, landmarks in enumerate(result.hand_landmarks):
        categories = result.gestures[index] if index < len(result.gestures) else []
        gesture = categories[0] if categories else None
        recognized = (gesture is not None
                      and gesture.score >= MIN_GESTURE_CONFIDENCE)
        is_open = bool(recognized and gesture.category_name == "Open_Palm")
        upward_palms.append(result_palm_up(result, index, width, height))
        if recognized and gesture.category_name == "Pointing_Up":
            labels.append(f"Pointing Up ({gesture.score:.0%})")
        elif (not is_open and calibration is not None
              and calibration.matches(landmarks, width, height)):
            labels.append("Point Behind")
        else:
            labels.append("Unknown gesture")

    combined = "Both Palms Up" if len(upward_palms) == 2 and all(upward_palms) else None
    if combined:
        labels = [combined] * 2
    return labels, combined


def draw_result(frame, result, calibration=None):
    """Draw hand skeletons and return recognition results without text overlays."""
    height, width = frame.shape[:2]
    labels, combined = classify_hands(result, width, height, calibration)

    for landmarks in result.hand_landmarks:
        points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[connection.start], points[connection.end],
                     (0, 220, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

    return labels, combined


def main():
    download_model()
    calibration = BehindCalibration()
    gesture_images = GestureImageWindow()
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        camera = cv2.VideoCapture(0)
        try:
            if not camera.isOpened():
                raise RuntimeError("Could not open the webcam.")

            last_timestamp_ms = -1
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                # VIDEO mode tracks hands between frames; timestamps must increase.
                timestamp_ms = max(time.monotonic_ns() // 1_000_000,
                                   last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = recognizer.recognize_for_video(image, timestamp_ms)
                height, width = frame.shape[:2]
                calibration.update(result, width, height)
                labels, combined = draw_result(frame, result, calibration)
                gesture_images.update(labels, combined)
                cv2.imshow("Camera", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    calibration.start()
                elif key == ord("r"):
                    calibration.reset()
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

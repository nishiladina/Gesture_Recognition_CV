# pip install mediapipe opencv-python


from pathlib import Path
import time
from urllib.request import urlopen

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).resolve().parent / "models" / "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MIN_GESTURE_CONFIDENCE = 0.6


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


def draw_result(frame, result):
    """Draw the hand skeleton and the highest-confidence gesture."""
    height, width = frame.shape[:2]
    label = "No hand detected"
    color = (0, 200, 255)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, points[connection.start], points[connection.end],
                     (0, 220, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

        label = "Unknown gesture"
        if result.gestures and result.gestures[0]:
            gesture = result.gestures[0][0]
            if (gesture.category_name not in (None, "None", "Unknown")
                    and gesture.score >= MIN_GESTURE_CONFIDENCE):
                name = gesture.category_name.replace("_", " ")
                label = f"{name} ({gesture.score:.0%})"
                color = (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (width, 50), (30, 30, 30), -1)
    cv2.putText(frame, label, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def main():
    download_model()
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
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
                draw_result(frame, result)
                cv2.imshow("Gesture Recognition - press Q to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

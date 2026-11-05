"""Checks for supported gestures and personal Point Behind calibration."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as Record
import unittest
from unittest.mock import patch

import webcam


def hand():
    # A compact fist with an upward, foreshortened thumb. No useful depth.
    pixels = [(0, 0), (0.2, -0.2), (0.3, -0.5), (0.35, -0.8), (0.4, -0.9),
              (0.2, -0.9), (0, -1), (-0.1, -0.9), (0, -0.7),
              (0, -1), (-0.2, -1.05), (-0.3, -0.9), (-0.1, -0.7),
              (-0.2, -0.9), (-0.4, -0.95), (-0.45, -0.8), (-0.3, -0.6),
              (-0.4, -0.8), (-0.5, -0.85), (-0.55, -0.7), (-0.4, -0.5)]
    return [Record(x=0.5 + x * 0.1, y=0.6 + y * 0.1, z=0) for x, y in pixels]


def result_for(*categories):
    return Record(hand_landmarks=[hand() for _ in categories],
                  gestures=[[Record(category_name=name, score=score)]
                            for name, score in categories])


def palm_world(side="Right", tilt=0, curled=False):
    """Open offering hand, rotated toward vertical or downward for negatives."""
    points = [(0, 0, 0)] * 21
    for base, x in zip((5, 9, 13, 17), (0.35, 0.1, -0.1, -0.35)):
        for step in range(4):
            z = -1 - 0.35 * step
            y = 0
            if curled and step >= 2:
                z = -1.1
                y = 0.2 * step
            points[base + step] = (x, y, z)
    angle = math.radians(tilt)
    mirror = 1 if side == "Right" else -1
    return [Record(x=mirror*x*0.05,
                   y=(y*math.cos(angle)-z*math.sin(angle))*0.05,
                   z=(y*math.sin(angle)+z*math.cos(angle))*0.05)
            for x, y, z in points]


def offering_result(tilts=(0, 0), curled=False, category="None"):
    result = result_for(*[(category, 0.9) for _ in tilts])
    sides = ("Left", "Right")
    result.hand_world_landmarks = [palm_world(sides[i], tilt, curled)
                                   for i, tilt in enumerate(tilts)]
    result.handedness = [[Record(category_name=sides[i], score=0.99)]
                         for i in range(len(tilts))]
    return result


def cupped_screen_hand(mirror=1):
    # Approximate the offering silhouette in the user's annotated screenshot:
    # wrist above fingers, an outward thumb, and curled/foreshortened fingers.
    points = [(198, 313), (168, 287), (126, 278), (93, 274), (62, 267),
              (107, 319), (91, 323), (77, 333), (57, 314),
              (125, 338), (125, 353), (90, 358), (89, 374),
              (152, 351), (125, 371), (125, 389), (91, 358),
              (183, 357), (185, 369), (185, 381), (184, 396)]
    return [Record(x=(350 + mirror*(x-198))/717, y=y/455, z=0)
            for x, y in points]


class GestureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calibration.json"
        self.calibration = webcam.BehindCalibration(self.path, default_samples=())

    def teach(self):
        self.calibration.samples = [webcam.pose_features(hand(), 640, 480)]

    def test_removed_labels_are_not_shown(self):
        for name in ("Closed_Fist", "Open_Palm", "Victory", "ILoveYou", "Thumb_Up",
                     "Thumb_Down", "Thumb Side Left", "Thumb Side Right", "None"):
            with self.subTest(name=name):
                labels, combined = webcam.classify_hands(result_for((name, 0.99)), 640, 480)
                self.assertEqual(labels, ["Unknown gesture"])
                self.assertIsNone(combined)

    def test_point_up_is_retained(self):
        self.assertEqual(webcam.classify_hands(
            result_for(("Pointing_Up", 0.9)), 640, 480)[0], ["Pointing Up (90%)"])

    def test_both_palms_up_with_relaxed_tilt(self):
        for tilts in ((0, 0), (25, 35), (-25, 10)):
            with self.subTest(tilts=tilts):
                labels, combined = webcam.classify_hands(offering_result(tilts), 640, 480)
                self.assertEqual(combined, "Both Palms Up")
                self.assertEqual(labels, [combined, combined])

    def test_forward_downward_mixed_and_single_hands_are_rejected(self):
        for tilts in ((), (0,), (90, 90), (180, 180), (0, 90), (0, 180)):
            with self.subTest(tilts=tilts):
                _, combined = webcam.classify_hands(offering_result(tilts), 640, 480)
                self.assertIsNone(combined)

    def test_canned_open_palm_cannot_override_orientation(self):
        _, combined = webcam.classify_hands(
            offering_result((90, 90), category="Open_Palm"), 640, 480)
        self.assertIsNone(combined)

    def test_fists_are_not_palms_up(self):
        _, combined = webcam.classify_hands(offering_result(curled=True), 640, 480)
        self.assertIsNone(combined)

    def test_missing_or_unreliable_handedness_does_not_trigger(self):
        for missing in (True, False):
            result = offering_result()
            if missing:
                result.handedness = []
            else:
                result.handedness[0][0].score = 0.3
            self.assertIsNone(webcam.classify_hands(result, 640, 480)[1])

    def test_degenerate_world_landmarks_are_rejected(self):
        world = [Record(x=0, y=0, z=0) for _ in range(21)]
        self.assertFalse(webcam.palm_up(world, "Right"))

    def test_cupped_screenshot_pose_without_depth_or_handedness(self):
        result = result_for(("None", 0.9), ("None", 0.9))
        result.hand_landmarks = [cupped_screen_hand(-1), cupped_screen_hand(1)]
        labels, combined = webcam.classify_hands(result, 717, 455)
        self.assertEqual(combined, "Both Palms Up")
        self.assertEqual(labels, [combined, combined])

    def test_one_cupped_hand_does_not_trigger_combined_pose(self):
        result = result_for(("None", 0.9))
        result.hand_landmarks = [cupped_screen_hand()]
        self.assertIsNone(webcam.classify_hands(result, 717, 455)[1])

    def test_cupped_pose_tolerates_small_changes_and_scale(self):
        points = cupped_screen_hand()
        for i in (12, 16, 20):
            points[i].y += 0.005
        scaled = [Record(x=0.2 + (p.x-0.5)*0.7, y=0.1+p.y*0.7, z=0)
                  for p in points]
        self.assertTrue(webcam.offering_palm(scaled, 717, 455))

    def test_closed_thumb_and_upright_fingers_rejected_by_visible_rule(self):
        points = cupped_screen_hand()
        points[4] = points[5]
        self.assertFalse(webcam.offering_palm(points, 717, 455))
        points = cupped_screen_hand()
        for i in (8, 12, 16, 20):
            points[i].y = points[0].y - 0.2
        self.assertFalse(webcam.offering_palm(points, 717, 455))

    def test_uncalibrated_pose_is_unknown(self):
        self.assertEqual(webcam.classify_hands(result_for(("None", 0.9)),
                         640, 480, self.calibration)[0], ["Unknown gesture"])

    def test_calibrated_pose_works_without_depth_or_canned_gesture(self):
        self.teach()
        self.assertEqual(webcam.classify_hands(result_for(("Thumb_Up", 0.9)),
                         640, 480, self.calibration)[0], ["Point Behind"])

    def test_translation_scale_and_horizontal_mirror(self):
        self.teach()
        for mirror in (-1, 1):
            moved = [Record(x=0.2 + mirror * (p.x - 0.5) * 1.5,
                            y=0.5 + (p.y - 0.6) * 1.5, z=10) for p in hand()]
            self.assertTrue(self.calibration.matches(moved, 640, 480))

    def test_different_thumb_or_extended_finger_is_rejected(self):
        self.teach()
        for index in (4, 8, 12, 16, 20):
            points = hand()
            points[index].y -= 0.1
            self.assertFalse(self.calibration.matches(points, 640, 480))

    def test_calibration_waits_for_one_hand_then_saves_and_reloads(self):
        with patch.object(webcam.time, "monotonic", return_value=10):
            self.calibration.start()
            self.calibration.update(result_for(("None", 0.9)), 640, 480)
            self.assertFalse(self.calibration.pending)
        with patch.object(webcam.time, "monotonic", return_value=13):
            self.calibration.update(result_for(), 640, 480)
            self.calibration.update(result_for(("None", 0.9), ("None", 0.9)), 640, 480)
            self.assertFalse(self.calibration.pending)
            for _ in range(20):
                self.calibration.update(result_for(("None", 0.9)), 640, 480)
        self.assertEqual(len(json.loads(self.path.read_text())), 20)
        self.assertIsNone(self.calibration.start_at)
        loaded = webcam.BehindCalibration(self.path, default_samples=())
        self.assertTrue(loaded.matches(hand(), 640, 480))
        loaded.reset()
        self.assertFalse(self.path.exists())
        self.assertFalse(loaded.matches(hand(), 640, 480))

    def test_invalid_calibration_and_degenerate_hand(self):
        self.path.write_text('{"bad": true}')
        loaded = webcam.BehindCalibration(self.path, default_samples=())
        self.assertFalse(loaded.samples)
        self.assertFalse(loaded.matches([Record(x=0, y=0) for _ in range(21)], 640, 480))

    def test_embedded_calibration_works_without_json_file(self):
        loaded = webcam.BehindCalibration(self.path)
        self.assertFalse(self.path.exists())
        self.assertGreater(len(loaded.samples), 0)
        for sample in loaded.samples:
            landmarks = [Record(x=0.5 + x * 0.1, y=0.6 + y * 0.1 * 640 / 480)
                         for x, y in sample]
            self.assertTrue(loaded.matches(landmarks, 640, 480))

    def test_reset_restores_embedded_samples(self):
        loaded = webcam.BehindCalibration(self.path)
        original = loaded.samples.copy()
        loaded.samples = []
        loaded.reset()
        self.assertEqual(loaded.samples, original)
        self.assertFalse(self.path.exists())

    def test_invalid_override_falls_back_to_embedded_samples(self):
        self.path.write_text('invalid json')
        loaded = webcam.BehindCalibration(self.path)
        self.assertGreater(len(loaded.samples), 0)


class GestureImageTests(unittest.TestCase):
    def setUp(self):
        self.window = webcam.GestureImageWindow()
        self.show = patch.object(webcam.cv2, "imshow").start()
        self.create = patch.object(webcam.cv2, "namedWindow").start()
        self.destroy = patch.object(webcam.cv2, "destroyWindow").start()
        self.addCleanup(patch.stopall)

    def recognize(self, label, combined=None):
        for _ in range(3):
            self.window.update([label], combined)

    def test_actual_images_load_and_map_to_all_gestures(self):
        for label, stem in webcam.GestureImageWindow.GESTURE_FILES.items():
            with self.subTest(label=label):
                self.assertIn(stem, self.window.images)
                self.recognize(label + " (90%)")
                self.assertEqual(self.window.current, stem)
                self.assertIs(self.show.call_args.args[1], self.window.images[stem])

    def test_brief_predictions_do_not_switch_images(self):
        self.window.update(["Pointing Up (90%)"], None)
        self.window.update(["Unknown gesture"], None)
        self.assertIsNone(self.window.current)
        self.show.assert_not_called()

    def test_unknown_and_no_hands_hide_image_immediately(self):
        for labels in ([], ["Unknown gesture"]):
            self.recognize("Point Behind")
            self.show.reset_mock()
            self.window.update(labels, None)
            self.assertIsNone(self.window.current)
            self.assertIsNone(self.window.candidate)
            self.assertEqual(self.window.candidate_frames, 0)
            self.show.assert_not_called()
        self.assertEqual(self.destroy.call_count, 2)

    def test_images_keep_original_dimensions_and_use_autosize(self):
        for label, stem in webcam.GestureImageWindow.GESTURE_FILES.items():
            path = next(webcam.IMAGE_DIRECTORY.glob(stem + ".*"))
            original = webcam.cv2.imread(str(path))
            self.assertEqual(self.window.images[stem].shape, original.shape)
            self.recognize(label)
        self.create.assert_called_once_with(self.window.WINDOW_NAME,
                                            webcam.cv2.WINDOW_AUTOSIZE)

    def test_window_reopens_after_gesture_returns(self):
        self.recognize("Point Behind")
        self.window.update([], None)
        self.recognize("Both Palms Up")
        self.assertEqual(self.window.current, "both_palms_up")
        self.assertEqual(self.create.call_count, 2)

    def test_combined_gesture_has_priority(self):
        for _ in range(3):
            self.window.update(["Pointing Up (90%)", "Point Behind"], "Both Palms Up")
        self.assertEqual(self.window.current, "both_palms_up")

    def test_conflicting_single_hands_have_consistent_priority(self):
        for labels in (["Pointing Up (90%)", "Point Behind"],
                       ["Point Behind", "Pointing Up (90%)"]):
            for _ in range(3):
                self.window.update(labels, None)
            self.assertEqual(self.window.current, "point_behind")

    def test_missing_image_does_not_open_window(self):
        self.window.images.clear()
        self.recognize("Point Behind")
        self.assertIsNone(self.window.current)
        self.show.assert_not_called()


if __name__ == "__main__":
    unittest.main()

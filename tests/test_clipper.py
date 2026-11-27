import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shorts_generator.local import clipper as K


def analysis(boxes_per_sample, size=(1280, 720), roi=None, n=40):
    return {"fps": 30.0, "n_frames": n * 8, "size": size, "roi": roi or (0, 0, size[0], size[1]),
            "sidx": [i * 8 for i in range(n)], "boxes": boxes_per_sample, "motion": None}


class Geometry(unittest.TestCase):
    def test_output_sizes_are_real_resolutions(self):
        self.assertEqual(K._out_size("9:16"), (1080, 1920))
        self.assertEqual(K._out_size("1:1"), (1080, 1080))
        self.assertEqual(K._out_size("4:5"), (1080, 1350))
        self.assertEqual(K._out_size("16:9"), (1920, 1080))
        self.assertEqual(K._ratio("nonsense"), 9 / 16)

    def test_filenames_are_stable_and_safe(self):
        self.assertEqual(K.clip_filename(7, "The Plane: Crash!"), "short_07_the-plane-crash.mp4")
        self.assertEqual(K.clip_filename(1, "ہندی"), "short_01.mp4")

    def test_letterbox_bars_are_found_only_when_they_are_in_every_frame(self):
        rows = [np.concatenate([np.zeros(40), np.full(640, 200), np.zeros(40)]) for _ in range(10)]
        self.assertEqual(K._find_bars(rows), (42, 42))                                # +2 px safety margin each side
        flicker = [r.copy() for r in rows]
        for r in flicker[:5]:
            r[:40] = 255                                                              # bright content in half the frames
        self.assertEqual(K._find_bars(flicker)[0], 0)
        self.assertEqual(K._find_bars([np.full(100, 200.0)] * 5), (0, 0))


class LayoutChoice(unittest.TestCase):
    def test_static_corner_face_means_split(self):
        box = (1050, 100, 70, 70)                                                     # a facecam in the top-right corner
        plan = K._plan(analysis([[box]] * 40), "auto", 9 / 16, 1080, 1920)
        self.assertEqual(plan["layout"], "split")
        x, y, w, h = plan["top"]
        self.assertGreater(x, 800)                                                    # the facecam panel crops the corner...
        bx, by, bw, bh = plan["bot"]
        self.assertLess(bx + bw, x + 1)                                               # ...and the gameplay panel stays clear of it

    def test_large_centred_face_means_track(self):
        plan = K._plan(analysis([[(520, 150, 240, 240)]] * 40), "auto", 9 / 16, 1080, 1920)
        self.assertEqual(plan["layout"], "track")
        self.assertEqual(len(plan["x0s"]), 40 * 8)
        self.assertTrue(np.all(np.diff(plan["x0s"]) == 0))                            # a still speaker: the window does not move

    def test_wandering_game_faces_never_steer_the_crop(self):
        boxes = [[(100 + (i * 97) % 900, 200 + (i * 53) % 300, 90, 90)] if i % 5 == 0 else [] for i in range(40)]
        self.assertEqual(K._plan(analysis(boxes), "auto", 9 / 16, 1080, 1920)["layout"], "fit")

    def test_no_face_means_fit_and_requested_layout_is_respected(self):
        self.assertEqual(K._plan(analysis([[]] * 40), "auto", 9 / 16, 1080, 1920)["layout"], "fit")
        self.assertEqual(K._plan(analysis([[(520, 150, 240, 240)]] * 40), "fit", 9 / 16, 1080, 1920)["layout"], "fit")

    def test_speaker_moves_slowly_not_jittery(self):
        xs = [600 if i < 20 else 900 for i in range(40)]                              # the speaker walks to the right
        boxes = [[(x, 150, 240, 240)] for x in xs]
        plan = K._plan(analysis(boxes), "track", 9 / 16, 1080, 1920)
        step = np.abs(np.diff(plan["x0s"]))
        self.assertLess(step.max(), plan["cw"] * 0.35 / 30)                           # never faster than the speed limit per frame
        self.assertGreater(plan["x0s"][-1], plan["x0s"][0])                           # but it does follow in the end


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class RealRender(unittest.TestCase):
    """One real render (a 3 s synthetic video) to prove the encode, captions, title card and audio path work."""

    def test_render_with_every_overlay(self):
        tmp = Path(tempfile.mkdtemp(prefix="render_"))
        try:
            src = tmp / "in.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=4",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(src)], check=True)
            out = tmp / "out.mp4"
            segs = [{"start": 10.0, "end": 12.0, "text": "hello there everybody",
                     "words": [{"start": 10.0, "end": 10.6, "text": "hello"}, {"start": 10.6, "end": 11.2, "text": "there"},
                               {"start": 11.2, "end": 12.0, "text": "everybody"}]}]
            opts = K.RenderOptions(captions=True, caption_style="karaoke", title_card=True, watermark="@test")
            # the clip is seconds 10..13 of a (pretend) longer video; `src` is already that section
            K.render_clip(str(src), str(out), 10.0, 13.0, opts, transcript_segments=segs, title="A great moment", precut=True)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type,width,height,duration",
                                    "-of", "compact", str(out)], capture_output=True, text=True).stdout
            self.assertIn("codec_name=h264", probe)
            self.assertIn("width=1080", probe)
            self.assertIn("height=1920", probe)
            self.assertIn("codec_type=audio", probe)
            frame = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                                    "-pix_fmt", "gray", "-"], capture_output=True).stdout
            self.assertEqual(len(frame), 1080 * 1920)
            self.assertFalse(list(tmp.glob("*.cut.mp4")))                       # no leftovers
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_local_file_is_cut_to_the_requested_range(self):
        tmp = Path(tempfile.mkdtemp(prefix="localcut_"))
        try:
            src = tmp / "in.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
                            "-f", "lavfi", "-i", "sine=frequency=330:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(src)], check=True)
            out = tmp / "out.mp4"
            K.render_clip(str(src), str(out), 2.0, 5.0, K.RenderOptions(aspect_ratio="1:1", layout="fit"), precut=False)
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration",
                                    "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip().split(",")
            self.assertEqual(probe[:2], ["1080", "1080"])
            self.assertAlmostEqual(float(probe[2]), 3.0, delta=0.2)           # 5 s - 2 s
            self.assertFalse(list(tmp.glob("*.cut.mp4")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cancel_leaves_no_half_written_file(self):
        from shorts_generator.local import control
        import threading

        tmp = Path(tempfile.mkdtemp(prefix="cancel_"))
        try:
            src = tmp / "in.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=6",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
            ev = threading.Event()
            ev.set()
            control.bind(ev, None)
            try:
                with self.assertRaises(control.Cancelled):
                    K.render_clip(str(src), str(tmp / "out.mp4"), 0.0, 6.0, K.RenderOptions(), precut=True)
            finally:
                control.unbind()
            self.assertFalse((tmp / "out.mp4").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

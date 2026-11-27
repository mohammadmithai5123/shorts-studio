"""The streamer-over-game layout, meme clips, and the automatic clean-up."""
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import app as webapp
import cv2
import numpy as np
import tests  # noqa: F401  (blank keys, temp output folder)
from shorts_generator import config, pipeline
from shorts_generator.local import clipper as K
from shorts_generator.local import cutout, desktop, editing, housekeeping, memes, sfx
from shorts_generator.local import project as P

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
DAY = 86400


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


# --------------------------------------------------------------------------------------------- #
class WhereTheStreamerIs(unittest.TestCase):
    def analysis(self, faces, fps=30.0):
        """``faces``: one face box or None per sample, a sample every half second."""
        step = int(fps / 2)
        return {"roi": (0, 0, 1920, 1080), "fps": fps, "n_frames": step * len(faces), "sidx": [i * step for i in range(len(faces))],
                "boxes": [[] if f is None else [f] for f in faces]}

    def test_only_a_smallish_face_low_in_the_picture_counts(self):
        self.assertIsNotNone(cutout._low_face([(500, 780, 110, 135)], (0, 0, 1920, 1080)))
        for box in ((500, 200, 110, 135), (500, 700, 300, 360), (500, 1000, 20, 30)):       # high up, too big, too small
            self.assertIsNone(cutout._low_face([box], (0, 0, 1920, 1080)), box)

    def test_each_place_gets_its_own_stretch_and_blips_are_absorbed(self):
        left, right = (500, 780, 110, 135), (1320, 790, 110, 135)
        an = self.analysis([left] * 20 + [None] + [left] * 10 + [right] * 20 + [None] * 12)
        runs = cutout.positions(an)
        self.assertEqual([None if r["face"] is None else round(r["face"][0]) for r in runs], [500, 1320, None])
        self.assertEqual((runs[0]["start"], runs[-1]["end"]), (0, an["n_frames"]))
        self.assertTrue(all(a["end"] == b["start"] for a, b in zip(runs, runs[1:])))          # no frame without a plan

    def test_evidence_tells_a_cut_out_from_a_webcam_box_or_a_still_scene(self):
        self.assertTrue(cutout.is_cutout({"bottom": True, "changing": 0.8, "boxiness": 0.6}))
        self.assertFalse(cutout.is_cutout({"bottom": True, "changing": 0.8, "boxiness": 0.97}))   # a filled rectangle
        self.assertFalse(cutout.is_cutout({"bottom": True, "changing": 0.05, "boxiness": 0.6}))   # nothing moves around them
        self.assertFalse(cutout.is_cutout({"bottom": False, "changing": 0.8, "boxiness": 0.6}))

    def test_a_clip_without_such_a_face_is_left_to_the_other_layouts(self):
        an = self.analysis([(900, 300, 150, 190)] * 20)
        self.assertIsNone(cutout.plan("unused.mp4", an, 1080, 1920))


@unittest.skipUnless(cutout.available(), "needs the segmentation model")
class TheCutOutPicture(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(1)
        return rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def make_plan(self):
        box = (300, 700, 800, 1080)
        run = {"start": 0, "end": 100, "box": box, "erase": np.ones((380, 500), np.float32), "face": (500, 780, 110, 135)}
        game = (240, 0, 1440, 1080)
        run["patch"] = cutout._patch(run, game)
        return {"layout": "cutout", "runs": [run], "top_h": 826, "game": game, "game_h": 810, "background": cutout.BACKGROUNDS["green"]}

    def test_streamer_panel_on_top_game_below(self):
        frame = self.frame()
        with mock.patch.object(cutout, "person_mask", lambda img, full=True: np.zeros((192, 192), np.float32)):
            out = cutout.compose(frame, self.make_plan(), 5, (0, 0, 1920, 1080), 1080, 1920)
        self.assertEqual(out.shape, (1920, 1080, 3))
        self.assertEqual(tuple(int(v) for v in out[100, 100]), cutout.BACKGROUNDS["green"])       # nobody found: just the colour
        self.assertGreater(float(out[900:1600].std()), 30)                                        # the game is there

    def test_the_spot_where_the_streamer_was_is_smoothed_away(self):
        frame = self.frame()
        game = frame[0:1080, 240:1680].copy()
        before = float(game[800:1000, 200:400].std())
        cutout._erase(game, self.make_plan()["runs"][0]["patch"])
        self.assertLess(float(game[800:1000, 200:400].std()), before / 3)
        self.assertTrue(np.array_equal(game[:500], frame[0:500, 240:1680]))                       # nothing else touched

    def test_only_the_biggest_body_is_kept(self):
        sure = np.zeros((192, 192), np.float32)
        sure[60:190, 50:140] = 1.0
        sure[10:25, 160:180] = 1.0                       # a bit of the chair that looked a little like a person
        got = cutout._one_person(sure)
        self.assertEqual(float(got[17, 170]), 0.0)
        self.assertEqual(float(got[120, 90]), 1.0)

    def test_the_real_model_runs(self):
        got = cutout.person_mask(np.full((300, 400, 3), 128, np.uint8))
        self.assertEqual(got.shape, (300, 400))
        self.assertTrue(0.0 <= float(got.min()) and float(got.max()) <= 1.0)

    def test_the_layout_can_be_asked_for_and_the_background_is_checked(self):
        self.assertIn("cutout", K.LAYOUTS)
        K.RenderOptions(layout="cutout", facecam_bg="pink").validate()
        with self.assertRaises(ValueError):
            K.RenderOptions(facecam_bg="rainbow").validate()


# --------------------------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class MemeLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="memes_"))
        cls.env = mock.patch.dict(os.environ, {"LOCAL_MEME_DIR": str(cls.tmp)})
        cls.env.start()
        memes.ensure_folder()
        ffmpeg("-f", "lavfi", "-i", "color=c=0x00b140:s=320x320:r=25:d=1", "-f", "lavfi", "-i", "sine=frequency=600:duration=1",
               "-vf", "drawbox=x=80:y=80:w=160:h=160:color=red:t=fill", "-shortest", "-pix_fmt", "yuv420p", str(cls.tmp / "laugh" / "green.mp4"))
        ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=1", "-pix_fmt", "yuv420p", str(cls.tmp / "shock" / "card.mp4"))
        ffmpeg("-f", "lavfi", "-i", "sine=frequency=900:duration=1", str(cls.tmp / "faah_shock.mp3"))
        cv2.imwrite(str(cls.tmp / "sticker.png"), np.full((100, 100, 3), 200, np.uint8))
        (cls.tmp / "laugh" / "notes.txt").write_text("not a meme")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_moods_come_from_the_folder_or_the_name(self):
        got = {m.name: (m.mood, m.kind) for m in memes.scan()}
        self.assertEqual(got, {"green.mp4": ("laugh", "video"), "card.mp4": ("shock", "video"), "faah_shock.mp3": ("shock", "sound"),
                               "sticker.png": ("any", "image")})
        self.assertEqual(memes.summary()["total"], 4)

    def test_cues_match_the_mood_keep_their_distance_and_do_not_repeat(self):
        library = memes.scan()
        moments = [(1.0, "laugh", 0), (4.0, "laugh", 0), (5.0, "shock", 0), (14.0, "shock", 0), (25.0, "laugh", 0), (39.5, "laugh", 0)]
        cues = memes.plan(moments, library, 40.0, seed="x")
        self.assertEqual([c.t for c in cues], [4.0, 14.0, 25.0])                      # not at the start / end, 6 s apart
        self.assertEqual(cues[0].meme.name, "green.mp4")
        self.assertEqual(cues[1].meme.mood, "shock")
        self.assertEqual(cues[2].meme.mood, "any")                                    # the only laugh clip is used already
        self.assertEqual([(c.t, c.meme.path) for c in cues], [(c.t, c.meme.path) for c in memes.plan(moments, library, 40.0, seed="x")])
        self.assertEqual(memes.plan(moments, [], 40.0), [])

    def test_green_screen_is_removed_and_other_videos_get_a_frame(self):
        by = {m.name: m for m in memes.scan()}
        green = memes.load_frames(memes.Cue(0, 0.5, by["green.mp4"], "laugh"), 25, 300, 300)
        self.assertGreaterEqual(len(green), 10)
        self.assertEqual(int(green[0][5, 5, 3]), 0)                  # background gone
        self.assertEqual(int(green[0][150, 150, 3]), 255)            # subject stays
        card = memes.load_frames(memes.Cue(0, 0.5, by["card.mp4"], "shock"), 25, 300, 300)
        self.assertEqual(tuple(int(v) for v in card[0][1, 1]), (255, 255, 255, 255))     # white frame
        self.assertEqual(len(memes.load_frames(memes.Cue(0, 1.0, by["sticker.png"], "any"), 25, 300, 300)), 1)

    def test_sounds_are_levelled_and_mixed_in_at_their_time(self):
        by = {m.name: m for m in memes.scan()}
        sound = memes.sound_of(memes.Cue(0, 1.0, by["faah_shock.mp3"], "shock"))
        self.assertAlmostEqual(float(np.abs(sound).max()), 0.5, delta=0.02)
        path = str(self.tmp / "track.wav")
        sfx.write_track([], 3.0, path, clips=[(1.0, sound)])
        db = editing.audio_levels(path)
        self.assertLess(float(db[:40].max()), -60)
        self.assertGreater(float(db[int(1.2 / editing.HOP)]), -20)
        self.assertIsNone(memes.sound_of(memes.Cue(0, 1.0, by["card.mp4"], "shock")))     # silent video

    def test_viral_uses_memes_instead_of_emoji(self):
        effects = editing.effects_for("viral")
        self.assertTrue(effects["memes"])
        self.assertFalse(effects["popups"])
        self.assertFalse(editing.effects_for("clean")["memes"])
        hints = {"popups": [{"at": 5.0, "mood": "shock"}, {"at": 15.0, "emoji": "😂"}]}
        plan = editing.build_plan(24.0, 25.0, [], None, effects, hints, library=memes.scan())
        self.assertEqual([(round(c.t), c.mood) for c in plan.memes], [(5, "shock"), (16, "laugh")])     # the first one is a second long
        self.assertEqual([c.meme.how for c in plan.memes], ["cut", "over"])                            # a green screen goes over
        self.assertEqual((plan.popups, plan.summary()["memes"]), ([], 2))


# --------------------------------------------------------------------------------------------- #
class CleaningUp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="house_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.now = time.time()
        patches = [mock.patch.object(config, "LOCAL_OUTPUT_DIR", str(self.tmp)), mock.patch.object(P, "LOCAL_OUTPUT_DIR", str(self.tmp)),
                   mock.patch.dict(os.environ, {"LOCAL_DELETE_UPLOADED": "1", "LOCAL_CACHE_DAYS": "3"})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        meta = {"kind": "youtube", "id": "house1", "url": "https://youtu.be/x", "title": "v", "duration": 100.0, "embeddable": True, "heatmap": None}
        self.project = P.new_project(meta, {"duration": 100.0, "language": "en", "source": "t", "segments": []},
                                     [{"title": "a", "start_time": 0.0, "end_time": 10.0, "score": 1}], {})
        self.project["clips"] = [{"candidate": 0, "file": n} for n in ("posted.mp4", "remade.mp4", "waiting.mp4", "mine.mp4")]
        P.save_project(self.project)
        self.folder = P.project_dir("house1")
        for name in ("posted.mp4", "remade.mp4", "waiting.mp4", "mine.mp4"):
            (self.folder / name).write_bytes(b"x" * 1000)

    def age(self, path, days):
        os.utime(path, (self.now - days * DAY, self.now - days * DAY))

    def queue(self, **records):
        (self.folder / "uploads.json").write_text(json.dumps(records), encoding="utf-8")

    def stamp(self, seconds_from_now):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.now + seconds_from_now))

    def test_only_what_is_on_youtube_goes_and_the_history_stays(self):
        for name in ("posted.mp4", "waiting.mp4"):
            self.age(self.folder / name, 1)
        self.queue(**{"posted.mp4": {"state": "uploaded", "uploaded": self.stamp(-3600), "url": "https://youtu.be/s1"},
                      "remade.mp4": {"state": "uploaded", "uploaded": self.stamp(-3600), "url": "https://youtu.be/s2"},   # rendered again since
                      "waiting.mp4": {"state": "queued"}})
        freed = housekeeping.sweep(self.now)
        self.assertEqual(freed["posted_clips"], 1000)
        self.assertEqual(sorted(p.name for p in self.folder.glob("*.mp4")), ["mine.mp4", "remade.mp4", "waiting.mp4"])
        clips = {c["file"]: c for c in P.load_project("house1")["clips"]}
        self.assertEqual((clips["posted.mp4"]["removed_after_upload"], clips["posted.mp4"]["youtube_url"]), (True, "https://youtu.be/s1"))
        self.assertNotIn("removed_after_upload", clips["remade.mp4"])
        self.assertTrue((self.folder / "uploads.json").is_file())

    def test_switched_off_nothing_posted_is_deleted(self):
        self.age(self.folder / "posted.mp4", 1)
        self.queue(**{"posted.mp4": {"state": "uploaded", "uploaded": self.stamp(-3600)}})
        with mock.patch.dict(os.environ, {"LOCAL_DELETE_UPLOADED": "0"}):
            self.assertEqual(housekeeping.sweep(self.now)["posted_clips"], 0)
        self.assertTrue((self.folder / "posted.mp4").is_file())

    def test_old_saved_data_goes_recent_data_and_projects_stay(self):
        old, new = self.tmp / "cache" / "old", self.tmp / "cache" / "new"
        for folder in (old, new, new / "sections", self.tmp / ".llm_cache"):
            folder.mkdir(parents=True, exist_ok=True)
        files = {"old_tr": old / "transcript.json", "new_tr": new / "transcript.json", "part": new / "sections" / "a.mp4",
                 "answer": self.tmp / ".llm_cache" / "a.json", "words": self.tmp / "cache" / "roman_words.json"}
        for f in files.values():
            f.write_bytes(b"x" * 10)
        for key in ("old_tr", "answer", "words"):
            self.age(files[key], 5)
        self.age(old, 5)
        self.age(files["part"], 1)
        housekeeping.sweep(self.now)
        self.assertEqual({k: f.exists() for k, f in files.items()}, {"old_tr": False, "new_tr": True, "part": False, "answer": False, "words": True})
        self.assertTrue((self.folder / "project.json").is_file())
        self.assertTrue((self.folder / "mine.mp4").is_file())

    def test_cleaned_captions_are_fetched_again_when_a_clip_needs_them(self):
        got = {"segments": [{"start": 0, "end": 1, "text": "hi"}], "language": "en", "duration": 100.0, "source": "t"}
        (P.cache_dir("house1") / "transcript.json").unlink(missing_ok=True)
        project = P.load_project("house1")
        project.pop("transcript", None)
        with mock.patch("shorts_generator.local.transcript.get_transcript", lambda *a, **k: got) as fetch:
            self.assertEqual(pipeline._transcript_for(project), got)
            self.assertIsNotNone(fetch)
        local = {**project, "source": {"kind": "file", "url": "c:/x.mp4"}}
        with self.assertRaises(KeyError):
            pipeline._transcript_for(local)


# --------------------------------------------------------------------------------------------- #
class RunningLikeAnApp(unittest.TestCase):
    H = {"X-Studio": "1"}

    def setUp(self):
        self.c = webapp.app.test_client()
        self.flag = webapp.ROOT / "output" / "state" / "quit.flag"

    def test_quit_is_refused_while_something_runs_and_otherwise_closes(self):
        busy = mock.Mock(status="running")
        with mock.patch.dict(webapp.JOBS, {"j": busy}, clear=True):
            self.assertEqual(self.c.post("/api/quit", headers=self.H).status_code, 409)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(webapp, "ROOT", Path(tmp)), mock.patch.dict(webapp.JOBS, {}, clear=True), \
                mock.patch.object(webapp.threading, "Timer") as timer:
            self.assertEqual(self.c.post("/api/quit", headers=self.H).status_code, 200)
            self.assertTrue((Path(tmp) / "output" / "state" / "quit.flag").is_file())
            self.assertIs(timer.call_args[0][1], webapp._exit_process)
        self.assertEqual(self.c.post("/api/quit").status_code, 403)                      # not from another site

    def test_autostart_and_memes_folder_routes(self):
        calls = []
        with mock.patch.object(desktop, "supported", lambda: True), mock.patch.object(desktop, "set_autostart", calls.append), \
                mock.patch.object(desktop, "autostart_enabled", lambda: True), mock.patch.object(webapp, "_explorer_open", calls.append):
            self.assertEqual(self.c.post("/api/app/autostart", json={"on": False}, headers=self.H).get_json(), {"autostart": True})
            self.assertEqual(self.c.get("/api/app").get_json(), {"supported": True, "autostart": True})
            self.assertEqual(self.c.post("/api/memes/open", headers=self.H).status_code, 200)
        self.assertEqual((calls[0], calls[1]), (False, memes.folder()))

    def test_settings_reach_the_page(self):
        cfg = self.c.get("/api/config").get_json()
        self.assertEqual((cfg["facecam_bg"], cfg["delete_uploaded"], cfg["cache_days"]), ("white", False, 3))
        self.assertIn("total", cfg["memes"])
        for bad in ({"facecam_bg": "rainbow"}, {"cache_days": "0"}, {"delete_uploaded": "maybe"}):
            self.assertEqual(self.c.post("/api/settings", json=bad, headers=self.H).status_code, 400, bad)

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_start_with_windows_writes_and_removes_one_registry_value(self):
        import winreg

        key = r"Software\ShortsStudioTests"
        self.addCleanup(lambda: winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key))
        with mock.patch.object(desktop, "RUN_KEY", key):
            desktop.set_autostart(True)
            self.assertTrue(desktop.autostart_enabled())
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
                value = winreg.QueryValueEx(k, desktop.RUN_VALUE)[0]
            self.assertIn("pythonw.exe", value)
            self.assertTrue(value.endswith("--background"))
            desktop.set_autostart(False)
            desktop.set_autostart(False)                       # twice is fine
            self.assertFalse(desktop.autostart_enabled())

    @unittest.skipUnless(os.name == "nt", "Windows only")
    def test_desktop_icon_is_created_and_removed(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(desktop, "set_autostart", lambda on: None):
            desktop.install([Path(tmp)])
            link = Path(tmp) / "Shorts Studio.lnk"
            self.assertTrue(link.is_file())
            self.assertGreater(link.stat().st_size, 200)
            desktop.uninstall([Path(tmp)])
            self.assertFalse(link.exists())


class TheKeeper(unittest.TestCase):
    def setUp(self):
        loader = importlib.machinery.SourceFileLoader("launcher_under_test", str(Path(webapp.ROOT) / "launcher.pyw"))
        self.m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
        loader.exec_module(self.m)
        tmp = Path(tempfile.mkdtemp(prefix="keeper_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        self.m.QUIT_FLAG = tmp / "quit.flag"
        self.starts = 0
        for p in (mock.patch.object(self.m, "_only_keeper", lambda: True), mock.patch.object(self.m, "running", lambda *a: False),
                  mock.patch.object(self.m.time, "sleep", lambda s: None)):
            p.start()
            self.addCleanup(p.stop)

    def server(self, quit_after):
        def start():
            self.starts += 1
            child = mock.Mock()
            child.wait = lambda: self.m.QUIT_FLAG.write_text("quit") if self.starts == quit_after else None
            return child

        return mock.patch.object(self.m, "start_server", start)

    def test_a_crash_restarts_it_and_quit_stops_it(self):
        with self.server(quit_after=3):
            self.m.keep_running()
        self.assertEqual(self.starts, 3)

    def test_it_gives_up_instead_of_looping_forever(self):
        with self.server(quit_after=999):
            self.m.keep_running()
        self.assertEqual(self.starts, 5)


if __name__ == "__main__":
    unittest.main()

"""Memes that take over the screen: the video waits for them, the AI picks them by what is said, and the file stays in step."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import tests  # noqa: F401  (blank keys, temp output folder)
from shorts_generator import pipeline
from shorts_generator.local import clipper as K
from shorts_generator.local import editing as E
from shorts_generator.local import memes, publish, quality, romanize

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def meme(name, how="cut", mood="laugh", auto=True, kind="video"):
    return memes.Meme(f"/nowhere/{name}.mp4", mood, kind, id=name, use=f"when {name}", how=how, auto=auto)


# --------------------------------------------------------------------------------------------- #
class WaitingTimeline(unittest.TestCase):
    def test_a_wait_pushes_everything_after_it_back(self):
        tl = E.Timeline([(0, 4), (6, 10)], inserts=[(3, 2.0)])
        self.assertEqual((tl.base_duration, tl.duration), (8.0, 10.0))
        self.assertEqual((tl.to_out(2.0), tl.to_out(3.0), tl.to_out(3.0, end=True), tl.to_out(7.0)), (2.0, 5.0, 3.0, 7.0))
        self.assertEqual(tl.pieces(), [("video", 0, 3), ("wait", 2.0, 0.0), ("video", 3, 4), ("video", 6, 10)])
        self.assertEqual(tl.joins(), [5.0, 6.0])                                # back from the meme, and the cut pause

    def test_waits_on_a_cut_and_at_the_very_end(self):
        tl = E.Timeline([(0, 4), (6, 10)], inserts=[(10, 1.5), (4, 1.0)])
        self.assertEqual(tl.pieces(), [("video", 0, 4), ("wait", 1.0, 0.0), ("video", 6, 10), ("wait", 1.5, 0.0)])
        self.assertEqual(tl.duration, 10.5)

    def test_a_word_ends_before_the_wait_and_the_next_starts_after_it(self):
        tl = E.Timeline([(0, 10)], inserts=[(3, 2.0)])
        words = E.remap_words([(2.5, 3.4, "punch"), (3.0, 3.5, "next")], tl)     # captions "end" at the next word's start
        self.assertEqual(words[0], (2.5, 3.0, "punch"))
        self.assertEqual(words[1][0], 5.0)

    def test_the_cut_is_made_after_the_sentence_not_inside_a_word(self):
        words = [(4.0, 4.4, "aur"), (4.4, 4.8, "phir"), (4.8, 6.2, "gaya."), (6.2, 6.6, "ab")]
        stop = E.phrase_end(5.0, words, None)
        self.assertAlmostEqual(stop, E.spoken_span(words[2])[1])
        self.assertEqual(E.phrase_end(20.0, words, None), 20.0)                  # nothing said around there


class Choosing(unittest.TestCase):
    def setUp(self):
        self.length = mock.patch.object(memes, "play_seconds", return_value=2.0)
        self.length.start()
        self.addCleanup(self.length.stop)
        self.library = [meme("laugh1"), meme("laugh2"), meme("bye", how="end"), meme("later", how="later", mood="awkward"),
                        meme("faah", how="over", mood="shock", kind="sound"), meme("neta", auto=False)]

    def ids(self, moments, duration=40.0):
        return [(c.meme.id, c.t) for c in memes.plan(moments, self.library, duration)]

    def test_the_named_meme_is_used_and_a_switched_off_one_never(self):
        self.assertEqual(self.ids([(8.0, "", 0, "laugh2"), (20.0, "", 0, "neta"), (30.0, "", 0, "unknown")]), [("laugh2", 8.0)])
        for seed in range(12):
            got = memes.plan([(8.0, "laugh", 0), (20.0, "laugh", 0)], self.library, 40.0, seed=str(seed))
            self.assertNotIn("neta", [c.meme.id for c in got])

    def test_a_closing_meme_goes_to_the_end_wherever_it_was_named(self):
        self.assertEqual(self.ids([(5.0, "", 0, "bye")]), [("bye", 40.0)])
        self.assertEqual(self.ids([(10.0, "awkward", 0)]), [])                  # end / later ones are never picked by mood

    def test_a_weak_reason_never_stops_the_video(self):
        self.assertEqual(self.ids([(10.0, "laugh", 2.0)]), [])
        self.assertEqual(self.ids([(10.0, "shock", 2.0)]), [("faah", 10.0)])
        self.assertEqual(self.ids([(10.0, "later", 0.4)]), [("later", 10.0)])

    def test_a_sound_does_not_push_out_the_closing_meme(self):
        got = self.ids([(10.0, "", 0, "faah"), (19.0, "", 0, "laugh1"), (26.0, "", 0, "bye")], 26.0)
        self.assertEqual(got, [("faah", 10.0), ("laugh1", 19.0), ("bye", 26.0)])

    def test_memes_never_become_the_clip(self):
        moments = [(float(t), "laugh", 0) for t in (6, 13, 20, 27)]
        self.assertEqual(len(self.ids(moments, 20.0)), 1)                        # a short clip waits only once
        self.assertLessEqual(len(self.ids(moments, 60.0)), 3)

    def test_memes_never_push_a_clip_over_a_minute(self):
        moments = [(float(t), "laugh", 0) for t in (6, 20, 40)]
        self.assertEqual(len(self.ids(moments, 50.0)), 2)                        # 54 s with them
        self.assertEqual(len(self.ids(moments, 56.0)), 1)                        # room for one 2 s meme only
        self.assertEqual(self.ids(moments, 58.0), [])
        self.assertEqual(self.ids([(10.0, "", 0, "faah"), (30.0, "", 0, "laugh1")], 75.0), [("faah", 10.0)])     # sounds add no time

    def test_a_clip_over_a_minute_ends_after_the_last_sentence_that_fits(self):
        segments = [{"start": float(t), "end": t + 4.0, "text": "x"} for t in range(100, 200, 4)]
        self.assertEqual(pipeline._short_end(100.0, 150.0, segments), 150.0)
        self.assertEqual(pipeline._short_end(100.0, 190.0, segments), 156.0)
        self.assertEqual(pipeline._short_end(100.0, 190.0, None), 159.0)
        self.assertTrue(pipeline._is_wide("16:9"))
        self.assertFalse(pipeline._is_wide("9:16") or pipeline._is_wide("1:1"))

    def test_moments_are_never_asked_for_longer_than_a_short(self):
        for asked, got in (((20, 60), (20, 59)), ((45, 90), (45, 59)), ((70, 180), (54, 59)), ((15, 30), (15, 30))):
            options = pipeline.AnalyzeOptions(min_seconds=asked[0], max_seconds=asked[1])
            pipeline._validate_analyze(options)
            self.assertEqual((options.min_seconds, options.max_seconds), got)

    def test_the_ai_choice_wins_over_guesses(self):
        tl = E.Timeline([(0, 40)])
        reactions = [(30.0, "laugh")]
        named = E.meme_moments([{"at": 5, "mood": "sad"}], [], [20.0], tl, reactions, picks=[{"at": 9.0, "id": "laugh1"}])
        self.assertEqual(named, [(9.0, "", 0.0, "laugh1")])
        none_fit = E.meme_moments([{"at": 5, "mood": "sad"}], [], [], tl, reactions, picks=[])
        self.assertEqual(none_fit, [(30.0, "laugh", 0.5)])                       # asked and said no: its moods are not reused
        old_kit = E.meme_moments([{"at": 5, "mood": "sad"}], [], [], tl)
        self.assertEqual(old_kit, [(5.0, "sad", 0.0)])

    def test_a_long_cut_out_wait_is_a_few_moments_later(self):
        tl = E.Timeline([(0, 10), (15, 30)])
        self.assertEqual(E.meme_moments(None, [], [], tl), [(10.0, "later", 0.4)])

    def test_the_plan_makes_room_and_drops_the_end_card_for_a_closing_meme(self):
        words = [(float(t), t + 1.0, f"w{t}.") for t in range(0, 30)]
        plan = E.build_plan(30.0, 25.0, words, None, E.effects_for("viral"), {"hook": "H", "memes": [{"at": 8.0, "id": "laugh1"}, {"at": 30, "id": "bye"}]},
                            library=self.library)
        self.assertEqual([c.meme.id for c in plan.memes], ["laugh1", "bye"])
        self.assertAlmostEqual(plan.duration, 34.0, places=2)
        self.assertEqual(plan.cta, "")
        self.assertEqual(plan.summary()["removed_seconds"], 0.0)
        first = plan.memes[0]
        self.assertAlmostEqual(first.t, first.at, places=2)
        self.assertFalse([w for w in plan.words if first.t + 0.01 < w[0] < first.t + first.dur - 0.01], "words are shown during a meme")
        self.assertFalse([t for t, _ in plan.sounds if first.t <= t < first.t + first.dur])


# --------------------------------------------------------------------------------------------- #
class AskingTheAI(unittest.TestCase):
    PROJECT = {"title": "Stream", "source": {}, "candidates": [{"title": "t", "start_time": 100.0, "end_time": 130.0}]}
    TRANSCRIPT = {"language": "hi", "duration": 200.0, "segments": [{"start": 101.0, "end": 105.0, "text": "arre kya bola"}]}
    MENU = [("kya_bola", "someone says something daring"), ("faah", "a sudden fail")]

    def test_the_memes_are_offered_by_use_and_only_listed_ids_are_kept(self):
        prompts = []

        def llm(prompt):
            prompts.append(prompt)
            return json.dumps({"clips": [{"id": 0, "title": "T", "memes": [{"at": 6.5, "id": "kya_bola"}, {"at": 9, "id": "made_up"},
                                                                           {"at": 999, "id": "faah"}, "junk"]}]})

        kit = publish.make_publish_kit(self.PROJECT, self.TRANSCRIPT, [0], llm, with_edit=True, meme_menu=self.MENU)[0]
        self.assertIn("kya_bola: someone says something daring", prompts[0])
        self.assertEqual(kit["memes"], [{"t": 106.5, "id": "kya_bola"}])
        plain = publish.make_publish_kit(self.PROJECT, self.TRANSCRIPT, [0], llm, with_edit=True)[0]
        self.assertNotIn("MEME LIST", prompts[1])
        self.assertNotIn("memes", plain)

    def test_the_choice_reaches_the_edit_with_clip_times(self):
        hints = pipeline._hints_for({"kit": {"memes": [{"t": 106.5, "id": "kya_bola"}, {"t": 20.0, "id": "faah"}]}}, 100.0, 130.0)
        self.assertEqual(hints["memes"], [{"at": 6.5, "id": "kya_bola"}])
        self.assertEqual(romanize.hints(hints)["memes"], hints["memes"])
        self.assertNotIn("memes", romanize.hints(pipeline._hints_for({"kit": {}}, 0, 10)))

    def test_old_notes_without_memes_are_asked_again_once(self):
        project = {"candidates": [{"kit": {"style": publish.STYLE, "popups": []}},
                                  {"kit": {"style": publish.STYLE, "popups": [], "memes": []}}]}
        with mock.patch.object(pipeline, "publish_kit_local") as ask:
            pipeline._edit_notes_for(project, [0, 1], wants_memes=True)
            ask.assert_called_once_with(project, [0], with_edit=True)
            ask.reset_mock()
            pipeline._edit_notes_for(project, [0, 1], wants_memes=False)
            ask.assert_not_called()


# --------------------------------------------------------------------------------------------- #
class TheFolders(unittest.TestCase):
    def test_both_folders_are_read_and_the_catalog_describes_files_by_name(self):
        tmp = Path(tempfile.mkdtemp(prefix="memefolders_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "memes" / "sad").mkdir(parents=True)
        (tmp / "Memes Assets").mkdir()
        cv2.imwrite(str(tmp / "memes" / "sad" / "tear.png"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(tmp / "Memes Assets" / "Boss-Entry.png"), np.zeros((8, 8, 3), np.uint8))
        catalog = tmp / "catalog.json"
        catalog.write_text(json.dumps({"_about": "x", "boss-entry.png": {"id": "boss", "mood": "hype", "use": "a boss arrives", "auto": False}}))
        with mock.patch.dict(os.environ, {"LOCAL_MEME_DIR": ""}), mock.patch.object(memes, "ROOT", tmp), mock.patch.object(memes, "CATALOG", catalog):
            found = {m.name: (m.id, m.mood, m.auto) for m in memes.scan()}
            self.assertEqual(found, {"tear.png": ("tear", "sad", True), "Boss-Entry.png": ("boss", "hype", False)})
            self.assertEqual(memes.menu(memes.scan()), [("tear", "tear (sad)")])
            self.assertEqual((memes.summary()["usable"], memes.summary()["off"]), (1, ["Boss-Entry.png"]))
            self.assertEqual(memes.folder(), tmp / "Memes Assets")

    def test_the_shipped_catalog_is_sound(self):
        known = memes.catalog()
        self.assertGreaterEqual(len(known), 36)
        self.assertEqual(len({v["id"] for v in known.values()}), len(known), "two memes answer to the same id")
        for name, entry in known.items():
            self.assertEqual(name, name.lower())
            self.assertIn(entry["how"], memes.HOWS, name)
            self.assertIn(entry["mood"], memes.MOODS, name)
            self.assertTrue(0 <= entry["start"] < entry["end"] and entry["end"] - entry["start"] <= memes.LONGEST, name)


# --------------------------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class RealRenderWithMemes(unittest.TestCase):
    """A blue "talk" with a 300 Hz voice; a red meme with 1500 Hz after second 8, a green closing one with 800 Hz."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="memerender_"))
        folder = cls.tmp / "lib"
        folder.mkdir()
        ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=640x360:r=25:d=25", "-f", "lavfi", "-i", "sine=frequency=300:duration=25",
               "-shortest", "-pix_fmt", "yuv420p", str(cls.tmp / "talk.mp4"))
        for name, color, tone in (("red", "red", 1500), ("bye", "0x00ff00", 800)):
            ffmpeg("-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=25:d=3", "-f", "lavfi", "-i", f"sine=frequency={tone}:duration=3",
                   "-shortest", "-pix_fmt", "yuv420p", str(folder / f"{name}.mp4"))
        catalog = cls.tmp / "catalog.json"
        catalog.write_text(json.dumps({"red.mp4": {"id": "red", "mood": "laugh", "how": "cut", "start": 0.5, "end": 2.0},
                                       "bye.mp4": {"id": "bye", "mood": "awkward", "how": "end", "start": 0.0, "end": 1.2}}))
        words = [{"start": float(t), "end": t + 1.0, "text": f"word{t}."} for t in range(25)]
        segments = [{"start": 0.0, "end": 25.0, "text": " ".join(w["text"] for w in words), "words": words}]
        cls.out, cls.info = cls.tmp / "out.mp4", {}
        with mock.patch.dict(os.environ, {"LOCAL_MEME_DIR": str(folder), "LOCAL_ENCODER": "cpu"}), mock.patch.object(memes, "CATALOG", catalog):
            K.render_clip(str(cls.tmp / "talk.mp4"), str(cls.out), 0.0, 25.0, K.RenderOptions(edit="viral"), transcript_segments=segments,
                          title="Talk", precut=True, hints={"hook": "WAIT", "memes": [{"at": 8.0, "id": "red"}, {"at": 25.0, "id": "bye"}]},
                          info=cls.info)
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(cls.out), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                             capture_output=True).stdout
        cls.sound = np.frombuffer(pcm, dtype=np.int16).astype(float)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def tone(self, t):
        piece = self.sound[int((t - 0.2) * 8000):int((t + 0.2) * 8000)]
        spectrum = np.abs(np.fft.rfft(piece * np.hanning(len(piece))))
        return float(np.argmax(spectrum) * 8000 / len(piece))

    def colour(self, t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(self.out), "-frames:v", "1", "-vf", "crop=200:100:440:910,scale=1:1",
                              "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
        return ("red", "green", "blue")[int(np.argmax(np.frombuffer(raw[:3], dtype=np.uint8)))]

    def test_the_clip_grows_by_the_memes_and_stays_in_step(self):
        self.assertEqual(self.info["edit"]["memes"], 2)
        self.assertAlmostEqual(self.info["duration"], 25.0 + 1.5 + 1.2, delta=0.1)
        self.assertEqual(quality.verify(str(self.out), self.info["size"], self.info["duration"], self.info["has_audio"]), [])
        streams = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(self.out)],
                                            capture_output=True, text=True).stdout)["streams"]
        lengths = {s["codec_type"]: float(s["duration"]) for s in streams}
        self.assertAlmostEqual(lengths["audio"], lengths["video"], delta=0.12)

    def test_the_video_waits_while_the_meme_plays_with_its_own_sound(self):
        end = self.info["duration"]
        start = 8.0                                                  # "word7." ends just before second 8
        self.assertEqual([self.colour(t) for t in (6.0, start + 0.75, start + 2.2, end - 0.6)], ["blue", "red", "blue", "green"])
        for t, hz in ((6.0, 300), (start + 0.75, 1500), (start + 2.4, 300), (end - 0.6, 800)):
            self.assertAlmostEqual(self.tone(t), hz, delta=40, msg=f"at {t:.1f}s")


if __name__ == "__main__":
    unittest.main()

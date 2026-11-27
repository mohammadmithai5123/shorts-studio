"""Automatic editing, per-shot framing and the render engine.

Planning is tested with made-up loudness and words; framing with made-up face positions; the engine with real ffmpeg renders
of a synthetic video (speech-like tone bursts with silences between them).
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
from werkzeug.exceptions import BadRequest

import app as webapp
from shorts_generator import pipeline
from shorts_generator.local import captions as C
from shorts_generator.local import clipper as K
from shorts_generator.local import editing as E
from shorts_generator.local import encoder, framing, health, llm, overlays, publish, quality, sfx
from shorts_generator.local import project as P

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def levels(pattern, hop=E.HOP):
    """[(seconds, dB), ...] -> a loudness array."""
    return np.concatenate([np.full(int(round(s / hop)), db, dtype=float) for s, db in pattern])


def words(*items):
    return [(float(a), float(b), t) for a, b, t in items]


class Pauses(unittest.TestCase):
    def test_real_pauses_are_found_and_spoken_words_protected(self):
        db = levels([(2, -20), (1.5, -65), (2, -20), (0.3, -65), (2, -20)])
        found = E.find_pauses(db, words((0, 2, "hello"), (3.5, 5.5, "world")), 7.8)
        self.assertEqual(len(found), 1)                                          # the 0.3 s breath stays
        a, b = found[0]
        self.assertTrue(2.0 < a < 2.3 and 3.2 < b < 3.5, found)
        # a word spoken quietly inside the silence protects its spoken part
        found = E.find_pauses(db, words((2.2, 3.5, "psst")), 7.8)
        self.assertTrue(all(b <= 2.2 or a >= 2.45 for a, b in found), found)

    def test_music_or_constant_noise_is_never_cut(self):
        self.assertEqual(E.find_pauses(levels([(8, -22)]), [], 8), [])
        self.assertEqual(E.find_pauses(None, [], 8), [])

    def test_kept_ranges_sit_on_the_frame_grid_and_never_remove_too_much(self):
        keep = E.keep_ranges([(1.013, 2.51), (4.0, 4.9)], 10.0, 25.0)
        self.assertEqual(len(keep), 3)
        for a, b in keep:
            self.assertAlmostEqual(a * 25, round(a * 25), places=6)
            self.assertAlmostEqual(b * 25, round(b * 25), places=6)
        keep = E.keep_ranges([(0.5, 4.5), (5.0, 9.5)], 10.0, 25.0)                # 8.5 s of 10 are quiet
        self.assertGreaterEqual(sum(b - a for a, b in keep), 10 * (1 - E.MAX_REMOVED) - 0.05)

    def test_timeline_maps_source_time_to_output_time(self):
        tl = E.Timeline([(0, 2), (3, 5)])
        self.assertEqual(tl.duration, 4)
        self.assertEqual((tl.to_out(1.5), tl.to_out(2.5), tl.to_out(3.5), tl.to_out(99)), (1.5, 2.0, 2.5, 4.0))
        self.assertEqual(tl.joins(), [2.0])
        self.assertFalse(tl.kept(2.5))
        out = E.remap_words(words((0.5, 1.0, "a"), (3.2, 3.6, "b")), tl)
        self.assertEqual([(round(a, 2), t) for a, _, t in out], [(0.5, "a"), (2.2, "b")])


class Planning(unittest.TestCase):
    def test_zooms_alternate_at_sentences_and_punch_on_peaks(self):
        tl = E.Timeline([(0, 30)])
        w = words((0, 1, "one."), (4, 5, "two."), (8, 9, "three."), (12, 13, "four"))
        steps, punches = E.zoom_plan(tl, w, [20.0], strong=True)
        self.assertEqual([lv for _, lv in steps][:4], [1.0, 1.12, 1.0, 1.12])
        self.assertTrue(all(b - a >= 2.5 for (a, _), (b, _) in zip(steps, steps[1:])))
        self.assertEqual(punches[0][2], 1.22)
        gentle, none = E.zoom_plan(tl, w, [20.0], strong=False)
        self.assertEqual((max(lv for _, lv in gentle), none), (1.07, []))
        plan = E.EditPlan(effects={}, timeline=tl, source_duration=30, steps=steps, punches=punches)
        self.assertAlmostEqual(plan.zoom_at(20.4), 1.22, places=3)
        self.assertGreater(plan.zoom_at(20.0), 1.0)

    def test_popups_prefer_the_ai_then_reactions_then_words_and_keep_clear_of_the_hook_and_end(self):
        tl = E.Timeline([(0, 60)])
        hints = [{"at": 10, "emoji": "🏠", "text": "ghar"}, {"at": 1.0, "emoji": "x"}, {"at": 59.5, "emoji": "y"}, {"at": "bad"}]
        w = words((20, 20.5, "paisa"), (30, 30.5, "प्यार"), (40, 40.5, "table"))
        got = E.popups_from(hints, w, [50.0], tl, reactions=[(45.0, "laugh"), (11.0, "applause")])
        by_time = {round(p.t): p.emoji for p in got}
        self.assertEqual(by_time[10], "🏠")
        self.assertEqual(by_time[45], "😂")                                       # the captions' own [laughter] note
        self.assertEqual(by_time[20], "💰")
        self.assertEqual(by_time[30], "❤️")                                       # Hindi keyword
        self.assertNotIn(11, by_time)                                             # too close to the AI popup
        self.assertTrue(all(2.8 <= p.t <= 58 for p in got))
        self.assertEqual([p.side for p in got], [i % 2 for i in range(len(got))])
        self.assertLessEqual(len(E.popups_from([{"at": 3 + i * 3, "emoji": "x"} for i in range(30)], [], [], tl)), 8)

    def test_build_plan_cuts_and_summarises(self):
        db = levels([(3, -20), (2, -65), (3, -20), (2, -65), (3, -20)])
        w = words((0, 3, "hello"), (5, 8, "paisa"), (10, 13, "bye"))
        plan = E.build_plan(13.0, 25.0, w, db, E.effects_for("viral"), {"hook": "Big news"}, "title")
        self.assertLess(plan.duration, 10.5)
        s = plan.summary()
        self.assertEqual(s["cuts"], 2)
        self.assertGreater(s["removed_seconds"], 2.5)
        self.assertEqual((plan.hook, plan.cta), ("Big news", ""))                  # no follow card by itself
        asked = E.build_plan(13.0, 25.0, w, db, E.effects_for("viral", {"cta": True}), {"hook": "Big news 😱"}, "title")
        self.assertEqual((asked.hook, asked.cta), ("Big news", "FOLLOW FOR MORE"))
        self.assertTrue(plan.sounds and plan.sounds[0][1] == "whoosh")
        off = E.build_plan(13.0, 25.0, w, db, E.effects_for("off"))
        self.assertEqual((off.duration, off.popups, off.sounds, off.hook, off.steps), (13.0, [], [], "", []))
        clean = E.build_plan(13.0, 25.0, w, db, E.effects_for("clean", {"jumpcut": False}), title="A very long headline " * 5)
        self.assertEqual((clean.duration, clean.popups, clean.sounds), (13.0, [], []))
        self.assertLessEqual(len(clean.hook), 46)

    def test_effect_overrides(self):
        self.assertFalse(E.effects_for("viral", {"popups": False})["popups"])
        self.assertTrue(E.effects_for("off", {"progress": True})["progress"])
        self.assertEqual(E.effects_for("nonsense"), E.effects_for("off"))


class CaptionsForEdits(unittest.TestCase):
    def test_sound_notes_are_removed_and_reactions_remembered(self):
        w = words(*[(i, i + 1, t) for i, t in enumerate("तो फिर [हंसी] अरे [हौसला अफ़ज़ाई की आवाज़] ठीक (music) [Applause]".split())])
        spoken, reactions = C.split_annotations(w)
        self.assertEqual([t for _, _, t in spoken], ["तो", "फिर", "अरे", "ठीक"])
        self.assertEqual([k for _, k in reactions], ["laugh", "applause", "applause"])
        spoken, _ = C.split_annotations(words((0, 1, "आवाज़]"), (1, 2, "माय")))          # note started on the previous line
        self.assertEqual([t for _, _, t in spoken], ["माय"])

    def test_sound_notes_never_reach_the_screen_or_the_edit(self):
        segs = [{"start": 0.0, "end": 4.0, "text": "", "words": [{"start": float(i), "end": i + 1.0, "text": t}
                                                                 for i, t in enumerate("अरे [हंसी] भाई [Music]".split())]}]
        self.assertEqual([t for _, _, t in K.clip_words(segs, 0.0, 4.0)], ["अरे", "भाई"])
        self.assertEqual(K.clip_reactions(segs, 0.0, 4.0), [(1.0, "laugh")])
        for style in ("pop", "classic", "karaoke"):
            doc = C.build_ass(segs, 0.0, 4.0, 1080, 1920, style=style)
            self.assertNotIn("[हंसी]", doc.upper() if style == "pop" else doc, style)
            self.assertNotIn("Music", doc, style)
            self.assertNotIn("MUSIC", doc, style)

    def test_pop_captions_highlight_the_spoken_word_and_key_words(self):
        segs = [{"start": 0, "end": 3, "text": "", "words": [{"start": i * 0.5, "end": i * 0.5 + 0.5, "text": t}
                                                           for i, t in enumerate("we won 100 crore".split())]}]
        doc = C.build_ass(segs, 0, 10, 1080, 1920, style="pop", keywords=["Crore"], hook="Crazy {moment}", cta="Follow for more")
        events = [ln for ln in doc.splitlines() if ln.startswith("Dialogue")]
        pop = [ln for ln in events if ",Pop," in ln]
        self.assertEqual(len(pop), 4)                                              # one line per spoken word
        self.assertIn(r"{\c&H00FFFF&\fscx112\fscy112}WE", pop[0])
        self.assertIn(r"{\c&H40FF40&\fscx100\fscy100}100", pop[0])             # a number is highlighted even when not spoken
        self.assertIn(r"{\c&H00FFFF&\fscx112\fscy112}CRORE", pop[3])
        self.assertTrue(any(",Hook," in ln and "CRAZY MOMENT" in ln for ln in events))        # ASS syntax stripped
        self.assertTrue(any(",Cta," in ln and "FOLLOW FOR MORE" in ln for ln in events))

    def test_devanagari_words_are_sized_up(self):
        segs = [{"start": 0, "end": 1, "text": "", "words": [{"start": 0, "end": 1, "text": "घर"}]}]
        self.assertIn(r"\fnNirmala UI\fs", C.build_ass(segs, 0, 2, 1080, 1920, style="pop"))

    def test_caption_cards_never_span_a_pause(self):
        segs = K._segments_from_words(words((0, 0.4, "one"), (0.4, 5.0, "two"), (5.0, 5.4, "three")))
        self.assertEqual([s["text"] for s in segs], ["one two", "three"])


class Drawing(unittest.TestCase):
    def test_popups_stay_on_screen_and_blend(self):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        sprite = np.full((100, 900, 4), 255, dtype=np.uint8)                      # very wide: would stick out on the left
        overlays.draw_popup(frame, sprite, age=0.6, duration=1.4, side=0)
        cols = np.where(frame.any(axis=(0, 2)))[0]
        self.assertGreaterEqual(cols[0], 4)
        self.assertLess(cols[-1], 1080 - 4)
        overlays.blend(frame, sprite, -2000, -2000)                               # completely outside: ignored
        self.assertEqual(overlays.pop_scale(1.0), 1.0)
        self.assertLess(overlays.pop_scale(0.0), 0.5)
        self.assertEqual(overlays.pop_alpha(1.4, 1.4), 0.0)

    def test_progress_bar_grows(self):
        frame = np.full((1920, 1080, 3), 100, dtype=np.uint8)
        overlays.draw_progress(frame, 0.5)
        self.assertEqual(tuple(frame[2, 100]), (0, 213, 255))
        self.assertNotEqual(tuple(frame[2, 900]), (0, 213, 255))

    def test_emoji_are_stickers_and_non_latin_badges_are_skipped(self):
        self.assertIsNone(overlays.text_sprite("घर", 60))
        self.assertIsNone(overlays.popup_sprite("", "", 1080))
        sprite = overlays.emoji_sprite("🎵", 160)
        if sprite is None:
            self.skipTest("no colour emoji font on this machine")
        rim = sprite[(sprite[:, :, 3] > 200)]
        self.assertTrue((rim[:, :3] > 235).all(axis=1).sum() > 100)              # a white border around the dark note
        both = overlays.popup_sprite("😂", "NO WAY", 1080)
        self.assertGreater(both.shape[0], sprite.shape[0])

    def test_long_badges_are_shrunk_to_fit(self):
        sprite = overlays.popup_sprite("", "IMAGE SUDHAR GAYI BHAI SAHAB", 1080)
        if sprite is not None:
            self.assertLessEqual(sprite.shape[1], int(1080 * 0.7) + 2)

    def test_sound_track(self):
        with tempfile.TemporaryDirectory() as d:
            path = sfx.write_track([(0.5, "pop"), (1.0, "ding"), (9.9, "whoosh"), (1.2, "unknown")], 3.0, os.path.join(d, "s.wav"))
            with wave.open(path) as w:
                self.assertAlmostEqual(w.getnframes() / w.getframerate(), 3.0, delta=0.01)
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.assertEqual(np.abs(pcm[:int(0.4 * 48000)]).max(), 0)
        self.assertGreater(np.abs(pcm[int(0.5 * 48000):int(0.6 * 48000)]).max(), 3000)
        self.assertLess(np.abs(pcm).max(), 32767 * 0.4)                            # never louder than speech


def det(x, y, h):
    return (int(x - h / 2), int(y - h / 2), int(h), int(h))


def analysis(per_sample, n_frames=250, size=(1280, 720), fps=25.0):
    sidx = list(range(0, n_frames, 6))
    return {"fps": fps, "n_frames": n_frames, "size": size, "roi": (0, 0, size[0], size[1]), "sidx": sidx,
            "boxes": [per_sample(i) for i in sidx], "motion": None}


class Framing(unittest.TestCase):
    def test_camera_cuts_are_found(self):
        thumbs = np.zeros((100, framing.THUMB_H, framing.THUMB_W), dtype=np.uint8)
        thumbs[40:70] = 200
        thumbs[70:] = 90
        self.assertEqual(framing.find_cuts(thumbs, 25.0), [40, 70])
        self.assertEqual(framing.shot_ranges([40, 95], 100, 25.0), [(0, 40), (40, 100)])        # the 5-frame stub joins
        noisy = (np.random.default_rng(1).integers(0, 30, (100, framing.THUMB_H, framing.THUMB_W))).astype(np.uint8)
        self.assertEqual(framing.find_cuts(noisy, 25.0), [])

    def test_one_speaker_gets_a_steady_crop(self):
        plan = framing.smart_plan(analysis(lambda i: [det(900, 250, 150)]), None, 9 / 16, 1920)
        self.assertEqual(plan["layout"], "smart")
        shot = plan["shots"][0]
        self.assertEqual(shot["kind"], "crop")
        x, y, w, h = shot["rect"]
        self.assertAlmostEqual(w / h, 9 / 16, places=3)
        self.assertTrue(x <= 900 <= x + w)
        self.assertNotIn("x0s", shot)                                              # a still speaker: the window does not move

    def test_a_stage_is_framed_around_the_performers_not_the_guards_at_the_edges(self):
        def stage(i):
            return [det(60, 200, 80), det(420, 200, 80), det(640, 190, 85), det(880, 230, 80), det(1230, 200, 80)]

        an = analysis(stage)
        thumbs = np.zeros((250, framing.THUMB_H, framing.THUMB_W), dtype=np.uint8)
        for k in range(0, 250, 2):                                                 # only the performers move
            thumbs[k, 10:50, 28:72] = 255
        plan = framing.smart_plan(an, None, 9 / 16, 1920)
        shot = plan["shots"][0]
        self.assertEqual(shot["kind"], "group")
        x, _, w, h = shot["rect"]
        self.assertLessEqual(x, 420 - 40)
        self.assertGreaterEqual(x + w, 880 + 40)                                   # nobody important is cut off
        self.assertLess(w, 1100)                                                   # and the guards at the edges do not widen it
        self.assertEqual(h, 720)

    def test_two_people_far_apart_are_stacked(self):
        plan = framing.smart_plan(analysis(lambda i: [det(300, 300, 170), det(980, 300, 170)]), None, 9 / 16, 1920)
        shot = plan["shots"][0]
        self.assertEqual(shot["kind"], "stack")
        for x, y, w, h in shot["panels"]:
            self.assertAlmostEqual(w / h, 2 * 9 / 16, places=3)
        self.assertLess(shot["panels"][0][0], shot["panels"][1][0])

    def test_nobody_means_the_whole_picture(self):
        self.assertEqual(framing.smart_plan(analysis(lambda i: []), None, 9 / 16, 1920)["layout"], "fit")

    def test_each_shot_is_framed_on_its_own(self):
        thumbs = np.zeros((250, framing.THUMB_H, framing.THUMB_W), dtype=np.uint8)
        thumbs[125:] = 220
        plan = framing.smart_plan(analysis(lambda i: [det(200, 300, 150)] if i < 125 else [det(1100, 300, 150)]), thumbs, 9 / 16, 1920)
        self.assertEqual([s["kind"] for s in plan["shots"]], ["crop", "crop"])
        a, b = plan["shots"]
        self.assertLess(a["rect"][0], 200)
        self.assertGreater(b["rect"][0] + b["rect"][2], 1100)                      # no slow slide across the cut
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in (0, 200):
            self.assertEqual(framing.compose_smart(frame, plan, i, an_roi(), 1080, 1920, zoom=1.2).shape, (1920, 1080, 3))

    def test_every_kind_of_shot_composes_to_the_output_size(self):
        frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
        for shot in ({"kind": "group", "rect": (200, 0, 800, 720)}, {"kind": "stack", "panels": [(0, 0, 640, 570), (600, 0, 640, 570)]},
                     {"kind": "fit"}):
            plan = {"layout": "smart", "shots": [{**shot, "start": 0, "end": 10, "focus": (640, 300)}]}
            self.assertEqual(framing.compose_smart(frame, plan, 3, an_roi(), 1080, 1920, zoom=1.15).shape, (1920, 1080, 3))

    def test_zoom_stays_inside(self):
        x, y, w, h = framing.zoom_rect(100, 0, 400, 720, 1.25, fx=110, fy=700)
        self.assertEqual((x, y + h <= 720, w, h), (100, True, 320, 576))


def an_roi():
    return (0, 0, 1280, 720)


class EncoderChoice(unittest.TestCase):
    def tearDown(self):
        encoder.reset()

    def test_setting_and_fallback(self):
        with mock.patch.dict(os.environ, {"LOCAL_ENCODER": "cpu"}):
            encoder.reset()
            self.assertEqual(encoder.choose(), "cpu")
        with mock.patch.dict(os.environ, {"LOCAL_ENCODER": "auto"}), mock.patch.object(encoder, "_works", lambda n: n == "qsv"):
            encoder.reset()
            self.assertEqual(encoder.choose(), "qsv")
            encoder.disable_hardware("driver crashed")
            self.assertEqual(encoder.choose(), "cpu")
        self.assertIn("h264_qsv", encoder.video_args("qsv"))
        self.assertIn("libx264", encoder.video_args("cpu"))

    def test_a_hardware_failure_is_retried_on_the_processor(self):
        used = []

        def fake_encode(in_path, out_path, an, plan, out_w, out_h, ass_text, normalize_audio, edit, name):
            used.append(name)
            if name != "cpu":
                raise K.EncodeFailed("no device")

        with mock.patch.object(K, "_encode", fake_encode), mock.patch.object(encoder, "choose", lambda: "qsv"), \
                mock.patch.object(encoder, "disable_hardware") as off:
            self.assertEqual(K._render("a", "b", {}, {}, 1, 1), "cpu")
        self.assertEqual(used, ["qsv", "cpu"])
        off.assert_called_once()


class AiNotes(unittest.TestCase):
    PROJECT = {"title": "Show", "candidates": [{"title": "c", "start_time": 100.0, "end_time": 140.0}]}
    TRANSCRIPT = {"language": "hi", "segments": [{"start": 100.0, "end": 104.0, "text": "ghar le liya"}, {"start": 150.0, "end": 151.0, "text": "later"}]}

    def test_edit_notes_are_requested_in_the_same_call_and_checked(self):
        asked = []

        def llm_fn(prompt):
            asked.append(prompt)
            return json.dumps({"clips": [{"id": 0, "title": "T", "description": "D", "hashtags": ["#a"], "hook": "HE BOUGHT A HOUSE",
                                          "popups": [{"at": 2, "emoji": "🏠", "text": "Ghar"}, {"at": 99, "emoji": "x"}, {"at": "?"}, "junk"],
                                          "keywords": ["ghar", ""]}]})

        kit = publish.make_publish_kit(self.PROJECT, self.TRANSCRIPT, [0], llm_fn, with_edit=True)[0]
        self.assertEqual(len(asked), 1)
        self.assertIn("[0.0] ghar le liya", asked[0])
        self.assertNotIn("later", asked[0])
        self.assertEqual(kit["popups"], [{"t": 102.0, "emoji": "🏠", "text": "Ghar"}])          # video time, survives a trim
        self.assertEqual((kit["hook"], kit["keywords"]), ("HE BOUGHT A HOUSE", ["ghar"]))
        plain = publish.make_publish_kit(self.PROJECT, self.TRANSCRIPT, [0], llm_fn)[0]
        self.assertNotIn("popups", plain)
        self.assertNotIn("timed lines", asked[-1])

    def test_hints_follow_a_trim(self):
        cand = {"kit": {"title": "T", "popups": [{"t": 102.0, "emoji": "a"}, {"t": 130.0, "emoji": "b"}]}}
        self.assertEqual([p["at"] for p in pipeline._hints_for(cand, 101.0, 120.0)["popups"]], [1.0])
        self.assertEqual(pipeline._hints_for({}, 0, 10), {"hook": "", "popups": [], "keywords": []})

    def test_optional_ai_work_has_a_time_budget(self):
        with llm.time_budget(30):
            self.assertLessEqual(llm._read_timeout(), 30)
            with self.assertRaises(llm.LLMError):
                llm._sleep(60)                                        # waiting longer than what is left
        self.assertEqual(llm._read_timeout(), 180.0)
        with llm.time_budget(0.01):
            time.sleep(0.05)
            with self.assertRaises(llm.LLMError):
                llm._read_timeout()


class Api(unittest.TestCase):
    def test_editing_options_are_validated(self):
        o = webapp._render_options({"edit": "viral", "effects": {"popups": False}})
        self.assertEqual((o.edit, o.effects, o.caption_style), ("viral", {"popups": False}, "pop"))
        self.assertEqual(webapp._render_options({}).edit, "off")
        c = webapp.app.test_client()
        for bad in ({"edit": "wild"}, {"effects": {"boom": True}}, {"effects": {"popups": "yes"}}, {"effects": ["popups"]}):
            with webapp.app.test_request_context(), self.assertRaises(BadRequest):
                webapp._render_options(bad)
        self.assertEqual(c.get("/api/settings").get_json()["encoder"], "auto")

    def test_the_doctor_reports_encoder_and_face_model(self):
        ids = {c["id"]: c for c in health.doctor(deep=False)["checks"]}
        self.assertIn("encoder", ids)
        self.assertEqual(ids["faces"]["detail"], "YuNet")


def make_talk_video(path: Path, size=(1280, 720)) -> None:
    """10 s: tone ("speech") 0-3, silence 3-5, tone 5-7, silence 7-8.5, tone 8.5-10."""
    audio = "aevalsrc='if(lt(t,3)+between(t,5,7)+gt(t,8.5), 0.4*sin(2*PI*330*t)*(0.7+0.3*sin(9*t)), 0.0002*random(0))':s=44100:d=10"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size[0]}x{size[1]}:rate=30:duration=10",
                    "-f", "lavfi", "-i", audio, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True)


SEGMENTS = [{"start": 0.0, "end": 3.0, "text": "hello paisa yaar", "words": [{"start": 0.0, "end": 1.0, "text": "hello"},
                                                                          {"start": 1.0, "end": 2.0, "text": "paisa"},
                                                                          {"start": 2.0, "end": 3.0, "text": "yaar"}]},
            {"start": 5.0, "end": 7.0, "text": "bahut mazak", "words": [{"start": 5.0, "end": 6.0, "text": "bahut"},
                                                                       {"start": 6.0, "end": 7.0, "text": "mazak"}]},
            {"start": 8.5, "end": 10.0, "text": "bye bye", "words": [{"start": 8.5, "end": 9.2, "text": "bye"},
                                                                   {"start": 9.2, "end": 10.0, "text": "bye"}]}]


def stream_durations(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class RealEditedRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="edit_"))
        cls.src = cls.tmp / "talk.mp4"
        make_talk_video(cls.src)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def render(self, name, **opts):
        info = {}
        out = self.tmp / name
        K.render_clip(str(self.src), str(out), 0.0, 10.0, K.RenderOptions(**opts), transcript_segments=SEGMENTS, title="Big talk",
                      precut=True, hints={"hook": "WATCH THIS", "popups": [{"at": 5.5, "emoji": "😂", "text": "haha"}]}, info=info)
        return out, info

    def test_a_viral_edit_cuts_the_pauses_and_keeps_sound_and_picture_together(self):
        out, info = self.render("viral.mp4", edit="viral")
        s = info["edit"]
        self.assertGreaterEqual(s["cuts"], 1)
        self.assertGreater(s["removed_seconds"], 1.5)
        d = stream_durations(out)
        self.assertEqual((d["video"]["width"], d["video"]["height"]), (1080, 1920))
        self.assertAlmostEqual(float(d["video"]["duration"]), info["duration"], delta=0.15)
        self.assertAlmostEqual(float(d["audio"]["duration"]), float(d["video"]["duration"]), delta=0.12)
        self.assertEqual(quality.verify(str(out), info["size"], info["duration"], info["has_audio"]), [])

    def test_the_sound_is_cut_exactly_where_the_picture_is(self):
        out, info = self.render("clean.mp4", edit="clean")             # clean: no sound effects that could fill a pause
        self.assertGreaterEqual(info["edit"]["cuts"], 1)
        db = E.audio_levels(str(out))
        longest, run = 0, 0
        for level in db:
            run = run + 1 if level < db.max() - 30 else 0
            longest = max(longest, run)
        self.assertLess(longest * E.HOP, 1.0, "a 1.5-2 s pause of the source is still in the output's sound")

    def test_editing_off_keeps_the_whole_clip(self):
        out, info = self.render("plain.mp4", edit="off")
        self.assertNotIn("edit", info)
        self.assertAlmostEqual(float(stream_durations(out)["video"]["duration"]), 10.0, delta=0.2)

    def test_the_quality_check_catches_broken_clips(self):
        black = self.tmp / "black.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30:d=3", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(black)], check=True)
        problems = quality.verify(str(black), (1080, 1920), 10.0, need_audio=True)
        text = " ".join(problems)
        for expected in ("length", "sound", "black"):
            self.assertIn(expected, text)
        self.assertIn("size", " ".join(quality.verify(str(self.src), (1080, 1920), 10.0)))
        self.assertTrue(quality.verify(str(self.tmp / "nope.mp4"), (1080, 1920), 3))


@unittest.skipUnless(HAS_FFMPEG, "needs ffmpeg")
class PipelineWithEditing(unittest.TestCase):
    """render_local with a local source file: the AI is a fake, everything else is real."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pipe_edit_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = self.tmp / "talk.mp4"
        make_talk_video(self.src, size=(640, 360))
        pid = "editpipe" + str(int(time.time() * 1000))
        meta = {"kind": "file", "id": pid, "url": str(self.src), "title": "Local", "duration": 10.0, "embeddable": False, "heatmap": None,
                "local_path": str(self.src)}
        tr = {"duration": 10.0, "language": "hi", "source": "test", "segments": SEGMENTS}
        self.p = P.new_project(meta, tr, [{"title": "moment", "start_time": 0.0, "end_time": 10.0, "score": 90}], {})
        P.save_project(self.p)
        P.cache_dir(pid).mkdir(parents=True, exist_ok=True)
        (P.cache_dir(pid) / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
        self.addCleanup(shutil.rmtree, P.project_dir(pid), True)

    def test_ai_notes_are_used_and_trims_survive_them(self):
        answer = json.dumps({"clips": [{"id": 0, "title": "AI title", "description": "d", "hashtags": ["#x"], "hook": "HOOK",
                                        "popups": [{"at": 1.0, "emoji": "🔥"}], "keywords": ["paisa"]}]})
        with mock.patch("shorts_generator.local.llm.call_local_llm", lambda prompt, use_cache=True: answer):
            records = pipeline.render_local(self.p, [0], K.RenderOptions(edit="viral", effects={"popups": True}), overrides={0: (0.0, 9.0)}, kit=True)
        r = records[0]
        self.assertIsNone(r["error"], r)
        self.assertEqual((r["start"], r["end"], r["edit"]), (0.0, 9.0, "viral"))        # the trim was not lost
        self.assertGreaterEqual(r["edit_summary"]["popups"], 1)
        saved = P.load_project(self.p["id"])
        self.assertEqual(saved["candidates"][0]["end_time"], 9.0)
        self.assertEqual(saved["candidates"][0]["kit"]["title"], "AI title")
        self.assertTrue((P.project_dir(self.p["id"]) / r["file"]).is_file())

    def test_without_the_ai_the_edit_still_happens(self):
        def down(prompt, use_cache=True):
            raise llm.LLMError("busy", kind="overloaded")

        with mock.patch("shorts_generator.local.llm.call_local_llm", down):
            records = pipeline.render_local(self.p, [0], K.RenderOptions(edit="viral"))
        self.assertIsNone(records[0]["error"])
        self.assertGreaterEqual(records[0]["edit_summary"]["cuts"], 1)

    def test_a_clip_that_fails_the_quality_check_is_not_kept(self):
        with mock.patch.object(quality, "verify", lambda *a, **k: ["the picture is completely black"]), \
                mock.patch("shorts_generator.local.llm.call_local_llm", lambda p, use_cache=True: "{}"):
            records = pipeline.render_local(self.p, [0], K.RenderOptions(edit="clean"))
        self.assertIsNone(records[0]["file"])
        self.assertIn("Quality check failed", records[0]["error"])


if __name__ == "__main__":
    unittest.main()

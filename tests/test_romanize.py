"""Subtitles in Roman letters (Hinglish / Roman Urdu). Offline: the AI is a stand-in."""
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import app as webapp
import tests  # noqa: F401  (blank keys, temp output folder)
from shorts_generator import config, pipeline
from shorts_generator.local import captions as C
from shorts_generator.local import clipper as K
from shorts_generator.local import editing as E
from shorts_generator.local import project as P
from shorts_generator.local import romanize as R
from shorts_generator.local import uploads


def fresh_cache():
    path = R._cache_file()
    path.unlink(missing_ok=True)
    R._memory.clear()


class Spelling(unittest.TestCase):
    def setUp(self):
        fresh_cache()

    def test_common_speech(self):
        self.assertEqual(R.text("मुझे नहीं पता क्या हुआ।"), "mujhe nahi pata kya hua")
        self.assertEqual(R.text("वो इस दिल की हकदार नहीं है, मेरी मोहब्बत"), "wo is dil ki hakdaar nahi hai, meri mohabbat")

    def test_english_words_in_hindi_letters_get_their_spelling_back(self):
        self.assertEqual(R.text("कम ऑन गाइस आई लव यू"), "come on guys I love you")
        self.assertEqual(R.text("गाड़ी आई थी, कम पैसे"), "gaadi aayi thi, kam paise")

    def test_rules_for_words_not_in_the_list(self):
        for word, roman in (("समझना", "samajhna"), ("सड़कें", "sadken"), ("अचानक", "achanak"), ("बताइए", "bataiye"), ("नमकीन", "namkeen")):
            self.assertEqual(R.text(word), roman, word)

    def test_urdu(self):
        self.assertEqual(R.text("مجھے نہیں پتا کہ وہ کہاں گیا ہے؟"), "mujhe nahi pata ke woh kahan gaya hai?")
        self.assertFalse(R.needed(R.text("ٹماٹر خریدنا ضروری تھا")))

    def test_nothing_else_changes(self):
        for line in ("Oh my God, 100%!", "", "line one\nline two", "🔥 W"):
            self.assertEqual(R.text(line), line)
        self.assertEqual(R.text("१२ लाख\nदे"), "12 lakh\nde")


class AskingTheAI(unittest.TestCase):
    def setUp(self):
        fresh_cache()

    def test_only_unknown_words_are_asked_once_and_remembered(self):
        asked = []

        def ai(prompt):
            words = json.loads(prompt.split("Words: ", 1)[1])
            asked.extend(words.values())
            return json.dumps({"words": {k: "SPELLED" + k for k in words}})

        self.assertEqual(R.prepare(["मुझे हेलीकॉप्टर रूपांतरित चाहिए", "hello"], ai), 1)
        self.assertEqual(asked, ["रूपांतरित"])                  # the others are in the built-in list
        self.assertEqual(R.text("रूपांतरित"), "SPELLED1")
        self.assertEqual(R.prepare(["रूपांतरित"], ai), 0)
        self.assertEqual(len(asked), 1)
        R._memory.clear()                                       # another run of the app: read back from disk
        self.assertEqual(R.text("रूपांतरित"), "SPELLED1")

    def test_a_bad_or_failing_answer_falls_back_to_the_rules(self):
        for answer in (lambda p: json.dumps({"words": {"1": "रूपांतरित"}}), lambda p: "not json", mock.Mock(side_effect=RuntimeError("busy"))):
            fresh_cache()
            self.assertEqual(R.prepare(["रूपांतरित"], answer), 0)
            self.assertEqual(R.text("रूपांतरित"), "rupantrit")


class SubtitlesAndTitles(unittest.TestCase):
    SEGMENTS = [{"start": 0.0, "end": 3.0, "text": "[हंसी] क्या बात है",
                 "words": [{"start": 0.0, "end": 0.5, "text": "[हंसी]"}, {"start": 0.5, "end": 1.0, "text": "क्या"},
                           {"start": 1.0, "end": 1.5, "text": "बात"}, {"start": 1.5, "end": 2.0, "text": "है।"}]}]

    def setUp(self):
        fresh_cache()

    def test_timing_is_kept_and_sound_notes_still_work(self):
        out = R.segments(self.SEGMENTS)
        self.assertEqual([(w["start"], w["end"]) for w in out[0]["words"]], [(w["start"], w["end"]) for w in self.SEGMENTS[0]["words"]])
        self.assertEqual([w["text"] for w in out[0]["words"]], ["[हंसी]", "kya", "baat", "hai"])
        self.assertEqual(self.SEGMENTS[0]["words"][1]["text"], "क्या")           # the saved transcript is not touched
        spoken, reactions = C.split_annotations(C._raw_words(out[0]))
        self.assertEqual([w[2] for w in spoken], ["kya", "baat", "hai"])
        self.assertEqual([kind for _, kind in reactions], ["laugh"])
        ass = C.build_ass(out, 0.0, 3.0, 1080, 1920, style="pop")
        self.assertIn("BAAT", ass)
        self.assertFalse(R.needed(ass))

    def test_segments_without_word_timing_and_ranges(self):
        items = [{"start": 0.0, "end": 2.0, "text": "क्या बात"}, {"start": 50.0, "end": 52.0, "text": "क्या बात"}]
        out = R.segments(items, [(0.0, 10.0)])
        self.assertEqual([s["text"] for s in out], ["kya baat", "क्या बात"])

    def test_edit_notes(self):
        got = R.hints({"hook": "क्या बात है", "popups": [{"at": 3.0, "emoji": "😂", "text": "बहुत अच्छा"}], "keywords": ["पैसे"]})
        self.assertEqual(got["hook"], "kya baat hai")
        self.assertEqual(got["popups"], [{"at": 3.0, "emoji": "😂", "text": "bahut accha"}])
        self.assertEqual(got["keywords"], ["paise"])

    def test_keyword_emoji_match_roman_hindi_without_false_alarms(self):
        def emoji(word):
            return next((e for pattern, e in E.KEYWORD_EMOJI if re.search(pattern, word)), None)

        for word, wanted in (("paise", "💰"), ("jahaaz", "✈️"), ("dimaag", "🧠"), ("mazaak", "😂"), ("kamaal", "😮"), ("dar", "😱")):
            self.assertEqual(emoji(word), wanted, word)
        for word in ("kya", "sadak", "darwaza"):
            self.assertIsNone(emoji(word), word)

    def test_upload_text_follows_the_setting_but_never_changes_what_the_person_typed(self):
        project = {"candidates": [{"title": "नाकाबंदी और तबाही", "kit": {"title": "क्या बात है", "description": "बहुत अच्छा", "hashtags": []}}]}
        meta = uploads._metadata_for(project, {"candidate": 0, "title": "x"}, {"privacy": "private"})["snippet"]
        self.assertEqual(meta["title"], "kya baat hai")
        self.assertTrue(meta["description"].startswith("bahut accha"))
        typed = uploads._metadata_for(project, {"candidate": 0}, {"override": {"title": "मेरा title"}})["snippet"]
        self.assertEqual(typed["title"], "मेरा title")
        with mock.patch.dict(os.environ, {"LOCAL_CAPTION_SCRIPT": "original"}):
            meta = uploads._metadata_for(project, {"candidate": 0}, {})["snippet"]
        self.assertEqual(meta["title"], "क्या बात है")


class ThroughThePipeline(unittest.TestCase):
    """What render_local hands to the renderer (the renderer itself is tested elsewhere)."""

    def setUp(self):
        fresh_cache()
        self.tmp = Path(tempfile.mkdtemp(prefix="pipe_roman_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        src = self.tmp / "talk.mp4"
        src.write_bytes(b"not really a video: the renderer is a stand-in")
        pid = "roman" + str(time.time_ns())
        meta = {"kind": "file", "id": pid, "url": str(src), "title": "Local", "duration": 10.0, "embeddable": False, "heatmap": None,
                "local_path": str(src)}
        tr = {"duration": 10.0, "language": "hi", "source": "test", "segments": SubtitlesAndTitles.SEGMENTS}
        self.p = P.new_project(meta, tr, [{"title": "क्या बात", "start_time": 0.0, "end_time": 10.0, "score": 90}], {})
        P.save_project(self.p)
        P.cache_dir(pid).mkdir(parents=True, exist_ok=True)
        (P.cache_dir(pid) / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
        self.addCleanup(shutil.rmtree, P.project_dir(pid), True)

    def render(self, edit):
        seen, prompts = {}, []

        def fake_render(media, out, start, end, options, **kw):
            seen.update(kw)
            return out

        def ai(prompt, use_cache=True):
            prompts.append(prompt)
            if "Words: " in prompt:
                return "{}"
            return json.dumps({"clips": [{"id": 0, "title": "वाह", "hook": "क्या सीन है", "popups": [], "keywords": ["बात"]}]})

        with mock.patch("shorts_generator.local.clipper.render_clip", fake_render), \
                mock.patch("shorts_generator.local.llm.call_local_llm", ai):
            records = pipeline.render_local(self.p, [0], K.RenderOptions(edit=edit, captions=True))
        self.assertIsNone(records[0]["error"], records[0])
        return seen, records[0], prompts

    def test_everything_on_screen_is_roman_and_the_saved_data_is_not_rewritten(self):
        seen, record, prompts = self.render("viral")
        self.assertEqual([w["text"] for w in seen["transcript_segments"][0]["words"]], ["[हंसी]", "kya", "baat", "hai"])
        self.assertEqual((seen["title"], record["title"]), ("kya baat", "kya baat"))
        self.assertEqual((seen["hints"]["hook"], seen["hints"]["keywords"]), ("kya scene hai", ["baat"]))
        self.assertIn("never Devanagari", next(p for p in prompts if "Words: " not in p))
        self.assertEqual(P.load_project(self.p["id"])["candidates"][0]["title"], "क्या बात")

    def test_original_script_when_chosen(self):
        with mock.patch.dict(os.environ, {"LOCAL_CAPTION_SCRIPT": "original"}):
            seen, record, prompts = self.render("off")
        self.assertEqual(seen["transcript_segments"][0]["words"][1]["text"], "क्या")
        self.assertEqual(record["title"], "क्या बात")
        self.assertEqual(prompts, [])


class TheSetting(unittest.TestCase):
    def test_saved_from_the_page(self):
        env_file = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "roman.env"
        client, headers = webapp.app.test_client(), {"X-Studio": "1"}
        with mock.patch.object(config, "ENV_FILE", env_file), mock.patch.dict(os.environ, {"LOCAL_CAPTION_SCRIPT": ""}):
            self.assertEqual(client.get("/api/config").get_json()["caption_script"], "roman")
            r = client.post("/api/settings", json={"caption_script": "original"}, headers=headers)
            self.assertEqual((r.status_code, r.get_json()["caption_script"]), (200, "original"))
            self.assertIn("LOCAL_CAPTION_SCRIPT=original", env_file.read_text())
            self.assertEqual(client.post("/api/settings", json={"caption_script": "x"}, headers=headers).status_code, 400)

    def test_roman_unless_asked_otherwise(self):
        for value, wanted in (("", "roman"), ("roman", "roman"), ("original", "original"), ("junk", "roman")):
            with mock.patch.dict(os.environ, {"LOCAL_CAPTION_SCRIPT": value}):
                self.assertEqual(config.caption_script(), wanted, value)
        with self.assertRaises(ValueError):
            config.save_settings({"LOCAL_CAPTION_SCRIPT": "klingon"})


if __name__ == "__main__":
    unittest.main()

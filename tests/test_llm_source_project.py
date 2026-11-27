import json
import os
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

from shorts_generator import config
from shorts_generator.local import llm, project as P, source as S, storage


def _clock():
    """A fake time.time() that jumps 30 s per call, so models parked for 20 s are free again in the next round."""
    t = [1000.0]

    def tick():
        t[0] += 30.0
        return t[0]
    return tick


def fake_response(status, body):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    return r


class LlmErrors(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(llm._classify(fake_response(503, {"error": {"message": "overloaded"}})).kind, "overloaded")
        q = llm._classify(fake_response(429, {"error": {"message": "Please retry in 9h39m34.5s"}}))
        self.assertEqual(q.kind, "quota")
        self.assertAlmostEqual(q.retry_after, 9 * 3600 + 39 * 60 + 34.5)
        self.assertEqual(llm._classify(fake_response(404, {"error": {"message": "no model"}})).kind, "notfound")
        self.assertEqual(llm._classify(fake_response(400, {"error": {"message": "API key not valid"}})).kind, "auth")
        self.assertEqual(llm._classify(fake_response(403, {"error": {"message": "x"}})).kind, "auth")


class GeminiFallback(unittest.TestCase):
    def setUp(self):
        llm._parked.clear()
        llm._last_model.clear()
        self.env = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k" * 12, "GEMINI_MODEL": "auto", "LLM_PROVIDER": "gemini"})
        self.env.start()
        self.chain = mock.patch.object(llm, "gemini_models", return_value=["m-new", "m-mid", "m-old"])
        self.chain.start()
        self.sleep = mock.patch.object(llm, "_sleep", lambda s: None)
        self.sleep.start()

    def tearDown(self):
        for p in (self.env, self.chain, self.sleep):
            p.stop()

    def test_falls_through_quota_and_overload_to_a_working_model(self):
        calls = []

        def gen(model, prompt, key):
            calls.append(model)
            if model == "m-new":
                raise llm.LLMError("quota", "quota", 429, 3600)
            if model == "m-mid":
                raise llm.LLMError("busy", "overloaded", 503)
            return '{"ok": true}'

        with mock.patch.object(llm, "_gemini_generate", gen):
            self.assertEqual(llm.call_gemini_llm("p"), '{"ok": true}')
        self.assertEqual(calls, ["m-new", "m-mid", "m-mid", "m-mid", "m-old"])         # 3 tries on the busy one
        self.assertGreater(llm._parked["m-new"], time.time() + 3000)                  # quota: parked for an hour
        self.assertGreater(llm._parked["m-mid"], time.time())                         # overloaded: parked briefly

    def test_next_call_starts_with_the_model_that_worked(self):
        llm._last_model["gemini"] = "m-old"
        calls = []
        with mock.patch.object(llm, "_gemini_generate", lambda m, p, k: calls.append(m) or "{}"):
            llm.call_gemini_llm("p")
        self.assertEqual(calls, ["m-old"])

    def test_bad_key_stops_at_once(self):
        def gen(model, prompt, key):
            raise llm.LLMError("bad key", "auth", 403)

        with mock.patch.object(llm, "_gemini_generate", gen):
            with self.assertRaises(llm.LLMError) as cm:
                llm.call_gemini_llm("p")
        self.assertEqual(cm.exception.kind, "auth")

    def test_busy_everywhere_waits_and_tries_the_chain_again(self):
        attempts = {"n": 0}

        def gen(model, prompt, key):
            attempts["n"] += 1
            if attempts["n"] <= 9:                 # round 1 and 2: three models x three tries, all busy
                raise llm.LLMError("busy", "overloaded", 503)
            return '{"ok": true}'

        with mock.patch.object(llm, "_gemini_generate", gen), mock.patch.object(llm.time, "time", side_effect=_clock()):
            self.assertEqual(llm.call_gemini_llm("p"), '{"ok": true}')
        self.assertGreater(attempts["n"], 9)

    def test_everything_down_gives_a_summary(self):
        def gen(model, prompt, key):
            raise llm.LLMError("quota", "quota", 429, 3600)

        with mock.patch.object(llm, "_gemini_generate", gen):
            with self.assertRaises(llm.LLMError) as cm:
                llm.call_gemini_llm("p")
        self.assertIn("Every Gemini model failed", str(cm.exception))
        self.assertIn("quota", str(cm.exception).lower())


class LlmCache(unittest.TestCase):
    def test_valid_answers_are_cached_invalid_ones_are_not(self):
        n = {"c": 0}
        answers = ['{"a": 1}', "garbage", '{"b": 2}']

        def fake(prompt):
            n["c"] += 1
            return answers[min(n["c"] - 1, 2)]

        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k" * 12}), \
                mock.patch.object(llm, "call_gemini_llm", fake):
            self.assertEqual(llm.call_local_llm("cache test A"), '{"a": 1}')
            self.assertEqual(llm.call_local_llm("cache test A"), '{"a": 1}')
            self.assertEqual(n["c"], 1)                                   # second answer came from disk
            llm.call_local_llm("cache test B")
            llm.call_local_llm("cache test B")                            # garbage was not cached: asked again
            self.assertEqual(n["c"], 3)
            llm.call_local_llm("cache test A", use_cache=False)
            self.assertEqual(n["c"], 4)


class Source(unittest.TestCase):
    INFO = {
        "language": "hi",
        "subtitles": {"live_chat": [{"ext": "json3", "url": "x"}], "en": [{"ext": "vtt", "url": "manual-en"}]},
        "automatic_captions": {
            "hi-orig": [{"ext": "vtt", "url": "hi-vtt"}, {"ext": "json3", "url": "hi-json"}],
            "hi": [{"ext": "json3", "url": "hi-translated"}], "en": [{"ext": "json3", "url": "en-translated"}],
        },
    }

    def src(self, info=None):
        return S.Source(kind="youtube", id="abc", url="u", title="t", duration=100.0, _info=info or self.INFO)

    def test_default_picks_the_original_language_track_and_prefers_json3(self):
        t = self.src().caption_track()
        self.assertEqual((t["kind"], t["key"], t["ext"], t["url"]), ("auto", "hi-orig", "json3", "hi-json"))

    def test_manual_subtitles_win_when_the_language_is_asked_for(self):
        self.assertEqual(self.src().caption_track("en")["kind"], "manual")

    def test_translated_track_is_the_last_resort_for_a_requested_language(self):
        info = {**self.INFO, "subtitles": {}}
        self.assertEqual(self.src(info).caption_track("en")["url"], "en-translated")

    def test_no_captions(self):
        self.assertIsNone(self.src({"language": "hi"}).caption_track())

    def test_friendly_errors(self):
        self.assertIn("bot check", S.friendly_error(Exception("ERROR: Sign in to confirm you’re not a bot")))
        self.assertIn("private", S.friendly_error(Exception("Private video. Sign in")))
        self.assertIn("rate-limiting", S.friendly_error(Exception("HTTP Error 429: Too Many Requests")))

    def test_resolve_rejects_nonsense_and_missing_files(self):
        with self.assertRaises(S.SourceError):
            S.resolve("")
        with self.assertRaises(S.SourceError):
            S.resolve("C:/definitely/not/here.mp4")

    def test_local_file_is_used_in_place(self):
        with mock.patch.object(S, "_ffprobe_duration", return_value=12.5):
            f = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "my video.mp4"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x")
            s = S.resolve(str(f))
        self.assertEqual((s.kind, s.duration, s.is_stream), ("file", 12.5, False))
        self.assertTrue(s.local_path.endswith("my video.mp4"))


class WhisperIsOptional(unittest.TestCase):
    def test_missing_whisper_gives_an_actionable_message(self):
        import sys
        from shorts_generator.local import transcriber

        with mock.patch.dict(sys.modules, {"faster_whisper": None}):
            with self.assertRaises(RuntimeError) as cm:
                transcriber.transcribe_local("does-not-matter.mp4")
        self.assertIn("pip install faster-whisper", str(cm.exception))


class Projects(unittest.TestCase):
    def make(self, pid="vid1"):
        meta = {"kind": "youtube", "id": pid, "url": "https://youtu.be/" + pid, "title": "T", "duration": 300.0}
        tr = {"duration": 300.0, "language": "en", "source": "youtube-captions:auto:en", "segments": [{"start": 0, "end": 5, "text": "hi"}]}
        cands = [{"title": "a", "start_time": 10.0, "end_time": 40.0, "score": 90}, {"title": "b", "start_time": 50.0, "end_time": 80.0, "score": 70}]
        return P.new_project(meta, tr, cands, {"focus": ""})

    def test_roundtrip_and_ids(self):
        p = self.make()
        P.save_project(p)
        q = P.load_project("vid1")
        self.assertEqual([c["id"] for c in q["candidates"]], [0, 1])
        self.assertEqual(q["transcript_info"]["segments"], 1)
        self.assertEqual(P.list_projects()[0]["id"], "vid1")

    def test_ids_are_validated(self):
        for bad in ("../x", "a/b", "", "x" * 200):
            with self.assertRaises(KeyError):
                P.load_project(bad)

    def test_manual_candidates_append_without_renumbering(self):
        p = self.make()
        c = P.add_manual_candidate(p, 100, 130, "mine")
        self.assertEqual(c["id"], 2)
        self.assertTrue(c["manual"] and c["score"] is None)

    def test_schema_1_projects_are_upgraded(self):
        old = {"id": "legacy1", "title": "Old", "source_url": "https://youtu.be/legacy1", "source_path": "nope.mp4", "duration": 100,
               "transcript": {"segments": [{"start": 0, "end": 1, "text": "x"}]}, "candidates": [], "clips": []}
        folder = P.project_dir("legacy1")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "project.json").write_text(json.dumps(old), encoding="utf-8")
        q = P.load_project("legacy1")
        self.assertEqual((q["schema"], q["source"]["kind"], q["source"]["local_path"]), (2, "youtube", None))
        self.assertEqual(P.load_transcript(q)["segments"][0]["text"], "x")

    def test_delete(self):
        P.save_project(self.make("todelete"))
        self.assertGreater(P.delete_project("todelete"), 0)
        with self.assertRaises(KeyError):
            P.load_project("todelete")


class ReanalysisKeepsClips(unittest.TestCase):
    def test_rendered_clips_survive_a_new_analysis_of_the_same_video(self):
        from shorts_generator import pipeline

        p = Projects().make("again1")
        p["clips"] = [{"candidate": 1, "file": "short_02_b.mp4", "title": "b"}, {"candidate": 0, "file": None, "error": "boom"}]
        P.save_project(p)
        kept = pipeline._kept_clips("again1")
        self.assertEqual([(c["file"], c["candidate"]) for c in kept], [("short_02_b.mp4", None)])    # failed ones are dropped
        self.assertEqual(pipeline._kept_clips("never-analysed"), [])
        self.assertEqual(len(pipeline._kept_clips("x", [{"file": "a.mp4", "candidate": 3}])), 1)


class Storage(unittest.TestCase):
    def test_usage_and_clean(self):
        base = Path(os.environ["LOCAL_OUTPUT_DIR"])
        base.mkdir(parents=True, exist_ok=True)
        (base / "source_old.mp4").write_bytes(b"x" * 1000)
        (base / "cache" / "v").mkdir(parents=True, exist_ok=True)
        (base / "cache" / "v" / "t.json").write_bytes(b"y" * 10)
        u = storage.usage()
        self.assertEqual(u["legacy_downloads"], 1000)
        freed = storage.clean(["legacy_downloads", "cache"])
        self.assertEqual(freed["legacy_downloads"], 1000)
        self.assertFalse((base / "source_old.mp4").exists())
        self.assertEqual(storage.usage()["legacy_downloads"], 0)


class Settings(unittest.TestCase):
    def test_validation_and_persistence(self):
        env = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "test.env"
        with mock.patch.object(config, "ENV_FILE", env), mock.patch.dict(os.environ, {}, clear=False):
            config.save_settings({"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4o-mini"})
            self.assertEqual(config.llm_provider(), "openai")
            self.assertIn("LLM_PROVIDER=openai", env.read_text())
            for bad in ({"LLM_PROVIDER": "evil"}, {"GEMINI_API_KEY": "short"}, {"GEMINI_API_KEY": "has space in it!!"},
                        {"NOT_ALLOWED": "x"}, {"GEMINI_MODEL": "a b"}):
                with self.assertRaises(ValueError):
                    config.save_settings(bad)
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("OPENAI_MODEL", None)


if __name__ == "__main__":
    unittest.main()

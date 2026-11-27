"""The second round of hardening, found by looking at what really went wrong during real use:

* an 11-hour stream came back with 280 "moments" although 10 were asked for (and one broken AI answer silently degraded the ranking),
* Gemini was overloaded / unreachable for hours, so people pressed "Try again" four times,
* then the YouTube side: credential check, Testing-mode detection, pause switch, review-before-upload, upload history, Autopilot
  never repeating a moment that is already on the channel.
"""
import json
import os
import time
import unittest
from unittest import mock

import app as webapp
from shorts_generator import highlights as H
from shorts_generator import pipeline
from shorts_generator.local import control, llm, project as P, uploads, youtube
from tests.test_youtube import YouTubeCase, make_clips, wait_job

HDR = {"X-Studio": "1"}


def setUpModule():
    webapp.start_worker()


# --------------------------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------------------------- #
def transcript(minutes=90):
    return {"duration": minutes * 60.0, "language": "en", "segments": [
        {"start": i * 10.0, "end": i * 10.0 + 9, "text": f"line {i}"} for i in range(minutes * 6)]}


def cands(n=12):
    return [{"title": f"c{i}", "start_time": 100.0 + i * 120, "end_time": 130.0 + i * 120, "score": 90 - i, "hook_sentence": "", "virality_reason": ""}
            for i in range(n)]


class RankingRobustness(unittest.TestCase):
    def test_a_damaged_ranking_answer_still_yields_its_scores(self):
        raw = ('{"ranking":[{"id":3,"score":88,"reason":"he says "wow" and laughs"},{"id":1,"score":71,"reason":"fine"},'
               '{"score":50,"id":2,"reason":"x"} {"id":99,"score":10}')
        out = H.salvage_ranking(raw, 5)
        self.assertEqual([(r["id"], r["score"]) for r in out], [(3, 88), (1, 71), (2, 50)])      # id 99 does not exist: dropped
        self.assertIn("wow", out[0]["reason"])
        self.assertEqual(H.salvage_ranking("no json here at all", 5), [])
        self.assertEqual(H.salvage_ranking('{"id": 2, "score": 500}', 5)[0]["score"], 100)       # clamped

    def test_rerank_uses_the_salvaged_scores_instead_of_giving_up(self):
        c = cands(12)
        answers = ['{"ranking":[{"id":4,"score":95,"reason":"a "quoted" word"},{"id":2,"score":90,"reason":"b"},{"id":7,"score":85,"reason":"c"}']
        out = H.rerank_highlights(c, transcript(), 3, {}, lambda p: answers[0])
        top = [h["title"] for h in out[:3]]
        self.assertEqual(sorted(top), sorted(["c" + str(i) for i in (4, 2, 7)]))
        self.assertTrue(all("rerank_reason" in h for h in out[:3]))

    def test_an_unreachable_ai_service_is_not_hidden_behind_unreliable_scores(self):
        def down(prompt):
            raise llm.LLMError("busy", kind="overloaded")

        with self.assertRaises(llm.LLMError):
            H.rerank_highlights(cands(12), transcript(), 3, {}, down)

    def test_garbage_that_cannot_be_salvaged_still_falls_back_to_the_chunk_scores(self):
        out = H.rerank_highlights(cands(12), transcript(), 3, {}, lambda p: "I cannot do that")
        self.assertEqual(len(out), 12)
        self.assertEqual(out[0]["title"], "c0")

    def test_moments_to_find_is_honoured_for_a_very_long_video(self):
        many = [{"title": f"m{i}", "start_time": i * 100.0, "end_time": i * 100.0 + 30, "score": 100 - (i % 50), "hook_sentence": "",
                 "virality_reason": ""} for i in range(280)]
        meta = {"kind": "youtube", "id": "capvid", "url": "https://youtu.be/capvid", "title": "Marathon", "duration": 41704.0,
                "embeddable": True, "heatmap": None, "local_path": None}
        opts = pipeline.AnalyzeOptions(candidates=10, use_audio=False, use_replay=False)
        with mock.patch.object(pipeline, "get_highlights", lambda *a, **k: {"highlights": many}), \
                mock.patch("shorts_generator.local.transcript.get_transcript", lambda *a, **k: transcript(10)):
            project = pipeline._build_project(meta, lambda: None, opts)
        self.assertEqual(len(project["candidates"]), 10)
        self.assertEqual([c["id"] for c in project["candidates"]], list(range(10)))
        self.assertTrue(all(c["score"] >= 99 for c in project["candidates"]))                       # the best ones, not the first ones

    def test_the_upper_limit_of_candidates_is_40(self):
        self.assertEqual(pipeline.MAX_CANDIDATES, 40)
        o = pipeline.AnalyzeOptions(candidates=500)
        pipeline._validate_analyze(o)
        self.assertEqual(o.candidates, 40)
        c = webapp.app.test_client()
        seen = []

        def fake(url, opts):
            seen.append(opts.candidates)
            return {"id": "x"}

        with mock.patch.object(webapp.pipeline, "analyze_local", fake):
            for asked in (35, 500):
                r = c.post("/api/analyze", json={"url": f"https://youtu.be/lim{asked}", "candidates": asked}, headers=HDR)
                self.assertEqual(r.status_code, 202)
                wait_job(c, r.get_json()["jobs"][0]["id"])
        self.assertEqual(seen, [35, 40])


# --------------------------------------------------------------------------------------------- #
# Jobs retry themselves when the AI service is busy
# --------------------------------------------------------------------------------------------- #
class SettingsDialog(unittest.TestCase):
    def test_a_rejected_key_does_not_turn_the_model_list_into_a_server_error(self):
        c = webapp.app.test_client()

        def rejected(refresh=False):
            raise llm.LLMError("The API key was rejected (400).", kind="auth")

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "some-key-12345678", "LLM_PROVIDER": "gemini"}),                 mock.patch.object(webapp.llm, "gemini_models", rejected):
            r = c.get("/api/models")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["models"], [])
        self.assertIn("rejected", r.get_json()["error"])


class JobAutoRetry(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()
        p = mock.patch.object(webapp, "RETRY_DELAYS", (0.05, 0.05, 0.05))
        p.start()
        self.addCleanup(p.stop)

    def submit(self, fn):
        with mock.patch.object(webapp.pipeline, "analyze_local", fn):
            r = self.c.post("/api/analyze", json={"url": "https://youtu.be/retry" + str(time.time())}, headers=HDR)
            self.assertEqual(r.status_code, 202)
            return wait_job(self.c, r.get_json()["jobs"][0]["id"], timeout=15)

    def test_a_busy_service_is_retried_by_itself_and_the_job_finishes(self):
        calls = []

        def flaky(url, opts):
            calls.append(1)
            control.report("rank", 0.2, "ranking")
            if len(calls) < 3:
                raise llm.LLMError("Every Gemini model failed: overloaded", kind="overloaded")
            return {"id": "recovered"}

        job = self.submit(flaky)
        self.assertEqual((job["status"], job["result"], len(calls)), ("done", {"project_id": "recovered"}, 3))
        self.assertEqual(job["attempt"], 2)
        self.assertEqual(sum("automatically" in line for line in job["log"]), 2)
        self.assertIsNone(job["error"])

    def test_the_waiting_job_says_when_it_tries_again_and_can_be_cancelled(self):
        with mock.patch.object(webapp, "RETRY_DELAYS", (3600,)):
            with mock.patch.object(webapp.pipeline, "analyze_local", lambda url, opts: (control.report("transcript", None, "x"), (_ for _ in ()).throw(llm.LLMError("net", kind="network")))[1]):
                r = self.c.post("/api/analyze", json={"url": "https://youtu.be/waiting1"}, headers=HDR)
                jid = r.get_json()["jobs"][0]["id"]
                deadline = time.time() + 10
                job = {}
                while time.time() < deadline:
                    job = self.c.get(f"/api/jobs/{jid}").get_json()
                    if job["attempt"] == 1:
                        break
                    time.sleep(0.05)
        self.assertEqual((job["status"], job["attempt"], job["max_attempts"]), ("queued", 1, 2))
        self.assertGreater(job["retry_at"], time.time() + 3000)
        self.assertIn("Trying again automatically in 60 min", job["note"])
        done = self.c.post(f"/api/jobs/{jid}/cancel", headers=HDR).get_json()
        self.assertEqual(done["status"], "cancelled")

    def test_it_gives_up_after_the_last_attempt_and_shows_the_error(self):
        calls = []

        def always(url, opts):
            calls.append(1)
            raise llm.LLMError("still busy", kind="overloaded")

        job = self.submit(always)
        self.assertEqual((job["status"], len(calls)), ("error", 4))                  # 1 try + 3 retries
        self.assertIn("still busy", job["error"])

    def test_wrong_keys_and_quota_are_not_retried(self):
        for kind in ("auth", "quota", "bad", "notfound"):
            calls = []

            def fail(url, opts, kind=kind):
                calls.append(1)
                raise llm.LLMError("no", kind=kind)

            job = self.submit(fail)
            self.assertEqual((job["status"], len(calls)), ("error", 1), kind)

    def test_after_rendering_started_a_failure_is_shown_not_retried(self):
        calls = []

        def late(url, opts):
            calls.append(1)
            control.report("render", 0.5, "clip 2 of 5")                              # a retry would re-render for minutes
            raise llm.LLMError("busy", kind="overloaded")

        job = self.submit(late)
        self.assertEqual((job["status"], len(calls)), ("error", 1))

    def test_other_errors_are_not_retried(self):
        calls = []

        def bad(url, opts):
            calls.append(1)
            raise webapp.SourceError("This video is private.")

        job = self.submit(bad)
        self.assertEqual((job["status"], len(calls), job["error"]), ("error", 1, "This video is private."))


# --------------------------------------------------------------------------------------------- #
# YouTube: the easier, safer connection and the review step
# --------------------------------------------------------------------------------------------- #
class Connection(YouTubeCase):
    def test_the_client_check_tells_good_from_bad_credentials(self):
        self.assertEqual(youtube.check_client(), {"ok": True})
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_SECRET": "GOCSPX-a-wrong-secret-0000"}):
            r = youtube.check_client()
        self.assertFalse(r["ok"])
        self.assertIn("does not recognise", r["error"])
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": "", "YOUTUBE_CLIENT_SECRET": ""}):
            self.assertFalse(youtube.check_client()["ok"])
        with mock.patch.object(youtube.requests, "post", side_effect=youtube.requests.ConnectionError("down")):
            r = youtube.check_client()
        self.assertTrue(r["network"] and not r["ok"])

    def test_the_client_check_route(self):
        c = webapp.app.test_client()
        self.assertEqual(c.post("/api/youtube/check", headers=HDR).get_json(), {"ok": True})
        self.assertEqual(c.post("/api/youtube/check").status_code, 403)

    def sign_in_through_google(self):
        url = youtube.auth_url("http://127.0.0.1:8765/")
        from urllib.parse import parse_qs, urlparse
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        self.fake.codes["c1"] = q["code_challenge"]
        return youtube.finish_auth(q["state"], "c1")

    def test_a_consent_screen_still_in_testing_is_detected_at_sign_in(self):
        self.fake.testing_mode = True
        self.sign_in_through_google()
        st = youtube.status()
        self.assertTrue(st["connected"] and st["testing_mode"])
        self.assertAlmostEqual(st["token_days_left"], 7.0, delta=0.1)

    def test_a_published_app_has_no_expiry_to_warn_about(self):
        self.sign_in_through_google()
        st = youtube.status()
        self.assertFalse(st["testing_mode"])
        self.assertIsNone(st["token_days_left"])

    def test_after_seven_days_in_testing_the_sign_in_is_reported_expired_before_google_says_so(self):
        self.fake.testing_mode = True
        self.sign_in_through_google()
        with mock.patch.object(youtube, "_now", lambda: time.time() + 8 * 86400):
            st = youtube.status()
            self.assertFalse(st["connected"])
            self.assertTrue(st["needs_reconnect"])
            self.assertIn("every 7 days", st["reconnect_reason"])
            with self.assertRaises(youtube.ReconnectNeeded):
                youtube.require_ready()

    def test_reconnecting_to_a_published_app_clears_the_old_warning(self):
        self.fake.testing_mode = True
        self.sign_in_through_google()
        self.fake.testing_mode = False
        self.sign_in_through_google()
        self.assertFalse(youtube.status()["testing_mode"])

    def test_access_denied_explains_test_users(self):
        url = youtube.auth_url("http://127.0.0.1:8765/")
        from urllib.parse import parse_qs, urlparse
        state = parse_qs(urlparse(url).query)["state"][0]
        with self.assertRaisesRegex(youtube.YouTubeError, "Test users"):
            youtube.finish_auth(state, None, "access_denied")


class PauseSwitch(YouTubeCase):
    def setUp(self):
        super().setUp()
        self.sign_in()
        self.c = webapp.app.test_client()

    def test_while_paused_nothing_is_uploaded_and_everything_stays_queued(self):
        p = make_clips(self.pid, n=2)
        uploads.enqueue(p, [c["file"] for c in p["clips"]])
        self.assertTrue(self.c.post("/api/youtube/pause", json={"paused": True}, headers=HDR).get_json()["paused"])
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["waiting"], len(self.fake.inits)), (0, 2, 0))
        self.assertIn("paused", res["reason"])
        self.assertEqual(webapp._upload_tick(), 0)                                      # the watcher respects it too
        self.assertFalse(self.c.post("/api/youtube/pause", json={"paused": False}, headers=HDR).get_json()["paused"])
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 2)

    def test_pause_survives_a_disconnect(self):
        youtube.set_paused(True)
        youtube.disconnect()
        self.assertTrue(youtube.status()["paused"])

class ReviewBeforeUpload(YouTubeCase):
    def setUp(self):
        super().setUp()
        self.sign_in()
        self.c = webapp.app.test_client()

    def test_clean_override_trims_to_what_youtube_accepts(self):
        o = uploads.clean_override({"title": "  Great   moment " + "x" * 200, "description": "d" * 9000, "hashtags": "#a, b #c,,"})
        self.assertEqual(len(o["title"]), 100)
        self.assertTrue(o["title"].startswith("Great moment"))
        self.assertEqual(len(o["description"]), 4000)
        self.assertEqual(o["hashtags"], ["#a", "#b", "#c"])
        self.assertEqual(uploads.clean_override({"title": "", "hashtags": []}), {"hashtags": []})
        self.assertEqual(uploads.clean_override("nope"), {})

    def test_what_the_person_typed_is_what_goes_on_youtube(self):
        p = make_clips(self.pid, n=2)
        f = [c["file"] for c in p["clips"]]
        uploads.enqueue(p, f, overrides={f[0]: {"title": "My own title", "description": "My text", "hashtags": ["#mine", "#mine2"]}})
        uploads.drain(self.pid)
        first, second = self.fake.inits[0]["meta"]["snippet"], self.fake.inits[1]["meta"]["snippet"]
        self.assertEqual(first["title"], "My own title")
        self.assertTrue(first["description"].startswith("My text"))
        self.assertIn("#mine #mine2", first["description"])
        self.assertEqual(second["title"], "Kit title 1")                              # the other clip keeps the AI's text

    def test_an_empty_description_is_respected(self):
        p = make_clips(self.pid, n=1)
        f = p["clips"][0]["file"]
        uploads.enqueue(p, [f], overrides={f: {"description": ""}})
        uploads.drain(self.pid)
        self.assertNotIn("About clip", self.fake.inits[0]["meta"]["snippet"]["description"])

    def test_the_upload_route_takes_the_review_text(self):
        p = make_clips(self.pid, n=1)
        f = p["clips"][0]["file"]
        r = self.c.post(f"/api/projects/{self.pid}/upload", json={"files": [f], "meta": {f: {"title": "Typed title"}}}, headers=HDR)
        self.assertEqual(r.status_code, 202)
        self.assertEqual(wait_job(self.c, r.get_json()["job"]["id"])["status"], "done")
        self.assertEqual(self.fake.inits[0]["meta"]["snippet"]["title"], "Typed title")
        for bad in ("text", ["a"], {f: "x"}):
            self.assertEqual(self.c.post(f"/api/projects/{self.pid}/upload", json={"files": [f], "meta": bad, "force": True}, headers=HDR).status_code, 400, bad)


class History(YouTubeCase):
    def setUp(self):
        super().setUp()
        self.sign_in()
        self.c = webapp.app.test_client()

    def test_the_uploads_page_lists_everything_newest_first(self):
        a = make_clips(self.pid, n=2)
        uploads.enqueue(a, [c["file"] for c in a["clips"]])
        uploads.drain(self.pid)
        pid2 = self.pid + "b"
        b = make_clips(pid2, n=1)
        uploads.enqueue(b, [b["clips"][0]["file"]])
        d = self.c.get("/api/uploads").get_json()
        states = [(r["project_id"], r["state"]) for r in d["items"]]
        self.assertEqual(states.count((self.pid, "uploaded")), 2)
        self.assertEqual(states.count((pid2, "queued")), 1)
        self.assertEqual(d["items"][0]["state"], "uploaded")                           # uploaded ones first (newest), queued ones after
        self.assertTrue(all(r["project_title"] == "Upload test" for r in d["items"]))
        self.assertEqual(d["status"]["uploads_today"], 2)
        self.assertNotIn(self.fake.client_secret, json.dumps(d))

    def test_stop_everything_clears_every_waiting_clip_but_not_the_uploaded_ones(self):
        a = make_clips(self.pid, n=2)
        uploads.enqueue(a, [a["clips"][0]["file"]])
        uploads.drain(self.pid)
        uploads.enqueue(a, [a["clips"][1]["file"]])
        pid2 = self.pid + "c"
        b = make_clips(pid2, n=2)
        uploads.enqueue(b, [c["file"] for c in b["clips"]])
        self.assertEqual(self.c.post("/api/youtube/queue/clear", headers=HDR).get_json(), {"removed": 3})
        self.assertEqual(uploads.counts(self.pid), {"uploaded": 1})
        self.assertEqual(uploads.counts(pid2), {})


class AutopilotDoesNotRepeat(YouTubeCase):
    """A second Autopilot run on the same video must move on to NEW moments, not upload the same ones again."""

    def setUp(self):
        super().setUp()
        self.sign_in()
        self.c = webapp.app.test_client()

        def fake_analyze(url, opts):
            old = []
            try:
                old = [{**c, "candidate": None} for c in P.load_project(self.pid).get("clips", [])]      # what _kept_clips does
            except KeyError:
                pass
            p = make_clips(self.pid, n=6, kit=False, rendered=False)
            p["clips"] = old
            P.save_project(p)
            return p

        def fake_render(project, ids, options, quality=1080, overrides=None, kit=False):
            folder, records = P.project_dir(project["id"]), []
            for i in ids:
                name = f"short_{i + 1:02d}_m{i}.mp4"
                (folder / name).write_bytes(os.urandom(20_000))
                records.append({"candidate": i, "file": name, "title": f"moment {i}", "start": 10.0 + i * 60, "end": 40.0 + i * 60,
                                "aspect_ratio": "9:16", "layout": "auto", "created": "now"})

            def put(fresh):
                for i in ids:
                    fresh["candidates"][i]["kit"] = {"title": f"T{i}", "description": "d", "hashtags": ["#a"]}
                fresh["clips"] = [c for c in fresh.get("clips", []) if c.get("file") not in {r["file"] for r in records}] + records
            P.update_project(project, put)
            return records

        for patcher in (mock.patch.object(webapp.pipeline, "analyze_local", fake_analyze), mock.patch.object(webapp.pipeline, "render_local", fake_render)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_ap(self, count):
        r = self.c.post("/api/analyze", json={"url": "https://youtu.be/same1", "auto_render": count, "upload": {"privacy": "private"}}, headers=HDR)
        self.assertEqual(r.status_code, 202, r.get_json())
        return wait_job(self.c, r.get_json()["jobs"][0]["id"])

    def test_the_second_run_picks_other_moments_and_the_third_has_nothing_left(self):
        j1 = self.run_ap(3)
        self.assertEqual((j1["status"], j1["result"]["uploaded"]), ("done", 3))
        j2 = self.run_ap(3)
        self.assertEqual((j2["status"], j2["result"]["uploaded"]), ("done", 3))
        titles = [i["meta"]["snippet"]["title"] for i in self.fake.inits]
        self.assertEqual(len(titles), 6)
        self.assertEqual(len(set(titles)), 6, titles)                                  # six different moments, none twice
        j3 = self.run_ap(3)
        self.assertEqual(j3["status"], "error")
        self.assertIn("already on your channel", j3["error"])
        self.assertEqual(len(self.fake.videos), 6)

    def test_autopilot_while_paused_queues_but_uploads_nothing(self):
        youtube.set_paused(True)
        j = self.run_ap(2)
        self.assertEqual((j["status"], j["result"]["uploaded"], j["result"]["waiting"]), ("done", 0, 2))
        self.assertIn("paused", j["result"]["reason"])
        self.assertEqual(len(self.fake.inits), 0)
        youtube.set_paused(False)
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 2)

    def test_a_manual_re_upload_is_still_allowed(self):
        self.run_ap(2)
        p = P.load_project(self.pid)
        f = [c["file"] for c in p["clips"] if c.get("file")][0]
        r = self.c.post(f"/api/projects/{self.pid}/upload", json={"files": [f], "force": True}, headers=HDR)
        self.assertEqual(r.status_code, 202)
        self.assertEqual(wait_job(self.c, r.get_json()["job"]["id"])["result"]["uploaded"], 1)
        self.assertEqual(len(self.fake.videos), 3)

    def test_overlap_helper(self):
        r = [(100.0, 130.0)]
        self.assertTrue(uploads.overlaps(105, 135, r))
        self.assertTrue(uploads.overlaps(100, 130, r))
        self.assertFalse(uploads.overlaps(125, 155, r))                                # only 5 of 30 seconds shared
        self.assertFalse(uploads.overlaps(200, 230, r))
        self.assertFalse(uploads.overlaps(100, 130, []))


if __name__ == "__main__":
    unittest.main()

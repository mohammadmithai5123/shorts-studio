import json
import os
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import app as webapp
from shorts_generator import config
from shorts_generator.local import control, project as P

H = {"X-Studio": "1"}


def setUpModule():
    webapp.start_worker()


def make_project(pid="appvid", n=4, duration=600.0):
    meta = {"kind": "youtube", "id": pid, "url": "https://youtu.be/" + pid, "title": "App test", "duration": duration,
            "embeddable": True, "heatmap": [{"start_time": 0, "end_time": 300, "value": 0.5}], "local_path": None}
    tr = {"duration": duration, "language": "en", "source": "youtube-captions:auto:en",
          "segments": [{"start": i * 10.0, "end": i * 10.0 + 9, "text": f"hello number {i}"} for i in range(60)]}
    cands = [{"title": f"moment {i}", "start_time": 20.0 + i * 60, "end_time": 50.0 + i * 60, "score": 90 - i, "signals": {"energy": 80}}
             for i in range(n)]
    p = P.new_project(meta, tr, cands, {"focus": ""})
    P.save_project(p)
    (P.cache_dir(pid)).mkdir(parents=True, exist_ok=True)
    (P.cache_dir(pid) / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
    (P.cache_dir(pid) / "energy.json").write_text(json.dumps({"db": [-20.0 + (i % 7) for i in range(600)]}), encoding="utf-8")
    return p


def wait_for(client, job_id, want=("done", "error", "cancelled"), timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in want:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish: " + json.dumps(job)[:300])


class Guards(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_foreign_host_and_missing_header_are_refused(self):
        self.assertEqual(self.c.get("/api/config", headers={"Host": "evil.example"}).status_code, 403)
        self.assertEqual(self.c.post("/api/settings/test", json={}).status_code, 403)
        self.assertEqual(self.c.get("/api/config").status_code, 200)

    def test_static_whitelist_and_clip_names(self):
        self.assertEqual(self.c.get("/static/app.js").status_code, 200)
        self.assertEqual(self.c.get("/static/index.html").status_code, 404)
        self.assertEqual(self.c.get("/static/../app.py").status_code, 404)
        make_project()
        self.assertEqual(self.c.get("/media/appvid/clip/..%2Fproject.json").status_code, 404)
        self.assertEqual(self.c.get("/media/appvid/clip/project.json").status_code, 404)
        self.assertEqual(self.c.get("/media/appvid/source").status_code, 404)           # streamed: no local source to serve
        self.assertEqual(self.c.get("/api/projects/..%2Fx").status_code, 404)

    def test_security_headers(self):
        r = self.c.get("/api/config")
        self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(r.headers["Cache-Control"], "no-store")


class Settings(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()
        self.env = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "app.env"
        self.p = mock.patch.object(config, "ENV_FILE", self.env)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("LLM_PROVIDER", None)

    def test_secrets_are_write_only(self):
        r = self.c.post("/api/settings", json={"gemini_api_key": "A" * 20, "llm_provider": "gemini"}, headers=H)
        self.assertEqual(r.status_code, 200)
        body = json.dumps(r.get_json()) + json.dumps(self.c.get("/api/settings").get_json()) + json.dumps(self.c.get("/api/config").get_json())
        self.assertNotIn("A" * 20, body)
        self.assertTrue(r.get_json()["gemini_key_set"])
        self.assertIn("GEMINI_API_KEY=" + "A" * 20, self.env.read_text())

    def test_bad_values_are_rejected(self):
        for body in ({"llm_provider": "x"}, {"gemini_api_key": "short"}, {"whisper_model": "../../etc"}):
            self.assertEqual(self.c.post("/api/settings", json=body, headers=H).status_code, 400)


class Projects(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()
        make_project()

    def test_public_project_hides_the_transcript_and_adds_excerpts(self):
        d = self.c.get("/api/projects/appvid").get_json()
        self.assertNotIn("transcript", d)
        self.assertEqual(d["source"]["kind"], "youtube")
        self.assertIn("hello number", d["candidates"][0]["excerpt"])
        self.assertTrue(d["source"]["has_heatmap"])
        self.assertNotIn("heatmap", d["source"])

    def test_signals_endpoint_feeds_the_timeline(self):
        d = self.c.get("/api/projects/appvid/signals").get_json()
        self.assertEqual(len(d["energy"]), 400)
        self.assertEqual(len(d["candidates"]), 4)
        self.assertTrue(d["replay"])

    def test_adding_your_own_clip_validates(self):
        self.assertEqual(self.c.post("/api/projects/appvid/candidates", json={"start": 100, "end": 90}, headers=H).status_code, 400)
        self.assertEqual(self.c.post("/api/projects/appvid/candidates", json={"start": 100, "end": 101}, headers=H).status_code, 400)
        self.assertEqual(self.c.post("/api/projects/appvid/candidates", json={"start": 100, "end": 9999}, headers=H).status_code, 400)
        r = self.c.post("/api/projects/appvid/candidates", json={"start": 100, "end": 140, "title": "mine"}, headers=H)
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.get_json()["id"], 4)
        self.assertEqual(len(self.c.get("/api/projects/appvid").get_json()["candidates"]), 5)

    def test_render_request_validation(self):
        url = "/api/projects/appvid/render"
        for body in ({"ids": []}, {"ids": [99]}, {"ids": [0], "layout": "zoom"}, {"ids": [0], "aspect_ratio": "0:9"},
                     {"ids": [0], "caption_style": "neon"}, {"ids": [0], "overrides": {"1": {"start": 1, "end": 9}}},
                     {"ids": list(range(13))}):
            self.assertEqual(self.c.post(url, json=body, headers=H).status_code, 400, body)

    def test_analyze_request_validation(self):
        for body in ({"url": ""}, {"urls": ["x"] * 11}, {"url": "x", "language": "hindi!!"}, {"url": "x", "min_seconds": 90, "max_seconds": 60}):
            self.assertEqual(self.c.post("/api/analyze", json=body, headers=H).status_code, 400, body)

    def test_delete_project(self):
        make_project("gone")
        self.assertEqual(self.c.delete("/api/projects/gone", headers=H).status_code, 200)
        self.assertEqual(self.c.get("/api/projects/gone").status_code, 404)


class Jobs(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_analyze_job_runs_reports_progress_and_finishes(self):
        def fake(url, opts):
            control.report("transcript", 0.5, "half way")
            control.report("rank", None, "thinking")
            print("[test] line printed during the job")
            return {"id": "fakeproj"}

        with mock.patch.object(webapp.pipeline, "analyze_local", fake):
            r = self.c.post("/api/analyze", json={"urls": ["https://youtu.be/x1", "https://youtu.be/x2"]}, headers=H)
            self.assertEqual(r.status_code, 202)
            jobs = r.get_json()["jobs"]
            self.assertEqual(len(jobs), 2)
            done = wait_for(self.c, jobs[1]["id"])
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"], {"project_id": "fakeproj"})
        self.assertEqual(done["stages"], ["transcript", "rank"])
        self.assertTrue(any("line printed" in line for line in done["log"]))

    def test_a_failing_job_reports_a_readable_error(self):
        def boom(url, opts):
            raise webapp.SourceError("This video is private.")

        with mock.patch.object(webapp.pipeline, "analyze_local", boom):
            job = self.c.post("/api/analyze", json={"url": "https://youtu.be/x3"}, headers=H).get_json()["jobs"][0]
            done = wait_for(self.c, job["id"])
        self.assertEqual((done["status"], done["error"]), ("error", "This video is private."))

    def test_a_running_job_can_be_cancelled(self):
        started = threading.Event()

        def slow(url, opts):
            started.set()
            for _ in range(200):
                control.check()
                time.sleep(0.05)
            return {"id": "never"}

        with mock.patch.object(webapp.pipeline, "analyze_local", slow):
            job = self.c.post("/api/analyze", json={"url": "https://youtu.be/x4"}, headers=H).get_json()["jobs"][0]
            self.assertTrue(started.wait(5))
            self.c.post(f"/api/jobs/{job['id']}/cancel", headers=H)
            done = wait_for(self.c, job["id"])
        self.assertEqual(done["status"], "cancelled")

    def test_render_job_passes_the_options_through(self):
        make_project("renderme")
        seen = {}

        def fake(project, ids, options, quality, overrides, kit):
            seen.update(ids=ids, options=options, quality=quality, overrides=overrides, kit=kit)
            return []

        with mock.patch.object(webapp.pipeline, "render_local", fake):
            body = {"ids": [2, 0], "layout": "split", "captions": True, "caption_style": "karaoke", "title_card": True,
                    "watermark": "@me", "aspect_ratio": "1:1", "quality": 720, "kit": True, "overrides": {"2": {"start": 100, "end": 130}}}
            job = self.c.post("/api/projects/renderme/render", json=body, headers=H).get_json()
            done = wait_for(self.c, job["id"])
        self.assertEqual(done["status"], "done", done)
        self.assertEqual(seen["ids"], [0, 2])
        self.assertEqual((seen["options"].layout, seen["options"].caption_style, seen["options"].aspect_ratio), ("split", "karaoke", "1:1"))
        self.assertEqual((seen["quality"], seen["kit"], seen["overrides"]), (720, True, {2: (100.0, 130.0)}))


if __name__ == "__main__":
    unittest.main()

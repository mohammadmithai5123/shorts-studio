import io
import json
import os
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import app as webapp
from shorts_generator import config, logsetup
from shorts_generator.local import health, project as P

H = {"X-Studio": "1"}


def setUpModule():
    webapp.start_worker()


def drain(timeout=20.0):
    """Wait until no job is queued or running, so one test cannot be disturbed by another's leftovers."""
    end = time.time() + timeout
    while time.time() < end and any(j.status in ("queued", "running") for j in list(webapp.JOBS.values())):
        time.sleep(0.05)


def project_with_clips(pid="opsvid", kit=True):
    meta = {"kind": "youtube", "id": pid, "url": "https://youtu.be/" + pid, "title": "Ops: test/video", "duration": 600.0}
    tr = {"duration": 600.0, "language": "en", "source": "x", "segments": [{"start": 0, "end": 5, "text": "hi"}]}
    cands = [{"title": "first moment", "start_time": 10.0, "end_time": 40.0, "score": 90},
             {"title": "second moment", "start_time": 100.0, "end_time": 130.0, "score": 80}]
    p = P.new_project(meta, tr, cands, {})
    if kit:
        p["candidates"][0]["kit"] = {"title": "A hooky title", "description": "Two short lines.", "hashtags": ["#one", "#two"]}
    folder = P.project_dir(pid)
    folder.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(p["candidates"]):
        name = f"short_{i + 1:02d}_x.mp4"
        (folder / name).write_bytes(b"fake-mp4-bytes-" + bytes([i]))
        p["clips"].append({"candidate": i, "file": name, "title": c["title"], "start": c["start_time"], "end": c["end_time"],
                           "layout": "auto", "captions": False})
    P.save_project(p)
    return p


class Doctor(unittest.TestCase):
    def test_every_check_is_well_formed(self):
        r = health.doctor()
        ids = {c["id"] for c in r["checks"]}
        self.assertTrue({"python", "ffmpeg", "opencv", "ytdlp", "llm_key", "disk", "writable", "whisper"} <= ids)
        for c in r["checks"]:
            self.assertIn(c["status"], ("ok", "warn", "fail"))
            self.assertTrue(c["label"])
        self.assertIn("Shorts Studio", health.format_report(r))

    def test_missing_ffmpeg_fails_and_says_how_to_fix_it(self):
        with mock.patch.object(health.shutil, "which", return_value=None):
            r = health.doctor()
        c = next(c for c in r["checks"] if c["id"] == "ffmpeg")
        self.assertEqual(c["status"], "fail")
        self.assertIn("winget", c["fix"])
        self.assertFalse(r["ok"])

    def test_outdated_ytdlp_is_a_warning(self):
        fake_net = mock.patch.object(health.requests, "get", return_value=mock.Mock(status_code=204))   # deep = network: never for real here
        fake_net.start()
        self.addCleanup(fake_net.stop)
        with mock.patch.object(health, "ytdlp_version", return_value="2024.01.01"):
            c = next(c for c in health.doctor()["checks"] if c["id"] == "ytdlp")
        self.assertEqual(c["status"], "warn")
        with mock.patch.object(health, "ytdlp_version", return_value="2026.10.26"), mock.patch.object(health, "latest_ytdlp", return_value="2026.11.23"):
            c = next(c for c in health.doctor(deep=True)["checks"] if c["id"] == "ytdlp")
        self.assertEqual(c["status"], "warn")
        self.assertIn("newer: 2026.11.23", c["detail"])
        with mock.patch.object(health, "ytdlp_version", return_value="2026.11.23"), mock.patch.object(health, "latest_ytdlp", return_value="2026.11.23"):
            self.assertEqual(next(c for c in health.doctor(deep=True)["checks"] if c["id"] == "ytdlp")["status"], "ok")

    def test_the_suite_never_sees_your_real_keys(self):
        self.assertEqual((config.gemini_key(), config.openai_key(), config.env("YTDLP_PROXY")), ("", "", ""))
        self.assertFalse(health.doctor()["checks"][[c["id"] for c in health.doctor()["checks"]].index("llm_key")]["status"] == "ok")

    def test_version_comparison(self):
        self.assertGreater(health._version_tuple("2026.11.23"), health._version_tuple("2026.10.26"))
        self.assertGreater(health._version_tuple("2026.10.26"), health._version_tuple("2025.12.31"))

    def test_low_disk_space_message_is_actionable(self):
        with mock.patch.object(health, "free_mb", return_value=50):
            with self.assertRaises(health.LowDiskSpace) as cm:
                health.require_free_space(Path("."), 400)
        self.assertIn("50 MB", str(cm.exception))
        self.assertIn("Free some space", str(cm.exception))
        with mock.patch.object(health, "free_mb", return_value=5000):
            health.require_free_space(Path("."), 400)

    def test_diagnostics_bundle_never_contains_a_secret(self):
        secret = "SECRETKEY1234567890"
        logsetup.setup()
        looks_like_a_key = "AIza" + "B" * 35
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": secret, "OPENAI_API_KEY": secret, "YTDLP_PROXY": "http://user:" + secret + "@proxy:8080"}):
            # even a secret that slipped into an exception message must not reach the file or the zip
            logsetup.get("test").error("call failed: https://h/x?key=%s via http://bob:%s@proxy:8080 (%s)", secret, secret, looks_like_a_key)
            data = health.diagnostics_zip(logsetup.log_dir(), [{"id": "j1", "label": "x " + secret + " " + looks_like_a_key}])
        z = zipfile.ZipFile(io.BytesIO(data))
        names = set(z.namelist())
        self.assertTrue({"health.json", "settings.json", "versions.txt", "jobs.json"} <= names)
        for n in names:
            self.assertNotIn(secret.encode(), z.read(n), n)
            self.assertNotIn(looks_like_a_key.encode(), z.read(n), n)
            self.assertNotIn(b"bob:", z.read(n), n)
        self.assertTrue(json.loads(z.read("settings.json"))["proxy_set"])
        self.assertIn(b"call failed", z.read("logs/studio.log"))             # the line itself is kept, only the secrets are hidden

    def test_unpatching_urllib3_is_harmless_when_yt_dlp_was_never_imported(self):
        health._unpatch_urllib3()
        import urllib3.util.url as u3

        self.assertTrue(hasattr(u3._PERCENT_RE, "sub"))


class ProjectSafety(unittest.TestCase):
    def test_a_damaged_project_file_falls_back_to_the_last_good_copy(self):
        p = project_with_clips("safe1")
        p["title"] = "second save"
        P.save_project(p)                                           # now project.json.bak holds the first save
        path = P.project_dir("safe1") / "project.json"
        self.assertTrue(path.with_name("project.json.bak").exists())
        path.write_text('{"half written', encoding="utf-8")
        self.assertEqual(P.load_project("safe1")["id"], "safe1")    # recovered from the backup, no crash

    def test_a_render_does_not_undo_what_the_person_did_while_it_ran(self):
        from shorts_generator import pipeline
        from shorts_generator.local import clipper

        p = project_with_clips("race1")
        src = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "race_src.mp4"
        src.write_bytes(b"not a real video")
        p["source"]["local_path"] = str(src)
        P.save_project(p)
        job_copy = P.load_project("race1")                          # the copy a long render job holds

        def fake_render(media, out, start, end, options, **kw):
            fresh = P.load_project("race1")                         # meanwhile, in the browser: add a clip, delete a clip
            P.add_manual_candidate(fresh, 200, 240, "added meanwhile")
            fresh["clips"] = [c for c in fresh["clips"] if c["file"] != "short_01_x.mp4"]
            P.save_project(fresh)
            Path(out).write_bytes(b"rendered")

        with mock.patch.object(clipper, "render_clip", fake_render):
            pipeline.render_local(job_copy, [1], clipper.RenderOptions())
        final = P.load_project("race1")
        self.assertEqual([c["title"] for c in final["candidates"]][-1], "added meanwhile")
        self.assertEqual(len(final["candidates"]), 3)
        self.assertEqual([c["candidate"] for c in final["clips"]], [1])         # the deleted clip stays deleted, the new one is recorded
        self.assertEqual(len(job_copy["candidates"]), 3)                        # the caller's copy was refreshed too

    def test_concurrent_saves_and_reads_of_one_project_never_fail(self):
        project_with_clips("conc1")
        errors = []

        def writer():
            try:
                for _ in range(25):
                    P.update_project(P.load_project("conc1"), lambda q: q.update(n=q.get("n", 0) + 1))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        def reader():
            try:
                for _ in range(60):
                    P.load_project("conc1")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=writer) for _ in range(3)] + [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(P.load_project("conc1")["n"], 75)               # 3 writers x 25: no update was lost


class JobHistory(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_running_job_becomes_interrupted_after_a_restart(self):
        old = {"id": "restart01", "kind": "analyze", "label": "https://youtu.be/zz", "project_id": None, "status": "running",
               "request": {"path": "/api/analyze", "body": {"urls": ["https://youtu.be/zz"]}}, "stage": "rank", "stages": ["source", "rank"],
               "fraction": 0.4, "note": "x", "log": ["line 1", "line 2"], "result": None, "error": None,
               "created": time.time() - 100, "started": time.time() - 90, "finished": None}
        path = webapp._jobs_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([old]), encoding="utf-8")
        webapp.JOBS.pop("restart01", None)
        webapp._load_jobs()
        job = self.c.get("/api/jobs/restart01").get_json()
        self.assertEqual(job["status"], "interrupted")
        self.assertIn("Try again", job["error"])
        self.assertEqual(job["request"]["path"], "/api/analyze")      # so Try again can re-send exactly this
        self.assertEqual(job["log"], ["line 1", "line 2"])

    def test_finished_jobs_are_saved_to_disk(self):
        with mock.patch.object(webapp.pipeline, "analyze_local", lambda url, opts: {"id": "persisted"}):
            job = self.c.post("/api/analyze", json={"url": "https://youtu.be/persist1"}, headers=H).get_json()["jobs"][0]
            end = time.time() + 10
            while time.time() < end and self.c.get(f"/api/jobs/{job['id']}").get_json()["status"] not in ("done", "error"):
                time.sleep(0.05)
        time.sleep(0.2)
        saved = json.loads(webapp._jobs_file().read_text(encoding="utf-8"))
        mine = next(j for j in saved if j["id"] == job["id"])
        self.assertEqual(mine["status"], "done")
        self.assertEqual(mine["request"]["body"]["urls"], ["https://youtu.be/persist1"])

    def test_concurrent_history_saves_do_not_collide(self):
        errors = []

        def hammer():
            for _ in range(40):
                webapp._save_jobs()

        with mock.patch.object(webapp.log, "exception", lambda *a, **k: errors.append(a)):
            threads = [threading.Thread(target=hammer) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])

    def test_the_same_request_twice_is_one_job(self):
        project_with_clips("dup1")
        gate = threading.Event()

        def slow(project, ids, options, quality, overrides, kit):
            gate.wait(5)
            return []

        with mock.patch.object(webapp.pipeline, "render_local", slow):
            body = {"ids": [0], "layout": "fit"}
            a = self.c.post("/api/projects/dup1/render", json=body, headers=H).get_json()
            b = self.c.post("/api/projects/dup1/render", json=body, headers=H).get_json()
            self.assertEqual(a["id"], b["id"])
            other = self.c.post("/api/projects/dup1/render", json={"ids": [1], "layout": "fit"}, headers=H).get_json()
            self.assertNotEqual(a["id"], other["id"])
            gate.set()
            end = time.time() + 10
            while time.time() < end and self.c.get(f"/api/jobs/{other['id']}").get_json()["status"] in ("queued", "running"):
                time.sleep(0.05)

    def test_too_many_waiting_jobs_are_refused(self):
        drain()
        gate = threading.Event()

        def held(url, opts):
            gate.wait(5)
            return {"id": "x"}

        with mock.patch.object(webapp.pipeline, "analyze_local", held), mock.patch.object(webapp, "MAX_WAITING_JOBS", 1):
            self.assertEqual(self.c.post("/api/analyze", json={"url": "https://youtu.be/qa1"}, headers=H).status_code, 202)
            r = self.c.post("/api/analyze", json={"url": "https://youtu.be/qa2"}, headers=H)
            self.assertEqual(r.status_code, 429)
            self.assertIn("Too many", r.get_json()["error"])
            gate.set()
            time.sleep(0.5)


class WebExtras(unittest.TestCase):
    def setUp(self):
        self.c = webapp.app.test_client()

    def test_page_has_a_content_security_policy_and_cannot_be_framed(self):
        r = self.c.get("/")
        self.assertIn("frame-ancestors 'none'", r.headers["Content-Security-Policy"])
        self.assertIn("https://www.youtube.com", r.headers["Content-Security-Policy"])
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")
        self.assertNotIn("Content-Security-Policy", self.c.get("/api/config").headers)

    def test_export_zip_contains_clips_posts_and_csv(self):
        project_with_clips("export1")
        r = self.c.get("/api/projects/export1/export.zip")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r.headers["Content-Disposition"])
        z = zipfile.ZipFile(io.BytesIO(r.data))
        self.assertEqual(set(z.namelist()), {"short_01_x.mp4", "short_02_x.mp4", "posts.txt", "clips.csv"})
        posts = z.read("posts.txt").decode("utf-8")
        self.assertIn("A hooky title", posts)
        self.assertIn("#one #two", posts)
        self.assertIn("second moment", posts)                          # no kit written: falls back to the clip title
        csv_text = z.read("clips.csv").decode("utf-8-sig")
        self.assertIn("A hooky title", csv_text)
        self.assertTrue(csv_text.startswith("file,title"))

    def test_export_without_clips_is_a_404(self):
        project_with_clips("export2")
        for f in P.project_dir("export2").glob("*.mp4"):
            f.unlink()
        self.assertEqual(self.c.get("/api/projects/export2/export.zip").status_code, 404)

    def test_deleting_a_clip_removes_file_and_record(self):
        project_with_clips("del1")
        self.assertEqual(self.c.delete("/api/projects/del1/clips/short_01_x.mp4", headers=H).status_code, 200)
        self.assertFalse((P.project_dir("del1") / "short_01_x.mp4").exists())
        self.assertEqual([c["file"] for c in P.load_project("del1")["clips"]], ["short_02_x.mp4"])
        self.assertEqual(self.c.delete("/api/projects/del1/clips/..%2Fproject.json", headers=H).status_code, 404)
        self.assertEqual(self.c.delete("/api/projects/del1/clips/project.json", headers=H).status_code, 404)

    def test_health_endpoint(self):
        d = self.c.get("/api/health").get_json()
        self.assertIn("checks", d)
        self.assertTrue(any(c["id"] == "ffmpeg" for c in d["checks"]))

    def test_ytdlp_update_runs_as_a_job(self):
        with mock.patch.object(webapp.health, "update_ytdlp", return_value={"before": "1", "after": "2", "changed": True}), \
                mock.patch.object(webapp.health, "ytdlp_version", return_value="1"):
            job = self.c.post("/api/ytdlp/update", json={}, headers=H).get_json()
            end = time.time() + 10
            while time.time() < end:
                j = self.c.get(f"/api/jobs/{job['id']}").get_json()
                if j["status"] not in ("queued", "running"):
                    break
                time.sleep(0.05)
        self.assertEqual((j["status"], j["result"]["after"]), ("done", "2"))

    def test_diagnostics_download(self):
        r = self.c.get("/api/diagnostics.zip")
        self.assertEqual(r.status_code, 200)
        self.assertIn("health.json", zipfile.ZipFile(io.BytesIO(r.data)).namelist())

    def test_cookies_file_and_proxy_settings(self):
        env = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "ops.env"
        cookies = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        with mock.patch.object(config, "ENV_FILE", env):
            self.assertEqual(self.c.post("/api/settings", json={"ytdlp_cookies_file": "C:/nope/missing.txt"}, headers=H).status_code, 400)
            self.assertEqual(self.c.post("/api/settings", json={"ytdlp_proxy": "ftp://x"}, headers=H).status_code, 400)
            r = self.c.post("/api/settings", json={"ytdlp_cookies_file": str(cookies), "ytdlp_proxy": "http://u:p@proxy:8080"}, headers=H)
            self.assertEqual(r.status_code, 200)
            body = r.get_json()
            self.assertEqual(body["ytdlp_cookies_file"], str(cookies))
            self.assertTrue(body["ytdlp_proxy_set"])
            self.assertNotIn("proxy:8080", json.dumps(body))             # the proxy (it may hold a password) is write-only
            from shorts_generator.local import source as S

            opts = S.ydl_opts()
            self.assertEqual((opts["cookiefile"], opts["proxy"]), (str(cookies), "http://u:p@proxy:8080"))
            self.assertNotIn("cookiesfrombrowser", opts)
            r = self.c.post("/api/settings", json={"ytdlp_cookies_file": "", "ytdlp_proxy": ""}, headers=H)
            self.assertFalse(r.get_json()["ytdlp_proxy_set"])
        for k in ("YTDLP_COOKIES_FILE", "YTDLP_PROXY"):
            os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()

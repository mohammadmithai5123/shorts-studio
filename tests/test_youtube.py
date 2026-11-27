"""YouTube channel connection + uploads, tested against a fake Google (tests/fake_google.py).

Real Google is never contacted, and nothing here uses real credentials. What these tests prove is that OUR side of the
protocols is right: OAuth with PKCE, token refresh, the resumable upload (chunks, resume after errors, expired sessions),
quota handling, the queue (cap, parking, retries, no double uploads), autopilot, and that no secret ever reaches the browser
or a log.
"""
import hashlib
import json
import os
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

import requests

import app as webapp
from shorts_generator import config, logsetup
from shorts_generator.local import control, health, project as P, uploads, youtube
from tests.fake_google import READ_SCOPE, REFRESH_TOKEN, UPLOAD_SCOPE, FakeGoogle

H = {"X-Studio": "1"}
_COUNTER = [0]


def setUpModule():
    webapp.start_worker()


def utc(y, mo, d, h=0, mi=0) -> float:
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def make_clips(pid: str, n: int = 3, size: int = 600_000, kit: bool = True, score0: int = 90, rendered: bool = True) -> dict:
    """A saved project with ``n`` candidates, and (``rendered``) their clips (random bytes, not real video: the fake does not care)."""
    meta = {"kind": "youtube", "id": pid, "url": "https://youtu.be/" + pid, "title": "Upload test", "duration": 900.0,
            "embeddable": True, "heatmap": None, "local_path": None}
    tr = {"duration": 900.0, "language": "en", "source": "youtube-captions:auto:en",
          "segments": [{"start": i * 10.0, "end": i * 10.0 + 9, "text": f"line {i}"} for i in range(90)]}
    cands = [{"title": f"moment {i}", "start_time": 10.0 + i * 60, "end_time": 40.0 + i * 60, "score": score0 - i * 5} for i in range(n)]
    p = P.new_project(meta, tr, cands, {})
    if kit:
        for i, c in enumerate(p["candidates"]):
            c["kit"] = {"title": f"Kit title {i}", "description": f"About clip {i}.", "hashtags": ["#funny", "#gaming"]}
    folder = P.project_dir(pid)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n if rendered else 0):
        name = f"short_{i + 1:02d}_m{i}.mp4"
        (folder / name).write_bytes(os.urandom(size))
        p["clips"].append({"candidate": i, "file": name, "title": f"moment {i}", "start": 10.0 + i * 60, "end": 40.0 + i * 60,
                           "aspect_ratio": "9:16", "layout": "auto", "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    P.save_project(p)
    return p


class YouTubeCase(unittest.TestCase):
    def setUp(self):
        self.fake = FakeGoogle()
        self.addCleanup(self.fake.close)
        for patcher in (
            mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": self.fake.client_id, "YOUTUBE_CLIENT_SECRET": self.fake.client_secret,
                                         "YOUTUBE_DAILY_UPLOADS": "10", "YOUTUBE_PRIVACY": "private", "YOUTUBE_AUTO_RESUME": "1",
                                         "YOUTUBE_CATEGORY": "22", "YOUTUBE_NOTIFY_SUBSCRIBERS": "0"}),
            mock.patch.dict(youtube.URLS, self.fake.urls()),
            mock.patch.object(youtube, "_sleep", lambda seconds: control.check()),      # back-off without waiting
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.wipe()
        self.addCleanup(self.wipe)
        control.unbind()
        _COUNTER[0] += 1
        self.pid = f"ytt{_COUNTER[0]}"

    @staticmethod
    def wipe() -> None:
        """The output folder is shared by the whole test run: no queued clip or token may leak from one test into the next."""
        import shutil

        youtube.state_file().unlink(missing_ok=True)
        for folder in Path(config.LOCAL_OUTPUT_DIR).glob("clips_ytt*"):
            shutil.rmtree(folder, ignore_errors=True)

    def sign_in(self, expires_in: float = 3600) -> None:
        token = self.fake.new_token()
        youtube._mutate(lambda d: d.update(
            refresh_token=REFRESH_TOKEN, access_token=token, expires_at=time.time() + expires_in, scope=f"{UPLOAD_SCOPE} {READ_SCOPE}",
            client_id=self.fake.client_id, channel={"id": "UC_fake", "title": "Test Channel"}))

    def clip_path(self, p: dict, i: int = 0) -> str:
        return str(P.project_dir(p["id"]) / p["clips"][i]["file"])


# --------------------------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------------------------- #
class Metadata(unittest.TestCase):
    def test_title_description_and_tags_follow_the_api_rules(self):
        m = youtube.build_metadata("<b>Wow</b> " + "word " * 40, "Nice <clip>", ["#Funny", "gaming", "#bad tag", "#funny"])
        t = m["snippet"]["title"]
        self.assertLessEqual(len(t), 100)
        self.assertNotIn("<", t)
        self.assertNotIn(">", m["snippet"]["description"])
        self.assertIn("#Funny #gaming", m["snippet"]["description"])
        self.assertTrue(m["snippet"]["description"].rstrip().endswith("#Shorts"))
        self.assertEqual(m["snippet"]["tags"], ["Funny", "gaming", "Shorts"])        # no duplicate, no tag with a space
        self.assertEqual(m["snippet"]["categoryId"], "22")
        self.assertIs(m["status"]["selfDeclaredMadeForKids"], False)
        self.assertEqual(m["status"]["privacyStatus"], "private")
        self.assertNotIn("publishAt", m["status"])

    def test_a_scheduled_video_is_private_until_it_goes_live(self):
        m = youtube.build_metadata("T", privacy="public", publish_at="2030-01-01T10:00:00Z")
        self.assertEqual((m["status"]["privacyStatus"], m["status"]["publishAt"]), ("private", "2030-01-01T10:00:00Z"))

    def test_an_empty_title_gets_a_fallback(self):
        self.assertEqual(youtube.build_metadata("   ")["snippet"]["title"], "Short")


class PacificTime(unittest.TestCase):
    def test_daylight_saving_boundaries(self):
        off = lambda ts: youtube.pacific_offset(ts).total_seconds() / 3600
        self.assertEqual(off(utc(2026, 7, 1, 12)), -7)
        self.assertEqual(off(utc(2026, 12, 1, 12)), -8)
        self.assertEqual((off(utc(2026, 3, 8, 9, 59)), off(utc(2026, 3, 8, 10))), (-8, -7))          # 2nd Sunday of March
        self.assertEqual((off(utc(2026, 11, 1, 8, 59)), off(utc(2026, 11, 1, 9))), (-7, -8))         # 1st Sunday of November

    def test_next_quota_reset_is_midnight_pacific(self):
        self.assertEqual(youtube.next_pacific_midnight(utc(2026, 11, 25, 12)), utc(2026, 11, 26, 8))
        self.assertEqual(youtube.next_pacific_midnight(utc(2026, 7, 1, 12)), utc(2026, 7, 2, 7))
        self.assertEqual(youtube.next_pacific_midnight(utc(2026, 11, 25, 7, 59)), utc(2026, 11, 25, 8))   # 23:59 Pacific, a minute to go
        self.assertEqual(youtube.pacific_day(utc(2026, 11, 25, 7, 59)), "2026-11-24")
        self.assertEqual(youtube.pacific_day(utc(2026, 11, 25, 8, 1)), "2026-11-25")

    def test_the_reset_over_a_clock_change_is_still_midnight(self):
        # the night of 2026-11-01 has 25 hours: midnight Nov 2 is 08:00 UTC (PST), not 07:00
        self.assertEqual(youtube.next_pacific_midnight(utc(2026, 11, 1, 20)), utc(2026, 11, 2, 8))
        self.assertEqual(youtube.next_pacific_midnight(utc(2026, 3, 8, 20)), utc(2026, 3, 9, 7))


# --------------------------------------------------------------------------------------------- #
# Sign-in
# --------------------------------------------------------------------------------------------- #
class SignIn(YouTubeCase):
    def start(self):
        url = youtube.auth_url("http://127.0.0.1:8765/oauth/youtube/callback")
        return url, {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}

    def test_the_auth_url_asks_for_offline_access_with_pkce(self):
        url, q = self.start()
        self.assertTrue(url.startswith(self.fake.urls()["auth"] + "?"))
        self.assertEqual((q["response_type"], q["access_type"], q["code_challenge_method"], q["client_id"]),
                         ("code", "offline", "S256", self.fake.client_id))
        self.assertEqual(set(q["scope"].split()), {UPLOAD_SCOPE, READ_SCOPE})
        self.assertGreaterEqual(len(q["state"]), 20)
        self.assertNotIn(self.fake.client_secret, url)                           # the secret never goes through the browser

    def test_a_full_sign_in_stores_the_token_and_names_the_channel(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        channel = youtube.finish_auth(q["state"], "c1")
        self.assertEqual(channel["title"], "Test Channel")
        st = youtube.status()
        self.assertTrue(st["connected"])
        self.assertEqual(st["channel"]["title"], "Test Channel")
        self.assertNotIn("ya29", json.dumps(st))
        self.assertNotIn(REFRESH_TOKEN, json.dumps(st))
        self.assertNotIn(self.fake.client_secret, json.dumps(st))
        self.assertIn(REFRESH_TOKEN, youtube.state_file().read_text(encoding="utf-8"))

    def test_a_state_can_only_be_used_once_and_must_be_ours(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        with self.assertRaises(youtube.YouTubeError):
            youtube.finish_auth("not-our-state", "c1")                           # a forged callback
        youtube.finish_auth(q["state"], "c1")
        with self.assertRaises(youtube.YouTubeError):
            youtube.finish_auth(q["state"], "c1")                                # a replayed callback

    def test_an_old_pending_sign_in_expires(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        with mock.patch.object(youtube, "_now", lambda: time.time() + 700):
            with self.assertRaisesRegex(youtube.YouTubeError, "not valid any more"):
                youtube.finish_auth(q["state"], "c1")

    def test_pkce_is_enforced_by_the_fake(self):
        _, q = self.start()
        self.fake.codes["c1"] = hashlib.sha256(b"someone else's verifier").hexdigest()      # a challenge that does not match ours
        with self.assertRaises(youtube.YouTubeError):
            youtube.finish_auth(q["state"], "c1")
        self.assertFalse(youtube.status()["connected"])

    def test_declining_or_unticking_the_upload_permission_is_explained(self):
        _, q = self.start()
        with self.assertRaisesRegex(youtube.YouTubeError, "did not give access"):
            youtube.finish_auth(q["state"], None, "access_denied")
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        self.fake.grant_scope = READ_SCOPE                                       # the person unticked the upload box
        with self.assertRaisesRegex(youtube.YouTubeError, "permission to upload"):
            youtube.finish_auth(q["state"], "c1")
        self.assertFalse(youtube.status()["connected"])

    def test_no_refresh_token_is_explained(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        self.fake.issue_refresh = False
        with self.assertRaisesRegex(youtube.YouTubeError, "long-lasting"):
            youtube.finish_auth(q["state"], "c1")

    def test_an_account_without_a_channel_is_refused_and_revoked(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        self.fake.channel = None
        with self.assertRaisesRegex(youtube.YouTubeError, "no YouTube channel"):
            youtube.finish_auth(q["state"], "c1")
        self.assertFalse(youtube.status()["connected"])
        self.assertEqual(self.fake.revoked, [REFRESH_TOKEN])

    def test_wrong_client_secret_is_explained(self):
        _, q = self.start()
        self.fake.codes["c1"] = q["code_challenge"]
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_SECRET": "GOCSPX-the-wrong-one-999"}):
            with self.assertRaisesRegex(youtube.YouTubeError, "client id / secret"):
                youtube.finish_auth(q["state"], "c1")

    def test_disconnect_forgets_the_token_but_keeps_the_day_counter(self):
        self.sign_in()
        youtube.count_upload()
        youtube.disconnect()
        st = youtube.status()
        self.assertFalse(st["connected"])
        self.assertEqual(st["uploads_today"], 1)
        self.assertEqual(self.fake.revoked, [REFRESH_TOKEN])
        self.assertNotIn("refresh_token", json.loads(youtube.state_file().read_text(encoding="utf-8")))

    def test_client_json_formats(self):
        text = json.dumps({"installed": {"client_id": "abc-1.apps.googleusercontent.com", "client_secret": "GOCSPX-xyz-123456"}})
        self.assertEqual(youtube.parse_client_json(text), ("abc-1.apps.googleusercontent.com", "GOCSPX-xyz-123456"))
        self.assertEqual(youtube.parse_client_json(json.dumps({"web": {"client_id": "a1", "client_secret": "b2"}})), ("a1", "b2"))
        for bad in ("not json", "{}", json.dumps({"installed": {"client_id": "only-id"}}), "[1,2]"):
            with self.assertRaises(ValueError):
                youtube.parse_client_json(bad)


class Tokens(YouTubeCase):
    def test_an_expired_access_token_is_refreshed_once_and_reused(self):
        self.sign_in(expires_in=-10)
        t1 = youtube.access_token()
        t2 = youtube.access_token()
        self.assertEqual(t1, t2)
        self.assertIn(t1, self.fake.valid)
        self.assertEqual(self.fake.calls("POST", "/token"), 1)

    def test_a_revoked_sign_in_means_reconnect_and_stops_calling_google(self):
        self.sign_in(expires_in=-10)
        self.fake.invalid_grant = True
        with self.assertRaises(youtube.ReconnectNeeded):
            youtube.access_token()
        st = youtube.status()
        self.assertTrue(st["needs_reconnect"])
        self.assertFalse(st["connected"])
        self.assertIn("expired or revoked", st["reconnect_reason"])
        calls = self.fake.calls("POST", "/token")
        with self.assertRaises(youtube.ReconnectNeeded):
            youtube.access_token()
        self.assertEqual(self.fake.calls("POST", "/token"), calls)                # no hammering on a dead sign-in
        with self.assertRaises(youtube.ReconnectNeeded):
            youtube.require_ready()

    def test_changing_the_google_client_requires_a_new_sign_in(self):
        self.sign_in()
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": "other-client-id.apps.googleusercontent.com"}):
            self.assertFalse(youtube.status()["connected"])
            self.assertTrue(youtube.status()["needs_reconnect"])
            with self.assertRaises(youtube.ReconnectNeeded):
                youtube.access_token()

    def test_not_configured_and_not_connected_are_told_apart(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": "", "YOUTUBE_CLIENT_SECRET": ""}):
            with self.assertRaisesRegex(youtube.NotConnected, "not set up"):
                youtube.require_ready()
        with self.assertRaisesRegex(youtube.NotConnected, "not connected"):
            youtube.require_ready()


# --------------------------------------------------------------------------------------------- #
# The resumable upload
# --------------------------------------------------------------------------------------------- #
class Upload(YouTubeCase):
    CHUNK = 256 * 1024

    def setUp(self):
        super().setUp()
        self.sign_in()
        self.p = make_clips(self.pid, n=1, size=700_000)
        self.path = self.clip_path(self.p)
        self.data = Path(self.path).read_bytes()
        self.meta = youtube.build_metadata("My short", "d", ["#a"])

    def up(self, progress=None):
        return youtube.upload_video(self.path, self.meta, progress, chunk_size=self.CHUNK)

    def test_a_file_goes_up_in_chunks_and_arrives_intact(self):
        seen = []
        res = self.up(lambda sent, size: seen.append((sent, size)))
        self.assertEqual(self.fake.videos[res["id"]]["bytes"], self.data)
        self.assertEqual(seen[0], (self.CHUNK, len(self.data)))
        self.assertEqual(seen[-1], (len(self.data), len(self.data)))
        self.assertEqual(sorted(s for s, _ in seen), [s for s, _ in seen])         # progress only moves forward
        init = self.fake.inits[0]
        self.assertEqual(init["headers"]["x-upload-content-length"], str(len(self.data)))
        self.assertEqual(init["headers"]["x-upload-content-type"], "video/mp4")
        self.assertTrue(init["headers"]["authorization"].startswith("Bearer ya29."))
        self.assertEqual(init["query"]["uploadType"], ["resumable"])
        self.assertEqual(init["query"]["notifySubscribers"], ["false"])
        self.assertEqual(init["meta"]["snippet"]["title"], "My short")

    def test_resumes_from_what_google_kept_after_a_503_and_a_dropped_connection(self):
        def after_first_chunk(sid, n):
            if n == self.CHUNK and not self.fake.fail_puts:
                self.fake.fail_puts += [503, "drop"]

        self.fake.on_put = after_first_chunk
        res = self.up()
        self.assertEqual(self.fake.videos[res["id"]]["bytes"], self.data)
        self.assertEqual(len(self.fake.sessions), 1)                                # one session the whole way
        self.assertEqual(sum(1 for r in self.fake.put_ranges if r.startswith("bytes 0-")), 1, "chunk 1 must not be sent twice")
        self.assertTrue(any(r.startswith("bytes */") for r in self.fake.put_ranges), "it must ask how much Google has")

    def test_a_session_that_expired_is_started_again(self):
        self.fake.fail_puts = [404]
        res = self.up()
        self.assertEqual(self.fake.videos[res["id"]]["bytes"], self.data)
        self.assertEqual(len(self.fake.inits), 2)

    def test_a_401_refreshes_the_token_and_goes_on(self):
        self.fake.expire_tokens = True
        res = self.up()
        self.assertIn(res["id"], self.fake.videos)
        self.assertEqual(self.fake.calls("POST", "/token"), 1)

    def test_daily_quota_parks_until_midnight_pacific(self):
        self.fake.init_errors = [(403, "quotaExceeded", "quota")]
        with self.assertRaises(youtube.QuotaExceeded) as cm:
            self.up()
        self.assertAlmostEqual(cm.exception.until, youtube.next_pacific_midnight(time.time()), delta=5)

    def test_the_channel_upload_limit_is_tried_again_in_three_hours(self):
        self.fake.init_errors = [(403, "uploadLimitExceeded", "limit")]
        with self.assertRaises(youtube.QuotaExceeded) as cm:
            self.up()
        self.assertAlmostEqual(cm.exception.until, time.time() + 3 * 3600, delta=5)

    def test_a_refused_video_is_a_permanent_error_with_googles_reason(self):
        self.fake.init_errors = [(400, "invalidPublishAt", "Invalid scheduled publishing time")]
        with self.assertRaisesRegex(youtube.UploadRejected, "invalidPublishAt"):
            self.up()

    def test_a_missing_channel_is_explained(self):
        self.fake.init_errors = [(401, "youtubeSignupRequired", "no channel")] * 2           # (the first 401 triggers one token refresh)
        with self.assertRaisesRegex(youtube.UploadRejected, "no YouTube channel"):
            self.up()

    def test_insufficient_scope_asks_for_a_new_sign_in(self):
        self.fake.init_errors = [(403, "insufficientPermissions", "Request had insufficient authentication scopes.")]
        with self.assertRaises(youtube.ReconnectNeeded):
            self.up()

    def test_it_gives_up_after_many_failures_without_hanging(self):
        self.fake.fail_puts = [503] * 100
        with self.assertRaises(youtube.NetworkError):
            self.up()

    def test_a_server_that_is_down_is_a_network_error(self):
        with mock.patch.object(youtube.requests, "request", side_effect=requests.ConnectionError("down")):
            with self.assertRaises(youtube.NetworkError):
                self.up()

    def test_a_server_that_never_keeps_the_data_does_not_loop_forever(self):
        self.fake.stall = True
        with self.assertRaisesRegex(youtube.NetworkError, "same part"):
            self.up()
        self.assertLess(len(self.fake.put_ranges), 20)

    def test_an_empty_file_is_refused_before_anything_is_sent(self):
        Path(self.path).write_bytes(b"")
        with self.assertRaises(youtube.UploadRejected):
            self.up()
        self.assertEqual(len(self.fake.inits), 0)

    def test_cancel_stops_between_chunks(self):
        ev = threading.Event()
        control.bind(ev, None)
        self.addCleanup(control.unbind)
        self.fake.on_put = lambda sid, n: ev.set()
        with self.assertRaises(control.Cancelled):
            self.up()
        self.assertEqual(len(self.fake.videos), 0)


# --------------------------------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------------------------------- #
class Queue(YouTubeCase):
    def setUp(self):
        super().setUp()
        self.sign_in()

    def project(self, n=3, **kw):
        return make_clips(self.pid, n=n, **kw)

    def files(self, p):
        return [c["file"] for c in p["clips"]]

    def test_enqueue_skips_what_is_missing_waiting_or_already_on_youtube(self):
        p = self.project(3)
        f = self.files(p)
        r = uploads.enqueue(p, [f[0], f[0], "short_99_nope.mp4"])
        self.assertEqual(r["queued"], [f[0]])
        self.assertEqual([s["file"] for s in r["skipped"]], ["short_99_nope.mp4"])
        r = uploads.enqueue(p, [f[0]])
        self.assertIn("Already waiting", r["skipped"][0]["reason"])
        uploads.drain(self.pid)
        r = uploads.enqueue(p, [f[0], f[1]])
        self.assertEqual((r["queued"], r["skipped"][0]["reason"]), ([f[1]], "Already on YouTube."))
        r = uploads.enqueue(p, [f[0]], force=True)
        self.assertEqual(r["queued"], [f[0]])
        self.assertEqual(uploads.load(self.pid)[f[0]]["previous"]["video_id"], "vid1")

    def test_drain_uploads_everything_in_order_and_records_the_links(self):
        p = self.project(3)
        uploads.enqueue(p, self.files(p), privacy="unlisted")
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["failed"], res["waiting"]), (3, 0, 0))
        recs = uploads.load(self.pid)
        self.assertEqual([recs[f]["video_id"] for f in self.files(p)], ["vid1", "vid2", "vid3"])
        r0 = recs[self.files(p)[0]]
        self.assertEqual((r0["state"], r0["actual_privacy"], r0["notice"]), ("uploaded", "unlisted", ""))
        self.assertEqual(r0["url"], "https://www.youtube.com/shorts/vid1")
        self.assertIn("studio.youtube.com/video/vid1", r0["studio_url"])
        self.assertEqual(youtube.uploads_today(), 3)
        titles = [self.fake.inits[i]["meta"]["snippet"]["title"] for i in range(3)]
        self.assertEqual(titles, ["Kit title 0", "Kit title 1", "Kit title 2"])         # the publish kit is what goes on YouTube
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 0)                         # nothing to do the second time
        self.assertEqual(len(self.fake.videos), 3)

    def test_the_daily_cap_leaves_the_rest_queued(self):
        p = self.project(3)
        uploads.enqueue(p, self.files(p))
        with mock.patch.dict(os.environ, {"YOUTUBE_DAILY_UPLOADS": "2"}):
            res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["waiting"]), (2, 1))
        self.assertAlmostEqual(res["until"], youtube.next_pacific_midnight(time.time()), delta=5)
        self.assertIn("limit of 2", res["reason"])
        self.assertEqual(uploads.counts(self.pid), {"uploaded": 2, "queued": 1})

    def test_quota_exceeded_parks_everything_and_nothing_is_tried_while_parked(self):
        p = self.project(3)
        uploads.enqueue(p, self.files(p))
        self.fake.init_errors = [(403, "quotaExceeded", "quota")]
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["waiting"]), (0, 3))
        self.assertTrue(youtube.parked())
        self.assertEqual(uploads.counts(self.pid), {"queued": 3})                       # the clip is queued again, not failed
        inits = len(self.fake.inits)
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], len(self.fake.inits)), (0, inits))           # parked: Google is not even asked
        youtube.unpark()
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 3)

    def test_a_refused_clip_fails_alone_and_the_others_go_on(self):
        p = self.project(3)
        self.fake.reject_titles = {"Kit title 1"}
        uploads.enqueue(p, self.files(p))
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["failed"]), (2, 1))
        rec = uploads.load(self.pid)[self.files(p)[1]]
        self.assertEqual(rec["state"], "failed")
        self.assertIn("invalidPublishAt", rec["error"])
        self.fake.reject_titles = set()
        r = uploads.enqueue(p, [self.files(p)[1]])                                       # "Try again" puts it back
        self.assertEqual(r["queued"], [self.files(p)[1]])
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 1)

    def test_a_network_failure_stops_the_queue_and_retries_later(self):
        p = self.project(2)
        uploads.enqueue(p, self.files(p))
        self.fake.fail_puts = [503] * 100
        res = uploads.drain(self.pid)
        self.assertEqual((res["uploaded"], res["waiting"]), (0, 2))
        first = uploads.load(self.pid)[self.files(p)[0]]
        self.assertEqual(first["state"], "queued")
        self.assertTrue(first["error"])
        self.assertGreater(first["retry_after"], time.time() + 200)                      # it waits a few minutes before the next try
        self.assertEqual(uploads.pending_projects(), [self.pid])                         # the second clip was never tried: still due
        self.fake.fail_puts = []
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 1)                         # the second goes; the first waits for its time
        self.assertEqual(uploads.counts(self.pid), {"queued": 1, "uploaded": 1})
        self.assertEqual(uploads.pending_projects(), [])
        self.assertEqual(uploads.pending_projects(now=time.time() + 400), [self.pid])    # ...and is due again after the wait

    def test_a_scheduled_batch_is_private_and_spaced(self):
        p = self.project(3)
        start = time.time() + 3600
        r = uploads.enqueue(p, self.files(p), privacy="public", schedule_start=start, every_hours=6)
        times = [uploads.load(self.pid)[f]["publish_at"] for f in r["queued"]]
        parsed = [uploads._parse_utc(t) for t in times]
        self.assertEqual([round(b - a) for a, b in zip(parsed, parsed[1:])], [6 * 3600, 6 * 3600])
        self.assertAlmostEqual(parsed[0], start, delta=1)
        uploads.drain(self.pid)
        for init in self.fake.inits:
            self.assertEqual(init["meta"]["status"]["privacyStatus"], "private")
            self.assertIn("publishAt", init["meta"]["status"])
        self.assertEqual(init["meta"]["status"]["publishAt"], times[2])
        self.assertEqual(uploads.load(self.pid)[self.files(p)[0]]["notice"], "")        # a scheduled video is private on purpose

    def test_a_schedule_too_soon_or_a_silly_gap_is_refused(self):
        p = self.project(1)
        with self.assertRaisesRegex(ValueError, "10 minutes"):
            uploads.enqueue(p, self.files(p), schedule_start=time.time() + 60)
        with self.assertRaisesRegex(ValueError, "gap"):
            uploads.enqueue(p, self.files(p), schedule_start=time.time() + 7200, every_hours=0)
        with self.assertRaisesRegex(ValueError, "private, unlisted or public"):
            uploads.enqueue(p, self.files(p), privacy="friends")

    def test_a_schedule_that_passed_while_waiting_is_moved_never_sent_in_the_past(self):
        p = self.project(1)
        uploads.enqueue(p, self.files(p), schedule_start=time.time() + 3600)
        past = youtube.iso_utc(time.time() - 3600)
        uploads._update(self.pid, lambda d: d[self.files(p)[0]].update(publish_at=past))
        uploads.drain(self.pid)
        sent = uploads._parse_utc(self.fake.inits[0]["meta"]["status"]["publishAt"])
        self.assertGreater(sent, time.time() + 300)

    def test_when_youtube_makes_it_private_the_clip_says_so(self):
        p = self.project(1)
        self.fake.force_privacy = "private"
        uploads.enqueue(p, self.files(p), privacy="public")
        uploads.drain(self.pid)
        rec = uploads.load(self.pid)[self.files(p)[0]]
        self.assertEqual((rec["state"], rec["actual_privacy"]), ("uploaded", "private"))
        self.assertIn("made this video private", rec["notice"])

    def test_cancel_puts_the_running_clip_back_in_the_queue(self):
        p = self.project(2, size=youtube.CHUNK + 100_000)                                # two chunks: there is a gap to cancel in
        uploads.enqueue(p, self.files(p))
        ev = threading.Event()
        control.bind(ev, None)
        self.addCleanup(control.unbind)
        self.fake.on_put = lambda sid, n: ev.set()
        with self.assertRaises(control.Cancelled):
            uploads.drain(self.pid)
        self.assertEqual(uploads.counts(self.pid), {"queued": 2})
        self.assertEqual(len(self.fake.videos), 0)

    def test_not_connected_or_expired_raise_before_touching_the_queue(self):
        p = self.project(1)
        uploads.enqueue(p, self.files(p))
        youtube.disconnect()
        with self.assertRaises(youtube.NotConnected):
            uploads.drain(self.pid)
        self.assertEqual(uploads.counts(self.pid), {"queued": 1})
        self.sign_in()
        youtube._mutate(lambda d: d.update(needs_reconnect=True, reconnect_reason="expired"))
        with self.assertRaises(youtube.ReconnectNeeded):
            uploads.drain(self.pid)
        self.assertEqual(uploads.counts(self.pid), {"queued": 1})

    def test_the_kit_is_written_once_for_all_clips_that_lack_one(self):
        p = self.project(3, kit=False)
        uploads.enqueue(p, self.files(p))
        calls = []

        def make_kit(project, ids):
            calls.append(list(ids))
            def put(fresh):
                for i in ids:
                    fresh["candidates"][i]["kit"] = {"title": f"AI title {i}", "description": "d", "hashtags": ["#x"]}
            P.update_project(project, put)

        uploads.drain(self.pid, ensure_kit=make_kit)
        self.assertEqual(calls, [[0, 1, 2]])
        self.assertEqual(self.fake.inits[1]["meta"]["snippet"]["title"], "AI title 1")

    def test_a_failing_kit_still_uploads_with_the_working_titles(self):
        p = self.project(1, kit=False)
        uploads.enqueue(p, self.files(p))

        def broken(project, ids):
            raise RuntimeError("the AI is down")

        self.assertEqual(uploads.drain(self.pid, ensure_kit=broken)["uploaded"], 1)
        self.assertEqual(self.fake.inits[0]["meta"]["snippet"]["title"], "moment 0")

    def test_an_upload_that_was_cut_off_by_a_crash_is_picked_up_again(self):
        p = self.project(1)
        uploads.enqueue(p, self.files(p))
        uploads._update(self.pid, lambda d: d[self.files(p)[0]].update(state="uploading"))
        self.assertEqual(uploads.drain(self.pid)["uploaded"], 1)

    def test_dequeue_only_touches_waiting_and_failed_clips(self):
        p = self.project(3)
        f = self.files(p)
        uploads.enqueue(p, f[:1])
        uploads.drain(self.pid)                                                          # f[0] is on YouTube
        uploads.enqueue(p, f[1:])
        self.assertEqual(uploads.dequeue(self.pid, [f[0], f[1]]), 1)                     # f[0] is untouched, f[1] removed
        self.assertEqual(set(uploads.load(self.pid)), {f[0], f[2]})
        self.assertEqual(uploads.dequeue(self.pid), 1)
        self.assertEqual(set(uploads.load(self.pid)), {f[0]})

    def test_a_cancelled_reupload_goes_back_to_uploaded(self):
        p = self.project(1)
        f = self.files(p)[0]
        uploads.enqueue(p, [f])
        uploads.drain(self.pid)
        uploads.enqueue(p, [f], force=True)
        uploads.dequeue(self.pid, [f])
        self.assertEqual(uploads.load(self.pid)[f]["state"], "uploaded")

    def test_wide_and_long_clips_get_a_warning(self):
        p = self.project(2)
        p["clips"][0]["aspect_ratio"] = "16:9"
        p["clips"][1]["end"] = p["clips"][1]["start"] + 200
        P.save_project(p)
        r = uploads.enqueue(p, self.files(p))
        self.assertIn("normal video", r["warnings"][self.files(p)[0]][0])
        self.assertIn("3 minutes", r["warnings"][self.files(p)[1]][0])

    def test_uploads_survive_a_rerender_and_a_new_search(self):
        p = self.project(1)
        f = self.files(p)[0]
        uploads.enqueue(p, [f])
        uploads.drain(self.pid)
        P.update_project(p, lambda fresh: fresh.__setitem__("clips", []))              # what re-rendering / re-searching does to clips
        self.assertEqual(uploads.load(self.pid)[f]["state"], "uploaded")


# --------------------------------------------------------------------------------------------- #
# The web layer
# --------------------------------------------------------------------------------------------- #
def wait_job(client, job_id, timeout=15.0):
    end = time.time() + timeout
    job = {}
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish: " + json.dumps(job)[:400])


class WebApi(YouTubeCase):
    def setUp(self):
        super().setUp()
        self.c = webapp.app.test_client()
        env = Path(os.environ["LOCAL_OUTPUT_DIR"]).parent / "yt_app.env"
        patcher = mock.patch.object(config, "ENV_FILE", env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_status_and_config_never_contain_a_secret(self):
        self.sign_in()
        for path in ("/api/youtube", "/api/config", "/api/settings"):
            text = self.c.get(path).get_data(as_text=True)
            for secret in (self.fake.client_secret, REFRESH_TOKEN, "ya29."):
                self.assertNotIn(secret, text, path)
        s = self.c.get("/api/settings").get_json()
        self.assertTrue(s["youtube_client_secret_set"])
        self.assertEqual(s["youtube_client_id"], self.fake.client_id)
        self.assertNotIn("youtube_client_secret", s)

    def test_credentials_from_the_downloaded_json_or_the_two_values(self):
        text = json.dumps({"installed": {"client_id": "json-client-1.apps.googleusercontent.com", "client_secret": "GOCSPX-json-secret-1"}})
        r = self.c.post("/api/youtube/credentials", json={"json": text}, headers=H)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(os.environ["YOUTUBE_CLIENT_ID"], "json-client-1.apps.googleusercontent.com")
        self.assertNotIn("GOCSPX-json-secret-1", r.get_data(as_text=True))
        r = self.c.post("/api/youtube/credentials", json={"client_id": "two-values-1.apps.googleusercontent.com", "client_secret": "GOCSPX-two-2222"}, headers=H)
        self.assertEqual(r.status_code, 200)
        for bad in ({"json": "nope"}, {"client_id": "x"}, {"client_id": "bad id!", "client_secret": "GOCSPX-abcdefgh"}, {}):
            self.assertEqual(self.c.post("/api/youtube/credentials", json=bad, headers=H).status_code, 400, bad)

    def test_connect_needs_credentials_and_a_real_header(self):
        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": "", "YOUTUBE_CLIENT_SECRET": ""}):
            self.assertEqual(self.c.post("/api/youtube/connect", headers=H).status_code, 400)
        self.assertEqual(self.c.post("/api/youtube/connect").status_code, 403)               # no X-Studio header

    def test_the_whole_browser_flow_through_the_callback_route(self):
        url = self.c.post("/api/youtube/connect", headers=H).get_json()["url"]
        hop = requests.get(url, allow_redirects=False, timeout=5)                           # "Google" approves and redirects back
        self.assertEqual(hop.status_code, 302)
        back = urlparse(hop.headers["Location"])
        self.assertEqual(back.path, "/")                                                    # the redirect form Google documents for Desktop clients
        self.assertTrue(self.fake.redirects[0].startswith("http://127.0.0.1:"))
        r = self.c.get(back.path + "?" + back.query)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertNotIn("<script", r.get_data(as_text=True).lower())                       # the callback page, not the app page
        self.assertIn("Test Channel", r.get_data(as_text=True))
        self.assertEqual(r.headers["Content-Security-Policy"].split(";")[0], "default-src 'none'")
        self.assertTrue(self.c.get("/api/youtube").get_json()["connected"])

    def test_a_forged_or_failed_callback_is_refused_and_escaped(self):
        r = self.c.get("/oauth/youtube/callback?state=forged&code=x")
        self.assertEqual(r.status_code, 400)
        r = self.c.get("/?state=forged&code=x")                                             # the root form too
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.c.get("/").status_code, 200)                                  # a plain visit is still the app
        self.assertEqual(self.c.get("/?state=only-a-state").status_code, 200)
        self.assertFalse(self.c.get("/api/youtube").get_json()["connected"])
        url = self.c.post("/api/youtube/connect", headers=H).get_json()["url"]
        state = parse_qs(urlparse(url).query)["state"][0]
        r = self.c.get("/oauth/youtube/callback", query_string={"state": state, "error": "<script>alert(1)</script>"})
        body = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("<script>", body)

    def test_disconnect(self):
        self.sign_in()
        self.assertFalse(self.c.post("/api/youtube/disconnect", headers=H).get_json()["connected"])

    def test_upload_requests_are_validated(self):
        p = make_clips(self.pid, n=2)
        f = [c["file"] for c in p["clips"]]
        url = f"/api/projects/{self.pid}/upload"
        self.assertEqual(self.c.post(url, json={"files": f}, headers=H).status_code, 400)           # not connected yet
        self.sign_in()
        for body in ({}, {"files": []}, {"files": "x.mp4"}, {"files": ["../project.json"]}, {"files": ["a.txt"]}, {"files": ["x.mp4"] * 61},
                     {"files": f, "privacy": "friends"}, {"files": f, "schedule_start": "soon"}, {"files": f, "every_hours": 0},
                     {"files": f, "schedule_start": time.time() + 30}):
            self.assertEqual(self.c.post(url, json=body, headers=H).status_code, 400, body)
        r = self.c.post(url, json={"files": ["short_77_x.mp4"]}, headers=H)                        # unknown clip: nothing to do, no job
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.get_json()["job"], r.get_json()["queued"]), (None, []))

    def test_upload_job_end_to_end(self):
        p = make_clips(self.pid, n=3)
        self.sign_in()
        f = [c["file"] for c in p["clips"]]
        r = self.c.post(f"/api/projects/{self.pid}/upload", json={"files": f, "privacy": "unlisted"}, headers=H)
        self.assertEqual(r.status_code, 202)
        self.assertEqual(r.get_json()["queued"], f)
        job = wait_job(self.c, r.get_json()["job"]["id"])
        self.assertEqual(job["status"], "done", job)
        self.assertEqual((job["result"]["uploaded"], job["result"]["waiting"]), (3, 0))
        self.assertIn("upload", job["stages"])
        proj = self.c.get(f"/api/projects/{self.pid}").get_json()
        states = {c["file"]: c["youtube"]["state"] for c in proj["clips"]}
        self.assertEqual(set(states.values()), {"uploaded"})
        self.assertTrue(all(c["youtube"]["url"].startswith("https://www.youtube.com/shorts/vid") for c in proj["clips"]))
        again = self.c.post(f"/api/projects/{self.pid}/upload", json={"files": f}, headers=H).get_json()
        self.assertEqual((again["job"], len(again["skipped"])), (None, 3))                           # never uploaded twice

    def test_an_expired_sign_in_fails_the_job_with_a_clear_message_and_keeps_the_queue(self):
        p = make_clips(self.pid, n=1)
        self.sign_in(expires_in=-10)
        self.fake.invalid_grant = True
        r = self.c.post(f"/api/projects/{self.pid}/upload", json={"files": [p["clips"][0]["file"]]}, headers=H)
        self.assertEqual(r.status_code, 202)
        job = wait_job(self.c, r.get_json()["job"]["id"])
        self.assertEqual(job["status"], "error")
        self.assertIn("Reconnect", job["error"])
        self.assertEqual(job["request"]["path"], f"/api/projects/{self.pid}/upload/run")              # "Try again" re-runs the queue
        self.assertEqual(uploads.counts(self.pid), {"queued": 1})

    def test_run_and_dequeue_routes(self):
        p = make_clips(self.pid, n=2)
        f = [c["file"] for c in p["clips"]]
        self.sign_in()
        url = f"/api/projects/{self.pid}/upload"
        self.assertEqual(self.c.post(url + "/run", headers=H, json={}).status_code, 400)             # nothing queued
        uploads.enqueue(p, f)
        self.assertEqual(self.c.post(url + "/dequeue", json={}, headers=H).status_code, 400)
        self.assertEqual(self.c.post(url + "/dequeue", json={"files": ["../x.mp4"]}, headers=H).status_code, 400)
        self.assertEqual(self.c.post(url + "/dequeue", json={"files": [f[0]]}, headers=H).get_json(), {"removed": 1})
        r = self.c.post(url + "/run", headers=H, json={})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(wait_job(self.c, r.get_json()["id"])["result"]["uploaded"], 1)
        uploads.enqueue(p, [f[0]])
        self.assertEqual(self.c.post(url + "/dequeue", json={"all": True}, headers=H).get_json(), {"removed": 1})

    def test_try_now_ends_the_wait_and_sends_the_queue(self):
        p = make_clips(self.pid, n=1)
        f = p["clips"][0]["file"]
        self.assertEqual(self.c.post("/api/youtube/resume", headers=H).status_code, 400)             # not connected
        self.sign_in()
        uploads.enqueue(p, [f])
        youtube.park(time.time() + 3600, "limit")
        youtube.set_paused(True)
        self.assertEqual(self.c.post("/api/youtube/resume", headers=H).status_code, 400)             # paused stays paused
        self.assertTrue(youtube.parked())
        youtube.set_paused(False)
        with mock.patch.dict(os.environ, {"YOUTUBE_AUTO_RESUME": "0"}):                              # the press counts anyway
            r = self.c.post("/api/youtube/resume", headers=H).get_json()
        self.assertEqual((r["started"], r["parked_until"], len(r["jobs"])), (1, None, 1))
        self.assertEqual(wait_job(self.c, r["jobs"][0]["id"])["result"]["uploaded"], 1)
        self.assertEqual(uploads.counts(self.pid), {"uploaded": 1})

    def test_the_watcher_resumes_waiting_uploads_once_and_respects_the_rules(self):
        p = make_clips(self.pid, n=1)
        f = p["clips"][0]["file"]
        self.sign_in()
        uploads.enqueue(p, [f])
        with mock.patch.dict(os.environ, {"YOUTUBE_AUTO_RESUME": "0"}):
            self.assertEqual(webapp._upload_tick(), 0)                                               # switched off in Settings
        youtube.park(time.time() + 3600, "quota")
        self.assertEqual(webapp._upload_tick(), 0)                                                   # YouTube said wait
        youtube.unpark()
        with mock.patch.dict(os.environ, {"YOUTUBE_DAILY_UPLOADS": "1"}):
            youtube.count_upload()
            self.assertEqual(webapp._upload_tick(), 0)                                               # today's cap is used
        youtube._mutate(lambda d: d.pop("uploads", None))
        gate = threading.Event()
        real = webapp.pipeline.upload_local
        webapp.pipeline.upload_local = lambda pid: (gate.wait(5), real(pid))[1]                      # hold the job so a 2nd tick sees it
        try:
            self.assertEqual(webapp._upload_tick(), 1)
            self.assertEqual(webapp._upload_tick(), 0)                                               # no second job for the same project
            gate.set()
            deadline = time.time() + 15
            while time.time() < deadline and uploads.counts(self.pid).get("uploaded") != 1:
                time.sleep(0.05)
        finally:
            webapp.pipeline.upload_local = real
            gate.set()
        self.assertEqual(uploads.counts(self.pid), {"uploaded": 1})

    def test_public_project_carries_the_upload_state_of_each_clip(self):
        p = make_clips(self.pid, n=2)
        self.sign_in()
        uploads.enqueue(p, [p["clips"][0]["file"]])
        d = self.c.get(f"/api/projects/{self.pid}").get_json()
        self.assertEqual([c["youtube"] and c["youtube"]["state"] for c in d["clips"]], ["queued", None])


class Autopilot(YouTubeCase):
    """link -> best clips -> render -> upload, with analysis and rendering replaced by fakes."""

    def setUp(self):
        super().setUp()
        self.c = webapp.app.test_client()
        self.calls = {}

        def fake_analyze(url, opts):
            self.calls["analyze"] = url
            return make_clips(self.pid, n=4, kit=False, rendered=False)      # scores 90, 85, 80, 75; no clip rendered yet
        self.fake_analyze = fake_analyze

        def fake_render(project, ids, options, quality=1080, overrides=None, kit=False):
            self.calls["render"] = {"ids": list(ids), "kit": kit, "quality": quality, "layout": options.layout}
            folder = P.project_dir(project["id"])
            records = []
            for i in ids:
                name = f"short_{i + 1:02d}_m{i}.mp4"
                (folder / name).write_bytes(os.urandom(50_000))
                records.append({"candidate": i, "file": name, "title": f"moment {i}", "start": 10.0 + i * 60, "end": 40.0 + i * 60,
                                "aspect_ratio": "9:16", "layout": "auto", "created": "now"})
            def put(fresh):
                for i in ids:
                    fresh["candidates"][i]["kit"] = {"title": f"Auto title {i}", "description": "d", "hashtags": ["#a"]}
                fresh["clips"] = records
            P.update_project(project, put)
            return records
        self.fake_render = fake_render

    def run_autopilot(self, body):
        with mock.patch.object(webapp.pipeline, "analyze_local", self.fake_analyze), mock.patch.object(webapp.pipeline, "render_local", self.fake_render):
            r = self.c.post("/api/analyze", json={"urls": ["https://youtu.be/auto1"], **body}, headers=H)
            if r.status_code != 202:
                return r, None
            return r, wait_job(self.c, r.get_json()["jobs"][0]["id"])

    def test_it_needs_a_connected_channel_and_a_clip_count(self):
        r, _ = self.run_autopilot({"auto_render": 3, "upload": {"privacy": "private"}})
        self.assertEqual(r.status_code, 400)
        self.assertIn("Connect your YouTube", r.get_json()["error"])
        self.sign_in()
        r, _ = self.run_autopilot({"upload": {"privacy": "private"}})
        self.assertEqual(r.status_code, 400)
        r, _ = self.run_autopilot({"auto_render": 3, "upload": {"privacy": "friends"}})
        self.assertEqual(r.status_code, 400)

    def test_it_renders_the_best_clips_and_uploads_them(self):
        self.sign_in()
        r, job = self.run_autopilot({"auto_render": 3, "upload": {"privacy": "public"}, "render": {"layout": "split"}})
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(self.calls["render"]["ids"], [0, 1, 2])
        self.assertTrue(self.calls["render"]["kit"], "uploads need a title and hashtags")
        self.assertEqual(self.calls["render"]["layout"], "split")
        self.assertEqual((job["result"]["uploaded"], job["result"]["queued"], job["result"]["project_id"]), (3, 3, self.pid))
        self.assertEqual([i["meta"]["snippet"]["title"] for i in self.fake.inits], ["Auto title 0", "Auto title 1", "Auto title 2"])
        self.assertEqual(self.fake.inits[0]["meta"]["status"]["privacyStatus"], "public")
        self.assertEqual(len(self.fake.videos), 3)
        self.assertEqual(job["stages"][-1], "upload")

    def test_only_clips_above_the_minimum_score_are_used(self):
        self.sign_in()
        _, job = self.run_autopilot({"auto_render": 4, "auto_min_score": 83, "upload": {}})
        self.assertEqual(self.calls["render"]["ids"], [0, 1])                 # scores 90 and 85
        self.assertEqual(job["result"]["uploaded"], 2)

    def test_a_minimum_score_nobody_reaches_explains_itself(self):
        self.sign_in()
        _, job = self.run_autopilot({"auto_render": 3, "auto_min_score": 99, "upload": {}})
        self.assertEqual(job["status"], "error")
        self.assertIn("scored 99 or more", job["error"])
        self.assertNotIn("render", self.calls)
        self.assertEqual(len(self.fake.videos), 0)

    def test_a_spread_out_schedule_starts_counting_when_the_clips_are_ready(self):
        self.sign_in()
        t0 = time.time()
        _, job = self.run_autopilot({"auto_render": 2, "upload": {"privacy": "public", "start_in_hours": 2, "every_hours": 12}})
        self.assertEqual(job["status"], "done", job)
        times = [uploads._parse_utc(i["meta"]["status"]["publishAt"]) for i in self.fake.inits]
        self.assertAlmostEqual(times[0], t0 + 2 * 3600, delta=120)
        self.assertEqual(round(times[1] - times[0]), 12 * 3600)
        self.assertTrue(all(i["meta"]["status"]["privacyStatus"] == "private" for i in self.fake.inits))

    def test_the_daily_cap_applies_and_the_rest_waits_for_the_watcher(self):
        self.sign_in()
        with mock.patch.dict(os.environ, {"YOUTUBE_DAILY_UPLOADS": "1"}):
            _, job = self.run_autopilot({"auto_render": 3, "upload": {}})
        self.assertEqual((job["status"], job["result"]["uploaded"], job["result"]["waiting"]), ("done", 1, 2))
        self.assertTrue(job["result"]["until"])
        self.assertEqual(uploads.counts(self.pid), {"uploaded": 1, "queued": 2})

    def test_without_upload_it_behaves_as_before(self):
        _, job = self.run_autopilot({"auto_render": 2})
        self.assertEqual((job["status"], self.calls["render"]["kit"]), ("done", False))
        self.assertNotIn("uploaded", job["result"])
        self.assertEqual(len(self.fake.inits), 0)


class Hygiene(YouTubeCase):
    def test_tokens_and_secrets_are_redacted_from_logs_and_diagnostics(self):
        text = (f"refresh {REFRESH_TOKEN} access ya29.a0AfH6SMBx-abcdefghijklmnopqrstuvwxyz1234 "
                f"secret GOCSPX-abcdefghijklmnop url http://127.0.0.1:8765/oauth/youtube/callback?state=s&code=4/0AbCdEfGhIj-123 "
                f"env {self.fake.client_secret}")
        out = logsetup.redact(text)
        for leak in (REFRESH_TOKEN, "ya29.a0AfH6", "GOCSPX-abcdefghijklmnop", "4/0AbCdEfGhIj-123", self.fake.client_secret):
            self.assertNotIn(leak, out)
        self.assertIn("state=s", out)

    def test_the_doctor_reports_the_youtube_state_without_a_channel_name(self):
        check = health.youtube_check
        self.assertIn("youtube_channel", [c["id"] for c in health.doctor(deep=False)["checks"]])

        with mock.patch.dict(os.environ, {"YOUTUBE_CLIENT_ID": "", "YOUTUBE_CLIENT_SECRET": ""}):
            self.assertEqual((check()["status"], check()["detail"]), ("ok", "not set up"))
        self.assertEqual(check()["status"], "warn")                                    # client saved, not connected
        self.sign_in()
        c = check()
        self.assertEqual(c["status"], "ok")
        self.assertNotIn("Test Channel", json.dumps(c))
        youtube._mutate(lambda d: d.update(needs_reconnect=True, reconnect_reason="x"))
        self.assertEqual(check()["status"], "warn")

    def test_the_state_file_is_never_in_the_diagnostics_zip(self):
        import io
        import zipfile

        self.sign_in()
        data = health.diagnostics_zip(logsetup.log_dir(), [])
        z = zipfile.ZipFile(io.BytesIO(data))
        blob = b"".join(z.read(n) for n in z.namelist())
        self.assertNotIn(REFRESH_TOKEN.encode(), blob)
        self.assertNotIn(b"ya29.", blob)
        self.assertFalse([n for n in z.namelist() if "youtube" in n.lower() and n != "health.json"])


if __name__ == "__main__":
    unittest.main()

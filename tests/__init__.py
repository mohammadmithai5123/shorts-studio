"""Offline test-suite:  python -m unittest discover -s tests -t .

Nothing here touches the network, your .env keys, or your real ``output/`` folder: the output directory is a
throw-away temp folder, set before any shorts_generator module is imported.
"""
import atexit
import os
import shutil
import tempfile

_TMP = tempfile.mkdtemp(prefix="shorts_tests_")
os.environ["LOCAL_OUTPUT_DIR"] = os.path.join(_TMP, "output")
os.environ.pop("YTDLP_COOKIES_BROWSER", None)
# config reads the real .env on import; an *existing* (even empty) variable wins over it, so no test can ever use your
# real keys, proxy or cookies (a live LLM call from a test would spend your free daily quota).
for _name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "YTDLP_PROXY", "YTDLP_COOKIES_FILE", "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET",
              "YOUTUBE_PRIVACY", "YOUTUBE_CATEGORY", "YOUTUBE_DAILY_UPLOADS", "YOUTUBE_AUTO_RESUME", "YOUTUBE_NOTIFY_SUBSCRIBERS", "LOCAL_ENCODER",
              "LOCAL_CAPTION_SCRIPT"):
    os.environ[_name] = ""
for _name in ("LOCAL_FACECAM_BG", "LOCAL_CACHE_DAYS"):
    os.environ[_name] = ""
os.environ["LOCAL_DELETE_UPLOADED"] = "0"                 # tests that look at clip files must find them; clean-up has its own tests
os.environ["LOCAL_MEME_DIR"] = os.path.join(_TMP, "memes")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

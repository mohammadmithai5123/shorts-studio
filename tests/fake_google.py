"""A small fake of the Google endpoints the YouTube feature talks to, so the whole flow is tested without Google.

It speaks the real protocols closely enough to catch real bugs: OAuth code + PKCE and refresh tokens, ``channels.list``, and the
resumable upload protocol (session start with ``Location``, chunks with ``Content-Range``, ``308`` + ``Range`` answers, status
queries with ``bytes */TOTAL``). Failures can be injected: 503s, dropped connections, expired sessions, quota errors, 401s.

Use it as ``with mock.patch.dict(youtube.URLS, fake.urls()):``.
"""
import base64
import hashlib
import json
import re
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
REFRESH_TOKEN = "1//0g-fake-refresh-token-for-the-tests-0123456789"


class FakeGoogle:
    def __init__(self, client_id: str = "123456789-test.apps.googleusercontent.com", client_secret: str = "GOCSPX-fake-test-secret-01"):
        self.client_id, self.client_secret = client_id, client_secret
        self.log = []                          # (method, path) of every request
        self.valid = set()                     # access tokens that currently work
        self.codes = {}                        # authorization code -> PKCE challenge
        self.grant_scope = f"{UPLOAD_SCOPE} {READ_SCOPE}"
        self.issue_refresh = True
        self.channel = {"id": "UC_fake", "title": "Test Channel"}
        self.invalid_grant = False             # the next refresh answers invalid_grant
        self.expire_tokens = False             # the next authorised call answers 401 and every token stops working
        self.init_errors = []                  # [(status, reason, message)] consumed by session starts
        self.fail_puts = []                    # 503 | "drop" | 404 | 400, consumed by PUTs that carry data
        self.testing_mode = False              # the consent screen is still in "Testing": the token answer carries refresh_token_expires_in
        self.stall = False                     # PUTs are answered 308 without storing anything
        self.on_put = None                     # callback(session id, bytes stored so far) after each stored chunk
        self.reject_titles = set()
        self.force_privacy = None
        self.inits, self.sessions, self.videos, self.revoked, self.redirects, self.put_ranges = [], {}, {}, [], [], []
        self._n = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):         # keep the test output quiet
                pass

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _send(self, code, obj=None, headers=None):
                raw = json.dumps(obj).encode() if obj is not None else b""
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)

            def _error(self, code, reason, message="error"):
                self._send(code, {"error": {"code": code, "message": message, "errors": [{"reason": reason, "message": message}]}})

            def _authorised(self) -> bool:
                if fake.expire_tokens:
                    fake.expire_tokens = False
                    fake.valid.clear()
                if self.headers.get("Authorization", "")[7:] not in fake.valid:
                    self._error(401, "authError", "Invalid Credentials")
                    return False
                return True

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                fake.log.append(("GET", url.path))
                q = urllib.parse.parse_qs(url.query)
                if url.path == "/auth":                       # the consent page: approve at once and go back to the app
                    code = f"code{len(fake.codes) + 1}"
                    fake.codes[code] = q["code_challenge"][0]
                    fake.redirects.append(q["redirect_uri"][0])
                    self.send_response(302)
                    self.send_header("Location", q["redirect_uri"][0] + "?" + urllib.parse.urlencode({"code": code, "state": q["state"][0]}))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif url.path == "/channels":
                    if not self._authorised():
                        return
                    self._send(200, {"items": [{"id": fake.channel["id"], "snippet": {"title": fake.channel["title"],
                                                                                      "thumbnails": {"default": {"url": "https://example.test/t.jpg"}}}}]}
                               if fake.channel else {"items": []})
                else:
                    self._error(404, "notFound")

            def do_POST(self):
                url = urllib.parse.urlparse(self.path)
                fake.log.append(("POST", url.path))
                body = self._body()
                if url.path == "/token":
                    form = {k: v[0] for k, v in urllib.parse.parse_qs(body.decode()).items()}
                    if form.get("client_id") != fake.client_id or form.get("client_secret") != fake.client_secret:
                        return self._send(401, {"error": "invalid_client"})
                    if form.get("grant_type") == "authorization_code":
                        challenge = fake.codes.pop(form.get("code", ""), None)
                        verifier = form.get("code_verifier", "")
                        digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                        if challenge is None or digest != challenge:
                            return self._send(400, {"error": "invalid_grant", "error_description": "Bad code or verifier"})
                        out = {"access_token": fake.new_token(), "expires_in": 3600, "scope": fake.grant_scope, "token_type": "Bearer"}
                        if fake.issue_refresh:
                            out["refresh_token"] = REFRESH_TOKEN
                        if fake.testing_mode:
                            out["refresh_token_expires_in"] = 604799
                        return self._send(200, out)
                    if form.get("grant_type") == "refresh_token":
                        if fake.invalid_grant or form.get("refresh_token") != REFRESH_TOKEN:
                            return self._send(400, {"error": "invalid_grant", "error_description": "Token has been expired or revoked."})
                        return self._send(200, {"access_token": fake.new_token(), "expires_in": 3600, "token_type": "Bearer"})
                    return self._send(400, {"error": "unsupported_grant_type"})
                if url.path == "/revoke":
                    fake.revoked.append(urllib.parse.parse_qs(body.decode()).get("token", [""])[0])
                    return self._send(200, {})
                if url.path == "/upload":
                    if not self._authorised():
                        return
                    meta = json.loads(body.decode("utf-8") or "{}")
                    fake.inits.append({"meta": meta, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "query": urllib.parse.parse_qs(url.query)})
                    if fake.init_errors:
                        code, reason, message = fake.init_errors.pop(0)
                        return self._error(code, reason, message)
                    if meta.get("snippet", {}).get("title") in fake.reject_titles:
                        return self._error(400, "invalidPublishAt", "Invalid scheduled publishing time")
                    sid = f"s{len(fake.sessions) + 1}"
                    fake.sessions[sid] = {"meta": meta, "data": bytearray(), "size": int(self.headers.get("X-Upload-Content-Length") or 0)}
                    return self._send(200, {}, {"Location": f"{fake.base}/session/{sid}"})
                self._error(404, "notFound")

            def do_PUT(self):
                url = urllib.parse.urlparse(self.path)
                fake.log.append(("PUT", url.path))
                body = self._body()
                fake.put_ranges.append(self.headers.get("Content-Range"))
                sess = fake.sessions.get(url.path.rsplit("/", 1)[-1])
                if sess is None:
                    return self._error(404, "notFound", "unknown session")
                if fake.stall and body:
                    return self._send(308, None)
                if body and fake.fail_puts:
                    action = fake.fail_puts.pop(0)
                    if action == "drop":
                        self.close_connection = True
                        try:
                            self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                        return
                    if action == 404:
                        fake.sessions.pop(url.path.rsplit("/", 1)[-1], None)
                        return self._error(404, "notFound", "session expired")
                    return self._error(action, "backendError" if action >= 500 else "badRequest", "injected")
                m = re.match(r"bytes (?:(\d+)-(\d+)|\*)/(\d+)", self.headers.get("Content-Range", ""))
                if not m:
                    return self._error(400, "badRequest", "no Content-Range")
                if m.group(1) is not None and int(m.group(1)) == len(sess["data"]):
                    sess["data"].extend(body)
                    if fake.on_put:
                        fake.on_put(url.path, len(sess["data"]))
                if len(sess["data"]) >= sess["size"]:
                    vid = f"vid{len(fake.videos) + 1}"
                    privacy = fake.force_privacy or sess["meta"].get("status", {}).get("privacyStatus")
                    fake.videos[vid] = {"meta": sess["meta"], "bytes": bytes(sess["data"])}
                    return self._send(201, {"id": vid, "status": {"privacyStatus": privacy, "uploadStatus": "uploaded"}})
                self._send(308, None, {"Range": f"bytes=0-{len(sess['data']) - 1}"} if sess["data"] else {})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True)
        self._thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def new_token(self) -> str:
        self._n += 1
        token = f"ya29.fake-access-token-number-{self._n:04d}-abcdefghij"
        self.valid.add(token)
        return token

    def urls(self) -> dict:
        return {"auth": self.base + "/auth", "token": self.base + "/token", "revoke": self.base + "/revoke",
                "channels": self.base + "/channels", "upload": self.base + "/upload"}

    def calls(self, method: str, path: str) -> int:
        return sum(1 for m, p in self.log if m == method and p == path)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

# Architecture

This file is the map of the code. **Keep it true:** when you add a module, change a contract (a JSON shape,
a function other layers call) or learn something that changes a design decision, update it in the same change.

## What the tool does

Give it a long video (a link or a file). It finds the moments most worth cutting into Shorts / Reels, lets a
person preview and trim them, and renders vertical clips (facecam + gameplay, speaker tracking, or the whole
frame), optionally with captions, a title card, a watermark and a ready-to-paste title + hashtags.

Two modes share the ranking logic:

| | `--mode api` | `--mode local` (default for the UI) |
|---|---|---|
| Who does the work | MuAPI (server-side download, transcribe, LLM, crop) | this machine |
| Video download | by MuAPI | **never the whole video** (see "Streaming") |
| Code | `muapi.py`, `downloader.py`, `transcriber.py`, `clipper.py` (top level) | everything under `local/` |

## Design principles

1. **Never download the whole video.** Metadata, captions and the audio track are small; a clip is fetched as its own
   time range. A 2-hour video used to be 1.7 GB and 30 minutes; now it is a few MB plus the clips you render.
2. **Pure logic is separate from I/O.** `highlights.py` knows transcripts and prompts, not files or networks;
   `captions.py` turns data into text; `signals.py` is arithmetic. They are the easy-to-test core.
3. **Everything expensive is cached and resumable.** Transcript, loudness series, every LLM answer. A crash, a 503 or a
   daily-quota error never costs the same work twice; "try again" continues where it stopped.
4. **Degrade, don't die.** Signals are bonuses (no audio -> no energy signal), captions have a Whisper fallback,
   LLM calls fall back across models, one failed clip does not lose the others.
5. **Cooperative cancel + structured progress** through one tiny module (`local/control.py`), so library code never
   knows whether a UI exists.
6. **Heavy imports are lazy.** `--help`, api mode and the unit tests never import OpenCV, yt-dlp or Flask unless used.
7. **The UI is local and locked down** (127.0.0.1 only, Host check, custom header on writes, no secrets ever returned).

## Data flow (local mode)

```
 link / file
    |  source.resolve()                      ~5 s, metadata only (title, duration, captions list, heatmap, formats)
    v
 Source ──┬── transcript.get_transcript()    YouTube captions (json3, ~2 s, word timing)  -> else Whisper on the audio track
          ├── audio.load_energy_db()         audio-only file (yt-dlp, ~50 s / 2 h) -> loudness per second -> file deleted   [background thread]
          └── heatmap_series()               YouTube "most replayed" (when the video has it)
    |
    v
 highlights.get_highlights()                 chunk (20 min) -> LLM candidates -> dedupe -> GLOBAL re-rank (+ signals blended)
    |                                        -> snap to sentences -> enforce clip length
    v
 project.json  (ranked candidates)   ───────►  UI: timeline, preview (YouTube player), trim, add own clip, pick
    |
    v   render_local(ids, options)
 source.SectionFetcher                       fetch ONLY each clip's range, H.264, <= chosen height, 2 ranges ahead
    |   (a local file is cut with ffmpeg instead)
    v
 pipeline._edit_notes_for()   (editing on)   one AI call: headline / popups / highlighted words / memes (75 s budget, optional)
    v
 clipper.render_clip()                       analyse (bars, faces, cuts) -> framing (split / per camera shot) -> edit plan
                                             -> threads compose frames -> ffmpeg (Quick Sync / NVENC / x264; sound cut + SFX,
                                             ASS captions, loudnorm) -> short_NN_<title>.mp4
    v
 quality.verify()                            size, length, sound, not black: a broken clip is reported, never kept
    v
 publish.make_publish_kit()   (optional)     title / description / hashtags per clip
```

## Module map

```
main.py                     CLI. Parses flags, calls pipeline.generate_shorts(), prints results. `--doctor [--deep]`
                            prints the health report, `--version`, `--clean`. Tracebacks go to the log file.
app.py                      Web layer ONLY: validation, one-at-a-time job queue (progress, cancel, live log, history
                            that survives a restart, exact retry), JSON endpoints, media serving, zip export,
                            security guards (Host check, X-Studio header, CSP), the YouTube routes (sign-in callback, upload
                            queue) and the upload watcher thread. No pipeline logic.
webui/                      index.html + app.css + app.js. Plain JS, no build, text via textContent only.
run_ui.bat                  Windows launcher (console window).
launcher.pyw                App-style start: hidden server, keeper that restarts it, Edge/Chrome app window; --install/--uninstall.
install_app.bat             Desktop + Start menu icon and start with Windows (local/desktop.py).

shorts_generator/
  __init__.py               exports generate_shorts, __version__
  _version.py               the single place the version number lives (shown in the UI, CLI, diagnostics)
  logsetup.py               rotating log file output/logs/studio.log (+ library loggers); log_dir(); redact() hides keys,
                            user:password@ in URLs and the saved key/proxy values in the log and in every diagnostics file
  config.py                 .env + environment; live accessors for settings the UI can change; save_settings()
  pipeline.py               ORCHESTRATION: analyze_local, reanalyze_local, render_local, publish_kit_local, upload_local,
                            generate_shorts. The only module that wires the pieces together.
  highlights.py             PURE ranking logic (prompts, chunking, re-rank, signal blending, snap, duration).
                            Pluggable `llm_fn`; shared by both modes.
  muapi.py downloader.py transcriber.py clipper.py     api mode (MuAPI does everything server-side)

  local/
    source.py               Where a video comes from. resolve() -> Source; captions track choice; audio file;
                            SectionFetcher (clip ranges); friendly_error() for yt-dlp failures.
    transcript.py           Captions -> transcript (json3 / vtt parsers), Whisper fallback, per-video cache.
    transcriber.py          Whisper (faster-whisper) — OPTIONAL dependency, imported lazily.
    audio.py                Per-second loudness from the audio file; cache.
    signals.py              Percentile of a window vs all windows of the same length; heatmap -> series.
    llm.py                  Gemini / OpenAI over HTTPS, model discovery + fallback chain, answer cache,
                            error classes (auth / quota / overloaded / notfound).
    clipper.py              Frame analysis, layout choice, edit orchestration, the threaded render engine, RenderOptions.
    faces.py                YuNet face detector (assets/models, MIT; sha256 in assets/models/README.txt); Haar fallback.
    cutout.py               Green-screen streamer over a game: detection (position runs, evidence from motion) and compositing
                            (PP-HumanSeg person on a colour on top, game below, streamer smoothed out of the game).
    memes.py                The person's meme library (Memes Assets/, memes/) described by assets/meme_catalog.json; cue
                            planning, full-screen and green-screen pictures, levelled sound.
    housekeeping.py         Automatic clean-up: posted clips, stale cache / AI answers, leftovers (startup, after jobs, 3-hourly).
    desktop.py              Shortcuts and the HKCU Run value for starting with Windows.
    framing.py              Pure framing: camera cuts, face tracks per shot, performers vs bystanders, crop / group / stack /
                            fit per shot, and the crop / zoom / compose geometry.
    editing.py              Pure edit planning: pauses -> kept ranges (frame grid), timeline mapping, zooms, popups, sounds,
                            headline, end card; presets off / clean / viral.
    overlays.py             Popups (colour emoji + badges via Pillow, sticker rim, bounce) and the progress bar.
    sfx.py                  Synthesised pop / whoosh / ding written to one WAV per clip.
    encoder.py              Which H.264 encoder works (qsv / nvenc / amf, else x264); `LOCAL_ENCODER`.
    quality.py              Checks a rendered file before it is kept or uploaded.
    captions.py             Transcript -> ASS subtitles: pop (animated, per word) + 4 styles, hook, end card, watermark;
                            strips bracketed sound notes ("[हंसी]", "[Music]") and remembers laughter / applause.
    romanize.py             Hindi / Urdu script -> Roman letters (word list -> cached AI answers -> rules); timing untouched.
                            Applied by render_local (captions, title, edit notes) and uploads (title, description); the
                            saved transcript and candidates keep the original script.
    publish.py              LLM-written title / description / hashtags.
    project.py              project.json store (schema 2, upgrades schema 1), cache/clip paths, list/delete. One lock per
                            project; update_project(project, mutate) = re-read, mutate, save (see "Concurrency").
    youtube.py              Your YouTube channel: OAuth 2.0 (Desktop client, PKCE, loopback redirect to this app), token refresh,
                            the resumable upload protocol, error classification (QuotaExceeded / ReconnectNeeded / UploadRejected /
                            NetworkError), per-Pacific-day counter, "parked until", video metadata. Plain `requests`, no Google SDK.
                            Every URL goes through `youtube.URLS` so tests can point it at a fake.
    uploads.py              The upload queue: <project>/uploads.json (state per clip file), enqueue / dequeue / drain, scheduling
                            (publishAt), the daily cap, never-twice, pending_projects() for the watcher.
    storage.py              Disk usage + cleanup of old whole-video downloads and caches.
    health.py               Doctor (ffmpeg, libass, OpenCV, yt-dlp age/latest, key, disk, writable, whisper, login/proxy;
                            `deep` also tests the network and the AI key), in-app yt-dlp update, disk-space preflight
                            (LowDiskSpace), diagnostics zip (never contains a secret).
    control.py              Cancel flag + structured progress (no-ops when nothing is bound).

tests/                      Offline unittest suite (no network, temp output dir). See "Testing".
tests/fake_google.py        A local fake of Google's OAuth + YouTube upload endpoints with failure injection.
```

**Dependency direction** (never import upward): `app.py / main.py` -> `pipeline` -> `local/*` and `highlights` ->
`config`. `highlights.py` imports nothing from `local/` except `control` (inside a function). `local/*` modules do
not import `pipeline` or `app`. Keep it that way: it is what makes the pieces testable.

## Contracts

### Transcript (`local/transcript.py`)
```json
{"duration": 7439.0, "language": "hi", "source": "youtube-captions:auto:hi-orig",
 "segments": [{"start": 0.3, "end": 2.6, "text": "...", "words": [{"start": 0.3, "end": 0.7, "text": "..."}]}]}
```
Segments never overlap. `words` is optional (Whisper has none; captions then spread words evenly).

### Candidate (inside `project.json`, produced by `highlights.get_highlights`)
`{id, title, start_time, end_time, score 0-100|null, hook_sentence, virality_reason, llm_score?, original_score?,
rerank_reason?, signals?: {energy, replay}, kit?: {title, description, hashtags}, manual?, edited?}`
`id` is the list index and **never changes** (manual clips are appended); re-analysis creates a new project list
and detaches already-rendered clips (`clips[].candidate = null`).

### `project.json` (schema 2, `local/project.py`)
`{schema, id, title, created, updated, duration, source{kind,id,url,title,duration,language,embeddable,heatmap,chapters,local_path},
options{...}, transcript_info{source,language,segments}, candidates[], clips[]}` where a clip is
`{candidate, file, title, start, end, layout, captions, caption_style, title_card, watermark, aspect_ratio, quality, error, created}`.
Schema-1 projects (embedded transcript, `source_path`) are upgraded in memory on load.

### Upload records (`local/uploads.py`)
`clips_<id>/uploads.json` = `{<clip file>: {state: queued|uploading|uploaded|failed, privacy, publish_at (UTC text|null), category,
queued, attempts, error, retry_after, title, video_id, url, studio_url, actual_privacy, notice, uploaded, previous?}}`.
It is separate from `project.json` on purpose: re-rendering a clip or searching the video again rewrites `clips[]`, and that must
never make the app forget that a clip is already on the channel (-> no accidental second upload). `previous` holds the first
upload's record while a deliberate re-upload is queued, so cancelling it restores "uploaded".

### Where things live (under `LOCAL_OUTPUT_DIR`, default `output/`)
```
clips_<video id>/project.json(.bak), short_NN_<title>.mp4  one folder per video; NN = candidate id + 1 (re-render replaces)
clips_<video id>/uploads.json                            what was / will be uploaded to YouTube, per clip file
state/youtube.json                                       refresh + access token, channel, day counters, parked_until (a SECRET file)
cache/<video id>/transcript.json, energy.json, sections/ transcript, loudness series, temporary clip ranges
.llm_cache/<sha256>.json                                 every parseable LLM answer, keyed by provider + model setting + prompt
state/jobs.json                                          the job history (last 40), rewritten when a job changes state
logs/studio.log (+ .1 .2 .3)                             rotating log of everything the app and its libraries say
```
`project.json` is written through a temp file, the previous good copy is kept as `project.json.bak`, and `load_project`
falls back to it if the main file is damaged (power cut mid-write).

### Progress and jobs
`control.report(stage, fraction|None, note)`; stages: `source, transcript, audio, signals, rank, fetch, render, kit, upload`.
`control.scale(base, span)` maps a step's 0..1 into its slice of the whole job (clip 2 of 3 = 1/3..2/3).
`control.check()` raises `Cancelled` (a `BaseException`, so no best-effort `except Exception` can swallow it).

A **job** (`app.Job`, also `state/jobs.json`): `{attempt, max_attempts, retry_at, id, kind: analyze|render|kit|upload|update, label, project_id, status:
queued|running|done|error|cancelled|interrupted, request: {path, body}, stage, stages[], fraction, note, log[], result,
error, created, started, finished}`. `request` is the exact HTTP request that created the job, so **Try again re-sends it
verbatim** (all finished work is cached, so it continues). On startup a job saved as queued/running becomes `interrupted`
(the app was closed). An identical request while one is queued/running returns that job (no double clicks); more than
`MAX_WAITING_JOBS` (25) waiting -> HTTP 429.

### Health report (`local/health.py`)
`doctor(deep=False)` -> `{ok, version, checks: [{id, label, status: ok|warn|fail, detail, fix?}]}`. Check ids: `python,
ffmpeg, ffprobe, libass, opencv, ytdlp, llm_key, youtube_channel, disk, writable, whisper, network_cfg` (+ `youtube, llm_live` when deep).
`youtube_channel` never names the channel (health.json goes into diagnostics zips). With ffmpeg present there are also
`encoder` (which H.264 encoder renders use) and `faces` (YuNet model present). A `fail`
means the tool cannot work; `warn` means it works but you should look (old yt-dlp, low disk, no key).

### HTTP API (all JSON; writes need header `X-Studio: 1`)
`GET /api/config|settings|models|storage|health[?deep=1]|projects|projects/<id>|projects/<id>/signals|jobs|jobs/<id>`
`POST /api/settings|settings/test|storage/clean|ytdlp/update|analyze|projects/<id>/{reanalyze,candidates,render,kit,open-folder}|jobs/<id>/cancel`
`GET /api/youtube` (status + `queued`, never a secret) · `POST /api/youtube/{credentials,check,connect,pause,disconnect}` · `POST /api/youtube/queue/clear` · `GET /api/uploads` (every record of every video, newest first)
`GET /?state=..&code=..` (and the alias `/oauth/youtube/callback`) completes the sign-in
`POST /api/projects/<id>/upload` (enqueue clips + start a job; optional `meta: {file: {title, description, hashtags}}` = the review step) · `.../upload/run` (drain the queue) · `.../upload/dequeue`
`POST /api/analyze` also takes `auto_render`, `auto_min_score` and `upload{privacy,start_in_hours,every_hours}` = **Autopilot**.
`DELETE /api/projects/<id>` · `DELETE /api/projects/<id>/clips/<file>`
`GET /media/<id>/clip/<file>` (Range supported) · `GET /media/<id>/source` (local-file projects only)
`GET /api/projects/<id>/export.zip` (clips + `posts.txt` + `clips.csv`) · `GET /api/diagnostics.zip` (health, settings without
secrets, versions, job history, the log files — attach it to a bug report)

## Subsystems in more detail

### Streaming (`local/source.py`)
* `resolve()` costs one metadata request (~5-7 s). Live streams, playlists, private/age/bot-check cases raise `SourceError`
  with a message a person can act on (`YTDLP_COOKIES_BROWSER` unlocks login-only videos).
* **Clip ranges** use yt-dlp `download_ranges` (ffmpeg reads only the needed bytes). `SectionFetcher` runs two at a time
  while the previous clip renders.
* **H.264 only.** OpenCV decodes H.264 1080p at ~92 fps but AV1 1080p at ~10 fps, so the format selector prefers
  `vcodec:h264`, and anything else is converted once (`_fast_decode`).
* The section arrives already trimmed and audio/video aligned (`start_time` 0 for both), so there is **no second cut**.
* **Throttling facts (measured):** a continuous ffmpeg read of a googlevideo URL is throttled to about real time
  (2 h of audio: 35+ minutes). yt-dlp's own chunked downloader fetched the same 58 MB in 49 s. Hence the audio track is
  downloaded to a temp file, analysed, and deleted; it is never "streamed through ffmpeg". Section reads by ffmpeg run at
  roughly 1x real time (28 s of 1080p in ~26 s) which is why they are prefetched in parallel.
* Re-feeding an old `info` dict to yt-dlp's plain HTTP downloader gives HTTP 403, so `fetch_audio_file` extracts afresh;
  the section fetch (ffmpeg downloader) is fine with a stored `info`.

### Transcript (`local/transcript.py`)
Hand-made subtitles in the video's language > automatic original-language captions (`<lang>-orig`) > other tracks. An
English translation of a Hindi video must not win (its timing does not match the speech). Captions with fewer than 5
segments or <5% speech coverage count as "no captions" and fall back to Whisper (optional install).

### Ranking (`highlights.py`)
1. `detect_content_type` (1 LLM call). 2. Videos over 30 min are cut into 20-min chunks with 60 s overlap; each chunk's
timestamps are made **chunk-relative** (otherwise every chunk after the first loses its highlights to the length clamp).
3. Per-chunk candidates are deduped. 4. **One global re-rank**: chunk scores are not comparable (every chunk rates its own
best moments ~90). 5. **Signals** (`energy`, `replay`) are shown to the LLM *and* blended into the score
(`signal_weights`: LLM keeps >= 40%), because LLMs under-use hints and LLM score vs voice energy correlated -0.16 on real
data (text-only picks were quiet game dialogue). 6. `snap_to_sentences`, then `enforce_duration` to the requested range.
`Options.focus` steers both prompts; `Options.pending` lets a slow signal (audio) finish in the background.

### LLM layer (`local/llm.py`)
Plain `requests`, no SDK. Gemini `auto` model = discovered Flash models, newest full-size first. On `overloaded`: retry
twice, park the model 20 s; on `quota`: park for the server's `retry in` (<= 12 h) because quota is **per model**; on
`notfound`: skip; on `auth`: stop. If every model is busy, wait 20 s / 40 s and run the chain again (3 rounds); if only quota
is left, fail at once with that message. In `auto` mode the last model that worked is tried first. Answers are cached only
when they parse as JSON. Cache key = provider + *configured* model + prompt.

### Roman subtitles (`romanize.py`)
Setting `LOCAL_CAPTION_SCRIPT` (`roman` default, `original`), read live through `config.caption_script()`; `GET/POST /api/settings`
field `caption_script`, select `#capScript` on the Create page. `prepare(texts, llm_fn)` sends only words missing from the built-in
list and from `cache/roman_words.json`, within `LLM_SECONDS`; it never raises (cancel excepted). `segments()` returns copies with
one word per original word and leaves bracketed sound notes alone, because `split_annotations` reads laughter / applause from them.
Text typed by the person in the upload review is never rewritten.

### Automatic editing (`editing.py`, `captions.py`, `overlays.py`, `sfx.py`)
Two timelines: *source* (the fetched section) and *output* (after jump cuts). Cut points are rounded to the output frame
grid, and the sound is cut with `atrim`/`concat` at exactly the same points (15 ms fades), so A/V can never drift.
* **Pauses** come from a 20 ms loudness curve of the section's own audio (not from caption times: a caption word's "end"
  is the next word's start, so it covers the pause). Only the spoken part of each word is protected; nothing is cut when
  the sound never goes quiet (music, loud game); at most 35 % of a clip.
* **Zooms**: instant jump zooms alternate at joins and sentence starts; punches ease in on the loudest moments. Group
  shots take only 35 % of a zoom so nobody at the edge is cut off.
* **Popups** (priority): the AI's `{at, emoji, text}` -> the captions' own `[laughter]`/`[applause]` notes -> keyword
  emoji (English / Hinglish / Devanagari) -> loudness peaks. At least 2.4 s apart, never over the hook or end card.
* **AI notes** are asked together with title / description / hashtags in one call (`publish.make_publish_kit(with_edit)`),
  stored on the candidate (`kit.popups[].t` = video time, so a later trim cannot shift them), under a 75 s
  `llm.time_budget`; if the AI is busy the edit is made from the words and the loudness.
* Presets: `off`, `clean` (jump cuts, gentle zooms, captions, hook, progress), `viral` (everything). Each effect can be
  overridden (`RenderOptions.effects`).

### Memes (`memes.py`, `editing.py`, `clipper.py`)
* **Catalog** `assets/meme_catalog.json`, by lower-case file name: `id`, `use` (shown to the AI), `mood`, `how`, `start`/`end`,
  `auto`. Unlisted files: mood from folder / name, video = `cut` (green screen = `over`), the id from the name.
* **how**: `cut` (video waits, full screen), `over` (sound / sticker / keyed clip on top), `end` (closes the clip), `later`
  (where >= 3 s were cut out in one go). `auto: false` is never picked.
* **Choosing**: `publish.MEME_RULES` lists `id: use` in the edit-notes request; the answer's unknown ids are dropped and kept as
  `kit.memes[] = {t (video time), id}`. A kit with popups but no `memes` key is asked again once. Fallback (`meme_moments`):
  popup moods only for kits that were never asked, caption laughter / applause (may stop the video), keywords and loud peaks
  (sounds only). Limits in `memes.plan`: 2-4 memes, 1-3 stops, stops <= max(4 s, 20 % of the clip, at most 8 s), 6 s apart, and the clip with its stops stays <= `config.SHORT_SECONDS`.
* **Under a minute** (`config.SHORT_SECONDS` = 59): `_validate_analyze` caps the moment length; `render_local` ends a longer
  portrait / square range after the last sentence that fits (`pipeline._short_end`); memes only get the room that is left. A
  Short over a minute with any Content ID claim is blocked worldwide. Wide renders are left alone.
* **Timeline**: `editing.place_memes` snaps each stop to the end of a sentence (`phrase_end`, then the quietest 20 ms) and
  builds `Timeline(keep, inserts)`. Everything in output time (captions, zooms, sounds, progress bar) is computed after that.
  A word that ends "at" the next word's start is clamped to the stop, so no caption sits on a meme.
* **Render**: `_pump` emits the meme frames when the source reaches the stop (the source is not advanced); the sound graph
  concatenates `anullsrc` of the same length between the speech pieces and the meme's sound comes with the effects track.
  An unreadable meme holds the last picture for the same time, so sound and picture cannot drift.

### Rendering (`local/clipper.py`)
* `_analyse`: samples ~4 fps; finds letterbox bars (lines black in >=90% of frames), YuNet faces, and a temporal motion map.
* **auto** = **split** for a static corner facecam, else **smart** (`framing.py`): camera cuts from 96x54 thumbnails of
  every frame (ffmpeg, in parallel with face detection); per shot, face tracks -> performers (edge bystanders and people
  who barely move are dropped in crowds) -> `crop` (one person or a tight group, steady; slow follow only if they walk),
  `stack` (two large faces far apart), `group` (just the stretch of stage with the performers, over a blurred background),
  or `fit`. Framing never slides across a cut.
* `_plan` (used for `auto`'s split test and the explicit layouts): a face that stays put in a **corner** for the whole clip =
  facecam -> **split**; a large/central face in >= 50% of samples -> **track**; otherwise **fit**. Game characters' faces
  wander, so they never form a static cluster and never steer the crop.
* The facecam **rectangle** is found from the motion map (the overlay is calm next to the game picture) via the sharpest
  outward step in the row/column motion profiles; the panel takes the overlay's own shape. Face-size guess is the fallback.
* `track`: median filter + dead-zone follower + speed limit, per source frame, sub-pixel crop (`warpAffine`). Follows the
  cluster face, or the biggest face where the camera cut to someone else.
* Never stretches: every layout crops or letterboxes with one scale factor.
* Engine: the reader decodes in order, 2-4 threads compose frames (`cv2.setNumThreads(1)` meanwhile), results are written
  to ffmpeg in order as **yuv420p** (half the bytes of BGR, and ffmpeg need not convert). Encoder from `encoder.choose()`
  (Quick Sync on the reference laptop); a hardware failure is retried once on x264 and remembered for the process.
  `-filter_complex` does colour boost, `ass` (scratch dir, relative names), the sound cut, SFX mix and `loudnorm` -14 LUFS.
  Error or cancel kills ffmpeg and deletes the half-written file.
* Measured on the reference laptop (i5-8350U, UHD 620): a 25 s 720p stage section with the full viral edit renders in
  13-18 s (before: 15 s of video took 25 s without any editing).

### Green-screen streamer layout (`cutout.py`)
`clipper._plan_framing` asks it first for portrait `auto` (or `cutout`, which skips the evidence test). A face of 6-26 % of the
height, low in the picture, in >= 60 % of the clip, grouped into position runs; per run 24 frames give the per-pixel variation:
person mask (PP-HumanSeg) plus the still pixels around it = what is laid over the game. It is a cut-out when the game changes
right up to that shape, the shape is not a filled rectangle (a webcam box is) and the person reaches the bottom edge.
Compose: 43 % panel on a colour with the person only (largest component; a half chair looks cheap), game cropped to 4:3 below
with the overlay filled from its surroundings (mirroring made fake second cars), blurred strip at the bottom. About 2x
realtime for a 60 fps 1080p source on the user's 4-core laptop; segmentation is ~27 ms per frame per thread.

### Running as an app, and keeping the computer light
`launcher.pyw` (stdlib only) starts `app.py --no-browser` with `python.exe` and CREATE_NO_WINDOW so child ffmpeg processes stay
invisible; `--background` holds a named mutex and restarts a crashed server (gives up after 5 in 10 minutes); `/api/quit` writes
`output/state/quit.flag`, which stops the keeper. `housekeeping.sweep()` deletes a posted clip only if the file is not newer than
the upload (a re-render stays), marks it `removed_after_upload` + `youtube_url` in the project, and never touches projects,
upload records, sign-ins or memes. Cleared captions are re-fetched by `pipeline._transcript_for`.

### Web UI (`app.py`, `webui/`)
The page is a small single-page app with hash routes: `#/create` (home: "Get started" card, the form, recent videos), `#/projects`, `#/project/<id>` (tabs *Choose moments* / *Clips & upload*; `/?project=<id>` still works), `#/youtube` (status + pause, the five-step wizard, upload settings, the upload log). Alerts (AI key, expired sign-in, paused, Testing-mode sign-in, "YouTube asked to wait") and the job cards are global, above the routed view. Moments are drawn 24 at a time (`PAGE`), the timeline expands the list when a hidden moment is clicked. Settings is a dialog with tabs (AI, Logins & network, Health, Disk); the YouTube preferences live on the YouTube page.
One worker thread = one job at a time. `print()` output is captured per job (live log). Settings write `.env` through
`config.save_settings` (whitelisted names, regex-validated values; keys, proxy and the cookies path rules are enforced
server-side; secrets are write-only). The page keeps only per-browser conveniences in `localStorage` (theme, last project,
the options you chose, dismissed job cards); everything that matters lives on the server, so a refresh or a restart loses
nothing. A queued/running job is re-attached after a refresh; an *interrupted* one (app was closed) is shown with Try again.
Security: 127.0.0.1 only, `Host` check, `X-Studio: 1` on every write, a Content-Security-Policy on the page (YouTube embed
and thumbnails allowed, no inline script, no framing), `Cache-Control: no-store`. All text is inserted with
`textContent`/`createElement`; build node lists with `.filter(Boolean)` (a `null` child renders as the word "null").
Layout rule: grid/flex children that hold long text need `min-width: 0` (a long path once pushed Settings off a phone screen).

### YouTube uploads (`local/youtube.py`, `local/uploads.py`, `app.py`)
* **Sign-in.** Desktop-app OAuth client (id + secret in `.env`), authorization code + PKCE (S256), `access_type=offline&prompt=consent`
  so a refresh token is always returned, redirect to `http://127.0.0.1:<port>/` on this very server (the form Google documents for Desktop clients; `GET /?state=..&code=..` is handled as the callback, and `/oauth/youtube/callback` is an alias). The
  `state` is random, single-use, expires after 10 minutes (a forged or replayed callback is refused). Scopes: `youtube.upload`
  plus `youtube.readonly` (only to show which channel is connected, which matters with brand accounts). Google lets people untick
  scopes, so a grant without the upload scope is detected and explained; a Google account without a channel is refused and revoked.
* **Tokens.** `access_token()` refreshes when it has under a minute left and on a 401, under one lock. `invalid_grant` (revoked,
  expired, or the consent screen still in *Testing*, which expires refresh tokens after 7 days) sets `needs_reconnect`, raises
  `ReconnectNeeded` and stops calling Google; queued clips stay queued. Changing the client id forces a new sign-in.
* **The upload** is the resumable protocol, chunks of 8 MiB (a multiple of 256 KiB). A 5xx or a dropped connection: back-off, then
  ask Google how much it has (`Content-Range: bytes */TOTAL` -> `308` + `Range`) and continue from there, never from zero. A `404`
  (expired session) starts a new session (at most twice). Eight consecutive failures, or six "continue" answers that store nothing,
  end the attempt with `NetworkError` (the clip stays queued, retried in 5 minutes).
* **Errors -> actions** (`_classify`): `quotaExceeded`/`dailyLimitExceeded` -> park until the next midnight Pacific (a hand-written
  DST rule, because tzdata is not always installed on Windows); `uploadLimitExceeded` -> park 3 h (the channel limit is a rolling 24 h; `POST /api/youtube/resume` = "Try now" unparks); `rateLimitExceeded` -> 15 min;
  `youtubeSignupRequired` (it arrives as a 401, so it is tested before the generic 401) -> that clip fails; 401 /
  `insufficientPermissions` -> reconnect; other 4xx -> that clip fails with Google's reason, the others continue; 5xx -> network.
* **The queue.** `enqueue` (request thread) writes `queued` records; `drain` (job worker) takes them one at a time: it first writes
  the publish kit for clips that have none (one AI call; a failure falls back to the working titles), re-checks parking and the
  daily cap before every clip, marks `uploading`, uploads, records the video id / link / the privacy YouTube actually applied (a
  mismatch with the request is shown as a notice), counts the upload. A leftover `uploading` (crash) is revived as `queued`.
  Scheduled clips (`start`, `every_hours`) go up private with `publishAt`; if the wait made the time pass, it is moved to
  "10 minutes from now" instead of being refused.
* **The watcher** (`app._upload_watcher`, every 60 s, only in the real server) starts a drain job for projects with due queued clips
  when YouTube is connected, not parked, under the cap and `YOUTUBE_AUTO_RESUME` is on. The job queue's duplicate guard makes repeated
  ticks harmless. It only runs while the app is open (scheduled-for-later videos do not need it: YouTube publishes those).
* **Autopilot** = the existing `/api/analyze` job with `auto_render` + `upload`: analyse (0-40 %), render the top N above the minimum
  score with the publish kit (40-85 %), enqueue + drain (85-100 %). Moments already on the channel or queued (`uploads.claimed_ranges`, >50 % overlap) are skipped, so running Autopilot again on a video moves on to new moments; if none is left the job says so.
* **Pause** (`state/youtube.json: paused`) stops `drain` and the watcher; queued clips stay queued. **Review** (`meta` overrides) is stored in the record and wins over the AI kit in `_metadata_for`.
* **Key check** (`youtube.check_client`): a deliberately bogus code is sent to the token endpoint; `invalid_grant` = the client is fine, `invalid_client` = mistyped secret / unknown id, `redirect_uri_mismatch` = not a Desktop client. **Testing-mode detection:** Google adds `refresh_token_expires_in` to the token answer only while the consent screen is in "Testing"; it is stored as `refresh_expires_at`, surfaced as `testing_mode` / `token_days_left` (an alert), and once it passes the sign-in is reported as expired before Google refuses it. `start_in_hours` is turned into an absolute time when the clips
  are queued (after the long analysis), not when the button was pressed.
* **Safe defaults:** visibility `private`, `notifySubscribers=false`, `selfDeclaredMadeForKids=false`, 10 uploads/day, "public"
  in Autopilot asks for a confirmation in the page.
* **Secrets:** client secret in `.env` (write-only), tokens only in `state/youtube.json`; `logsetup.redact` knows `ya29.`, `1//`,
  `GOCSPX-` and the OAuth `code=`; the diagnostics zip contains neither the file nor those strings. The callback page is a static
  string with every dynamic part HTML-escaped and its own `default-src 'none'` CSP.
* **Known gap:** if the app dies after YouTube has accepted a video but before `uploads.json` records it, the clip is uploaded
  again on the next run (a duplicate, never a loss). The window is a few milliseconds.

### Self-healing jobs (`app._worker`)
A busy or unreachable AI service (`LLMError` kind `overloaded` / `network`) was the commonest real failure (Gemini 503s, 180 s read timeouts, a dropped connection: one 11-hour video needed four manual retries). `_retry_delay` puts such a job back in the queue after 1, 3, 7, 15 minutes (`RETRY_DELAYS`; a `threading.Timer` re-queues it; status stays `queued` with `retry_at`, so Cancel and the duplicate guard keep working). Only while redoing is cheap: `kit` jobs, and `analyze` jobs still in `source/transcript/audio/signals/rank` (everything finished there is cached). After rendering began a retry would re-render for minutes, so the error is shown instead. Wrong keys, quota and other errors are never retried.

### Ranking safeguards (`highlights.py`, `pipeline._build_project`)
* **"Moments to find" is a promise.** An 11.6-hour stream used to come back with 280 "moments" (every chunk's candidates are kept through ranking); `_build_project` now keeps the best `opts.candidates` (3..40).
* **One broken answer no longer degrades the ranking.** If the global re-rank answer is not valid JSON (an unescaped quote in a Hinglish reason broke it), `salvage_ranking` still pulls every `"id"` + `"score"` out of it; the prompt asks for short reasons without double quotes; and an `LLMError` (service down) now propagates instead of silently settling for per-chunk scores that are not comparable across chunks (the job-level retry above handles it, and the chunk answers are cached).

### Concurrency (why `update_project` exists)
Two threads write the same files: the job worker and the web requests. Findings that shaped the code:
* A render job holds its copy of `project` for minutes. Saving that stale copy at the end silently undid a clip the person
  added or deleted meanwhile. Fix: long-running writers call `update_project(project, mutate)`, which re-reads the saved
  file under the project lock and changes only what that caller owns (`render_local` -> its clip records and trims;
  `publish_kit_local` -> `kit`; the add-clip / delete-clip routes -> their own field). `reanalyze_local` builds a new candidate
  list on purpose and keeps rendered clips detached (`_kept_clips`).
* On Windows `os.replace` onto a file another thread is writing or reading can raise `PermissionError` (WinError 32).
  `save_project` and `_save_jobs` serialise writers (a lock each) and retry the replace a few times.

## Testing
`python -m unittest discover -s tests -t .` (about 130 s, offline, ~310 tests; `test_light`: cut-out layout, green-screen memes,
clean-up, the app launcher, `test_memes`: full-screen memes incl. a real render, `test_romanize`: Roman subtitles). `test_editing` (39: pause cutting,
timeline, zoom / popup planning, caption notes, overlays, SFX, shot detection and per-shot framing, encoder fallback, AI
edit notes, quality check, real edited renders, the pipeline with a fake AI) was mutation-checked with 16 bugs. Covers: caption/transcript parsing, ranking logic
(re-rank, blending, snapping, duration), LLM error handling + fallback + cache, source/caption-track choice, project store
and schema upgrade, storage, settings validation, every API guard/validation path, job lifecycle + cancel + persistence +
interrupted/retry + duplicate guard, doctor/health, diagnostics without secrets, export zip, delete clip, the update-while-
rendering race, concurrent saves, layout planning, and **real ffmpeg renders** of a synthetic video (every overlay, and
cancel leaving no file). Files: `test_transcript_captions`, `test_highlights`, `test_llm_source_project`, `test_clipper`,
`test_app` (web layer + jobs), `test_ops` (health, safety, history, extras), `test_polish` (31: salvaged re-rank answers, the candidate cap, job auto-retry, key check, Testing-mode detection, pause, review-before-upload, upload history, Autopilot never repeating a moment), `test_youtube` (76 tests: OAuth + PKCE, token refresh,
the resumable upload incl. resume / expired session / 401 / stalls, quota and cap, queue, scheduling, autopilot with faked analysis
and rendering, the watcher, secrets never leaving the server) against `tests/fake_google.py`.
The YouTube tests were also *mutation-checked* (round 1: 16 deliberate bugs; round 2 for `test_polish`: 16 more, e.g. candidate cap removed, re-rank not salvaged, retry after rendering, typed title ignored, pause ignored, Autopilot repeating moments, Testing mode not detected): 16 deliberate bugs (resume offset ignored, no PKCE verifier, state not checked,
secret in the status, scheduled video not private, double upload allowed, daily cap or parking ignored, HTML not escaped ...)
were put into the code one at a time and every one made a test fail.

**The suite never uses your real keys:** `tests/__init__.py` sets `GEMINI_API_KEY`, `OPENAI_API_KEY`, `YTDLP_PROXY`,
`YTDLP_COOKIES_FILE` to empty *before* anything imports `config` (an existing variable beats `.env`) and points
`LOCAL_OUTPUT_DIR` at a temp folder. (A "deep" health check once reached the live Gemini API from a test and spent real
quota and minutes; a test asserts this cannot happen again.) Any new test that would touch the network must mock it.

The browser UI is driven in headless Edge over DevTools against an in-process server and the fake Google (`cdp_ui2_test`: ~90 checks: routes, the Get-started card, alerts, the wizard and key check, connect through the redirect, pause, preferences, history, paging, review dialog, upload / upload again / quota wait, Autopilot, disconnect, no "null" text on any page, no overflow at 360 px on every page and every Settings tab). Those scripts are scratch files, not part of the repository.

Not covered by automated tests: **real Google** (OAuth consent, a real upload: needs a person's account), live YouTube/Gemini calls (flaky by nature) and the browser UI, which was checked with
DevTools-protocol scripts (headless Edge: render, trim, timeline, YouTube embed under the CSP, settings/health, zip export,
interrupted job card, remembered options, delete clip, no JS errors, no horizontal overflow at 360-1280 px). Those scripts
were scratch files and are not part of the repository.

## Extension points
* **New layout:** add a branch in `_plan` (return a dict with `layout`, geometry) and a branch in `_render`; add its name to `LAYOUTS`.
* **New ranking signal:** produce a `pct(start, end)` function (see `signals.window_pct_fn`), add an entry to `SIGNAL_INFO`
  in `highlights.py` (label, prompt note, weight), pass it in `Options.signals` or `Options.pending` from `pipeline._build_project`.
* **New LLM provider:** add `call_<name>_llm` in `llm.py`, branch in `call_local_llm`, add its settings to `config._VALIDATORS`.
* **New caption style:** add to `STYLES` and the style table in `captions.build_ass`.
* **Another upload target (TikTok, Instagram...):** copy the shape of `local/youtube.py` (auth + `upload_video` raising the same four
  error classes) and `uploads.py` (queue file per target); `pipeline.upload_local` and the watcher need only a target switch.
* **New site:** anything yt-dlp supports already works as kind `web`; captions need a `vtt`/`json3` track, otherwise Whisper.

## Decisions and measured facts worth remembering
* AV1 decodes ~9x slower than H.264 in OpenCV (10 vs 92 fps at 1080p) -> H.264 only.
* x264 intermediate re-encodes of 1080p cost more than real time on a laptop CPU -> sections are used as delivered.
* OpenCV 5 removed `CascadeClassifier`; `av` 17+ broke faster-whisper -> pinned (`opencv-python-headless<5`, `av<17`).
* Whisper `base` forced to the wrong language produced script-mixed garbage that poisoned ranking and captions; this is
  why captions are the default transcript source.
* Timings on the reference machine (4-core laptop, 2-hour video): metadata 7 s, captions 3 s, audio 50 s + 50 s analysis
  (parallel with the LLM), 10 s of 1080p clip: fetch ~13 s, analyse 8 s, render 28 s.
* Benchmarks on this machine are noisy (+-40% run to run); do not trust small speed differences.
* yt-dlp is updated *in the running process* (`health.update_ytdlp`). It monkey-patches urllib3's `_PERCENT_RE`; purging and
  re-importing it without restoring urllib3 first breaks every `requests` call (`'Urllib3PercentREOverride' object has no
  attribute 'sub'`). `_unpatch_urllib3()` runs before the modules are purged.
* `setup.bat` was verified from an empty folder (fresh `venv`, ~80 s, then the whole suite) on Python 3.14; `cmd` can refuse
  to run `setup.bat` by bare name from the current directory on some systems, so scripts call each other as `"%~dp0name.bat"`.
* An 11.6-hour stream (9,949 caption segments, ~24 chunks) was analysed end to end on this laptop; its failures were all the AI service (503, read timeouts, one DNS outage), none the pipeline.
* **Google's own documents disagree about YouTube upload limits** (one page: `videos.insert` costs 1 unit in its own bucket of 100
  calls/day and unaudited projects are *not* forced private; older posts and the quota calculator: 1600 units of 10,000/day and
  private-only until audited). The code assumes neither: the daily cap is a setting, `quotaExceeded` parks the queue, and a video
  that YouTube makes private against the request is reported on its card.
* An OAuth consent screen in *Testing* expires refresh tokens after 7 days (Google's OAuth documentation); the setup guide says to
  publish it ("In production"). Unverified production apps show a warning and are capped at 100 users: fine for one's own channel.
* Not done on purpose: CI workflow (the project is not in a git repository here), Docker, a pip package (the app serves its
  own `webui/` folder from the checkout).

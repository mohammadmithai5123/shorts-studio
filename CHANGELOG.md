# Changelog

## 2.7.0

- **Waiting uploads say why** and when the next try is, and have a **Try now** button. After YouTube's channel upload limit the
  app now tries again every 3 hours (before: 12 hours, with only "waiting" on the page).
- **Titles that make people stop**: a little story naming whose video it is ("... ke peeche lag gayi police"), same for the
  headline on the clip. Saved text from older versions is rewritten when a clip is rendered again.
- **Your channel's name on every clip** (badge, top left; "Your branding" on the Create page changes it, "-" removes it) and as
  the first hashtag. The "Follow for more" card is off unless ticked.

## 2.6.1

- **Every Short stays under a minute** (`SHORT_SECONDS = 59`): moments are picked up to 59 s, a longer clip (an old moment, a
  manual trim) is cut after the last sentence that fits, and full-screen memes are only added while there is room. YouTube
  blocks a Short over a minute everywhere when it has any Content ID claim. Wide (16:9) renders keep their length.

## 2.6.0

**Memes that take over the screen, chosen by what is said.**

- A video meme now **stops the video**: it fills the screen with its own sound (blurred copy behind), then the clip carries on.
  Sounds (`faah`) and green-screen clips still play over the video. A closing meme ("Directed by ...") ends the clip; "A few
  moments later" can stand where a long wait was cut.
- The **AI picks the meme** from a list of what each one is for, in the same request that writes titles. Without a pick,
  laughter / applause in the captions and loud moments choose, more cautiously.
- `assets/meme_catalog.json` describes the files of `Memes Assets/` (36 of them): purpose, mood, which seconds to play.
  Two are switched off (a politician; a joke about women). Files not listed still work by folder or name.
- Cuts for a meme are made after the sentence, never inside a word; captions, zooms and sounds follow the longer timeline.
- At most 3 stops (and about a fifth of the clip), 6 seconds apart, never in the first 3 seconds.
- Fixed: reading a project at the very moment it was being saved could fail on Windows ("Permission denied").

## 2.5.1

- Fixed: a long stream failed with "Highlight generator produced invalid output" when one of its parts had no moment worth a clip
  (or got a bad answer). Such a part is now skipped; the analysis only fails when most of the video cannot be read.
  Retries are new questions, so a saved bad answer is not replayed forever.

## 2.5.0

**Streamers on top, your memes, a light computer, and an app icon.**

### Added
- **Green-screen streamer layout** (automatic, or *Streamer on top* in the look options): when the person is cut out over a game
  with no box, they are drawn big on a plain colour (white by default) at the top and the game, three quarters of its width,
  below; the small copy of the person in the game picture is smoothed away. Person segmentation: PP-HumanSeg (OpenCV Zoo,
  Apache-2.0, 6 MB). Captions sit on the dividing line instead of over the face.
- **Your own memes** (`memes/<mood>/`): video (green screen keyed out), sounds and pictures, placed at moments chosen by the AI,
  laughter in the captions, words and loud reactions. Replaces the emoji stickers in *Viral* (they can still be ticked).
- **Automatic clean-up**: posted clips are deleted from the computer (history and the YouTube link stay), saved captions and AI
  answers go after 3 days, leftovers of interrupted work too.
- **Runs like an app**: `install_app.bat` adds a desktop and Start menu icon and starts it with Windows, with no console window;
  the keeper restarts it if it crashes; *Quit* button in Settings -> App.

### Fixed
- Clips rendered after the saved captions were cleared came out without subtitles: the captions are now fetched again.
- Rendering reads only the frames it uses (faster with 60 fps sources).

## 2.4.0

**Subtitles in Roman letters.**

- Hindi (Devanagari) and Urdu-script captions, headlines, pop-ups and upload titles / descriptions are written in Roman letters
  (Hinglish / Roman Urdu), with the same word timing. Letters change, the language does not. On by default; *Subtitle letters* on
  the Create page (or `LOCAL_CAPTION_SCRIPT=original`) keeps the spoken script. Titles you type yourself are never changed.
- Spelling comes from a built-in list of common words, then one time-limited AI request for unknown words (remembered in
  `output/cache/roman_words.json`), then spelling rules, so it works when the AI is busy or out of quota.
- Clips rendered earlier keep their old subtitles until they are rendered again.
- Fixed keyword-emoji false matches on Roman Hindi ("kya", "sadak", "darwaza").

## 2.3.0

**Reels that edit themselves, framed properly for shows, and rendered much faster.**

### Added
- **Automatic editing** (Viral / Clean / Off, each effect switchable): jump cuts of pauses (sound and picture cut on the same
  frame), punch-in zooms, emoji and word popups (planned by the AI together with the title in one call; laughter / applause from
  the captions, Hindi and English keywords and loud moments as backup), synthesised pop / whoosh / ding, an animated headline,
  word-by-word "pop" captions with key words highlighted, a progress bar, "Follow for more", a little colour boost. Autopilot uses it.
- **Per-shot framing for shows, podcasts and panels**: camera cuts are detected, and each shot is cropped to the speaker, stacked
  (two people), or narrowed to the performers (YuNet face detection; guards at the edges are left out).
- **Quality check** of every rendered clip (size, length, sound, not black) before it is kept or uploaded.
- `pillow` for the popups; the YuNet model in `assets/models`; `LOCAL_ENCODER`; `--edit` on the command line.

### Faster
- Hardware encoder (Intel Quick Sync on the test laptop) with automatic fallback, frames composed on several threads, yuv420p to
  ffmpeg: a fully edited 25 s clip in 13-18 s.
- The AI popup plan has a 75 s budget: a busy AI service never holds up a render.

### Fixed
- Captions printed YouTube's sound notes ("[हंसी]", "[Music]"). The page and the running app can be different versions after an
  update: the page now sends only what the running app understands and asks to restart it.


## 2.2.0

An easier, safer tool, driven by what went wrong in real use (an 11.6-hour GTA stream, Gemini overloaded for hours).

### Fixed (found in real use)
- **"Moments to find" was ignored on long videos:** 10 asked, **280** returned. It now keeps the best N (3 to 40).
- **One broken AI answer silently degraded the ranking.** The re-rank answer is now salvaged when it is damaged JSON, and a down AI service no longer falls back to scores that are not comparable.
- **Gemini 503s / timeouts / a DNS blip needed four manual "Try again"s.** Jobs now retry themselves (1, 3, 7, 15 minutes) while the work is cheap to redo, with a visible countdown.
- A stale "YouTube asked to wait" note, a literal "null" on the YouTube page, toasts hiding behind the action bar, finished job cards piling up, Settings forced open on the first visit and an AI call on every Settings save.

### Changed (the interface)
- **Navigation:** Create / Videos / YouTube pages in a top bar with routes (back button and refresh work), a "Get started" checklist, global alerts (AI key, expired sign-in, paused, "still in Testing", YouTube asked to wait), jobs visible on every page, progress in the tab title, finished jobs tidy themselves away.
- **Create:** two clear modes, **Let me choose** and **🚀 Autopilot**; fewer options up front.
- **A video:** two steps (*Choose moments* → *Clips & upload*), a bottom action bar, moments 24 at a time, "More look options" folded.
- **Settings** is tabbed (AI, Logins & network, Health, Disk) and no longer pops up by itself.

### Added (YouTube)
- A **guided five-step connection page**: a button per Google screen, steps ticked off, a **drop zone** for the downloaded JSON (or paste / type it), a **key check** with Google before the sign-in, detection of a consent screen still in **Testing** (the sign-in would expire in 7 days) with an alert and the fix, a clearer message for "Access blocked".
- **Review before upload:** edit any clip's title, description and hashtags.
- **Pause / Resume** switch (also from the alert), **upload history** across all videos (waiting / on YouTube with links / problems), today's usage meter, **Stop all waiting uploads**.
- **Autopilot never repeats a moment** that is already on the channel or queued.
- Moments to find: up to 40.
- 31 new tests (`tests/test_polish.py`), mutation-checked; a ~90-check browser script.


## 2.1.0

**Upload to your YouTube channel, automatically.**

### Added
- **Connect YouTube** (Settings → YouTube): OAuth 2.0 with PKCE through your own Google "Desktop app" client; a step-by-step guide,
  `client_secret.json` loading, the connected channel's name, Reconnect / Disconnect. No Google SDK is installed.
- **Upload to YouTube**: tick clips (or **Select all**) → visibility → now or scheduled (first at a time you pick, then one every N
  hours; YouTube publishes by itself). Titles, descriptions, hashtags and `#Shorts` come from the publish kit.
- **🚀 Autopilot**: paste a link and the best N clips (above a minimum score) are found, rendered, titled and uploaded, all at once or
  spread out. Private by default; public asks for confirmation.
- A **resumable, self-healing upload**: big chunks, back-off, continues from what Google already has, restarts an expired session,
  refreshes tokens, detects a revoked sign-in and a missing upload permission.
- An **upload queue** that survives restarts: never uploads a clip twice, respects your daily cap, parks itself when YouTube's quota
  or the channel limit is reached, continues by itself while the app is open (a watcher), *Remove from queue*, per-clip status with
  links to the video and to YouTube Studio, a notice when YouTube changes the visibility you asked for.
- Health check `youtube_channel`; secrets (`ya29.`, `1//`, `GOCSPX-`, the OAuth `code`) are redacted from logs and diagnostics.
- `tests/fake_google.py` (a fake Google with failure injection) and 76 tests for all of it, mutation-checked.

### Changed
- `POST /api/analyze` accepts up to 12 clips for auto-render, a minimum score, and `upload` options.

### Not verified
- Never run against real Google (needs a person's account): do one first upload as *Private* and check it in YouTube Studio.

## 2.0.0

The local mode was rebuilt around one idea: **never download the whole video.**

### Added
- **Shorts Studio**, a local web app (`python app.py` / `run_ui.bat`): paste several links, a timeline of the ranked
  moments, in-browser preview (YouTube player), trimming, your own clips, live progress with cancel and a live log,
  a remembered set of options, light/dark theme, works at phone width.
- **Streaming architecture:** metadata (~7 s), the video's own captions (~2 s, word timing) and an audio-only temporary file
  (~50 s for 2 hours, deleted after analysis). Each clip is fetched as its own time range (H.264, up to 1080p).
  The old whole-video download option is gone (a legacy 1.7 GB file can be deleted from Settings).
- **Better picks:** chunked ranking with one *global* re-rank, voice-energy and "most replayed" signals blended into the score,
  edges snapped to sentence boundaries, clip length enforced, a "what should I look for" focus.
- **Frames that look right:** facecam + gameplay split (the overlay rectangle is detected), a stable single-speaker follower,
  or the whole frame over a blurred background; letterbox bars removed; nothing is stretched.
- **Captions** (classic / yellow / boxed / karaoke; Latin, Devanagari, Arabic scripts), title card, watermark, loudness
  normalisation, publish kit (title, description, hashtags), zip export (`clips + posts.txt + clips.csv`).
- **Reliability:** every AI answer cached; Gemini model fallback with quota/overload handling; per-clip error isolation;
  jobs survive a page refresh *and an app restart* (shown as "interrupted" with an exact **Try again**); duplicate-request
  guard; atomic `project.json` with a `.bak` copy; low-disk preflight; one-writer-at-a-time saves.
- **Operations:** `--doctor` / Settings → Health (ffmpeg, libass, OpenCV, yt-dlp age, key, disk, network), one-click **yt-dlp
  update**, cookies file and proxy settings, rotating log file, diagnostics zip (never contains a secret), `--version`.
- `setup.bat`, `pyproject.toml` (lint config), unit tests (offline, ~95, including real ffmpeg renders).

### Changed
- Lite install: 16 packages / ~200 MB (no PyTorch, no AI SDKs). Whisper is optional and only a fallback.
- The local web server only answers on 127.0.0.1, checks the Host header, requires a custom header on writes, sends a
  Content-Security-Policy, and never returns a stored secret.

### Fixed (found while testing)
- Highlights after the first 20-minute chunk were lost (chunk timestamps were absolute).
- A Hindi video was transcribed from its English translation track.
- Crop followed only one face and ignored camera cuts.
- A project edited in the browser while a render was running was overwritten when the render finished.
- Job history could fail to save on Windows when two threads saved at once.
- The Settings window overflowed sideways on phone-width screens.
- The test-suite could spend a real Gemini quota (a "deep" health check hit the live API); the suite now never sees real keys.

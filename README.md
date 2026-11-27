# Shorts Studio

Paste a long YouTube video, get vertical Shorts: the app finds the best moments from the captions, cuts only those parts (the whole
video is never downloaded), edits them (framing, Roman-letter captions, memes, headline), writes titles and uploads them to your
YouTube channel. Runs on your own Windows PC.

Built on [AI-Youtube-Shorts-Generator](https://github.com/Anil-matcha/AI-Youtube-Shorts-Generator) by Anil Chandra Naidu Matcha
(MIT licence), heavily rewritten for local use. See `ARCHITECTURE.md` for how it works and `CHANGELOG.md` for what changed.

**Setting up:** run `setup.bat`, then `run_ui.bat` (or `install_app.bat` for a desktop icon). Copy `.env.example` to `.env`, or
enter your keys in *Settings* inside the app. Your meme clips are not part of the repository: put them in a `Memes Assets` folder.
Only post videos and clips you have the right to use.

## Features

**Find the best moments of a whole video, then cut them into vertical clips — without ever downloading the video.**

- **🪶 No giant downloads.** Local mode reads the video's *captions* (about 2 MB for a two-hour video, in ~2 seconds), ranks the moments, and fetches **only the clips you render**, as their own time ranges, at up to 1080p. A two-hour video used to mean a 1.7 GB, 30-minute download first.
- **🖥️ A real app, not just a CLI.** `python app.py` opens *Shorts Studio*: paste links (several at once), see the ranked moments on a **timeline**, **preview each one in the browser**, **trim** it (type times, nudge ±1 s, or use "Start = now" while the video plays), add your own clips, tick the ones you like, render.
- **🎯 Tell it what to look for.** One click — *Funny, Insights, Intense, Emotional, Tips* — or your own words ("every time he gets scared"). Choose the clip length (15–30 s, 20–59 s, 45–59 s). Every Short stays under a minute.
- **🧠 Picks the best from the whole video.** Chunk scores are not comparable, so one **global re-rank** compares every candidate. It also uses signals that don't depend on speech recognition: how **loud and excited the voice** gets, and YouTube's **"most replayed"** data when the video has it.
- **📐 Frames that look right, shot by shot.** A streamer's facecam on top and gameplay below (the facecam rectangle is detected from the footage). For shows, podcasts and panels every **camera shot** gets its own steady framing: close-up on one speaker, two people stacked, or just the stretch of stage with the performers (bystanders at the edges are left out, nobody important is cut off). Letterbox bars are removed. Nothing is ever stretched.
- **🎬 Automatic editing (new).** *Viral* (default): pauses cut out, punch-in zooms, emoji / word pop-ups at the funny bits (planned by the AI, with laughter from the captions as a backup), sound effects, a headline, animated word-by-word captions, a progress bar and "Follow for more". *Clean*: the calm version. *Off*: just the framing. Every effect can be switched on or off.
- **⚡ Fast.** Renders use the graphics chip (Intel Quick Sync / NVIDIA / AMD) when it works, with several threads composing frames: on a 2018 laptop a fully edited 25-second clip takes under 20 seconds.
- **🎮 Streamer on top.** Green-screen gaming streams (the person cut out over the game): they go big on a plain colour at the top, the game below.
- **😂 Your memes, not emoji.** Put clips and sounds in `Memes Assets/` (or `memes/laugh`, `memes/shock`, ...). The AI picks the one that fits what was just said; the video pauses, the meme fills the screen, the video carries on. `assets/meme_catalog.json` says what each file is for and which seconds to play.
- **🧹 Stays light.** Posted clips and old saved data are removed by themselves.
- **🖥️ Works like an app.** Run `install_app.bat` once: desktop icon, starts with Windows, no console window.
- **🔤 Roman subtitles (default).** Hindi and Urdu speech is written in Roman letters (Hinglish / Roman Urdu: *mujhe nahi pata*), with English words spelled as English; headlines and upload titles follow. Pick *As spoken* on the Create page for the original script.
- **✍️ Captions that pop.** Classic, yellow, boxed, or **karaoke** (words light up as they are said), in Latin, Devanagari and Arabic/Urdu script; an optional **title card** and **watermark**; audio normalised to −14 LUFS.
- **📋 A publish kit.** An AI-written title, description and hashtags per clip, one click to copy.
- **🚀 Straight to your YouTube channel.** A guided five-step page connects your channel once (it even checks your key with Google). Then tick clips and press **Upload to YouTube** (review or edit each title first), or choose **Autopilot**: paste a link and the best clips are found, rendered, titled and uploaded as Shorts with no further clicks (private first, public, or one per day on a schedule). Autopilot never repeats a moment that is already on your channel, and a big **Pause** switch stops every upload at once.
- **🧭 Easy to move around.** Three pages in the top bar (**Create**, **Videos**, **YouTube**), a "Get started" checklist on the first visit, alerts that say what needs attention (AI key, expired sign-in, paused uploads), a two-step video page (*Choose moments* then *Clips & upload*), and progress in the browser tab title.
- **🛟 Built to survive bad days.** AI answers are cached, so a crash, a 503 or a used-up daily quota never costs the same call twice; Gemini models fall back to each other automatically; **a job that fails because the AI service is busy or unreachable puts itself back in the queue (after 1, 3, 7, 15 minutes) instead of making you press Try again**; every long step can be **cancelled**.
- **⚙️ Easy setup.** Paste an API key in *Settings* and press **Test connection**. No `.env` editing needed.
- **🧰 Still a CLI and a Python library.** `python main.py URL --mode local --captions --focus "funny moments"`, or `from shorts_generator import generate_shorts`.
- **🔀 API mode** (MuAPI does download, transcription, ranking and cropping server-side) is still there for people who want zero local setup.

## Installation (Self-Hosted)

You need **Python 3.10+** and **ffmpeg** on your PATH. That's it for local mode — no PyTorch, no AI SDKs.

**Windows, the easy way:** double-click **`setup.bat`** (creates the `venv`, installs everything, checks ffmpeg and prints a health
report), then **`run_ui.bat`**. `run_ui.bat` runs the setup by itself the first time.

By hand (any system):

```bash
git clone https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator.git
cd AI-Youtube-Shorts-Generator
python -m venv venv
venv\Scripts\activate                       # macOS / Linux:  source venv/bin/activate
pip install -r requirements-local.txt       # api mode only needs requirements.txt
python main.py --doctor                     # checks ffmpeg, yt-dlp, the AI key, disk space ... and says how to fix what is missing
```

Optional: `pip install -r requirements-whisper.txt` adds speech recognition for the rare video that has **no captions at all**.

Then start the app. **Settings** opens by itself the first time so you can add a free [Gemini API key](https://aistudio.google.com/apikey):

```bash
python app.py        # Windows: double-click run_ui.bat
```

## Using Shorts Studio

The top bar has three pages: **Create** (start here), **Videos** (everything you made) and **YouTube** (connection, settings, upload history). ⚙ **Settings** holds the AI key, logins/proxy, health and disk.

1. **Paste a link** (or several, one per line), pick what to look for and the clip length, choose **Let me choose** or **🚀 Autopilot**, press the button. The first run on a long video takes a few minutes (captions, audio loudness, a handful of AI calls); the answers are saved, so running it again is fast.
2. **Review.** The timeline shows the loudness curve, the "most replayed" curve (when it exists) and where each suggested moment sits — click one to jump to it. Each card has a score, why it was picked, what is said, and a preview. Trim the times if you like.
3. **Render.** Tick the moments, choose the **Editing** style (✨ Viral, Clean or Off), the shape (9:16, 1:1, 4:5) and captions (single effects are under *More look options*), press **Render** in the bar at the bottom. Progress shows at the top of every page (and in the tab title); you land on the **Clips & upload** tab when it is done. A long video shows its moments 24 at a time (*Show more*); *Moments to find* (More options) goes up to 40.
4. **Publish.** Download the clips one by one, or **Download everything (zip)**: the clips plus `posts.txt` (title + description +
   hashtags per clip) and `clips.csv`. Copy a clip's title + hashtags with one click. A clip you don't like can be deleted.

5. **Upload (optional).** After the one-time setup below, tick clips under *Rendered clips* (or **Select all**) and press **Upload to YouTube**; or tick **🚀 Autopilot** next to the link box to do steps 1 to 5 in one go.

If anything fails, the card says why in plain words and offers **Try again** (it re-sends exactly the same request and continues
from what is already saved) and **Copy details** for a bug report. Closing the app mid-job is safe: the job is listed as
*interrupted* the next time, with the same button. **Settings → Health** checks your setup and can **update yt-dlp** in one click.

## Command line

```bash
# local mode: nothing is downloaded as a whole
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --num-clips 3

# with options
python main.py "URL" --mode local --num-clips 5 --focus "funny moments" --min-seconds 15 --max-seconds 45 \
    --captions --caption-style karaoke --title-card --watermark "@mychannel" --kit --quality 1080

# a file on your disk
python main.py "C:\Videos\input.mp4" --mode local

# api mode (MuAPI)
python main.py "URL" --mode api --num-clips 5 --output-json result.json
```

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `api` | `api` (MuAPI) or `local` (this machine, no whole-video download) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | `9:16` TikTok/Reels/Shorts, `1:1` square, `4:5`, anything `W:H` |
| `--language` | the video's own | Caption / speech language code (`en`, `hi`, ...) |
| `--focus` | — | What to look for, in your own words |
| `--min-seconds` / `--max-seconds` | `20` / `59` | Clip length range (enforced after the AI answers too; never over 59) |
| `--quality` | `1080` | Tallest source picture fetched per clip (`480`–`2160`) |
| `--layout` | `auto` | `auto` / `split` / `track` / `fit` |
| `--captions` · `--caption-style` | off · `classic` | `classic` / `yellow` / `box` / `karaoke` |
| `--title-card` · `--watermark TEXT` | off | The clip's title for its first seconds; small corner text |
| `--no-normalize` | off | Keep the original loudness (default: −14 LUFS) |
| `--kit` | off | Also write title, description and hashtags per clip |
| `--refresh` | off | Ignore the saved transcript and signals |
| `--clean` | — | Show disk use and delete old whole-video downloads from earlier versions, then exit |
| `--format` | `720` | **api mode only**: resolution MuAPI downloads at |
| `--output-json` | — | Write the full result to a file |

Clips go to `output/clips_<video id>/short_NN_<title>.mp4`, one folder per video, next to a `project.json` (open it again in the UI any time).

```python
from shorts_generator import generate_shorts

result = generate_shorts("https://youtu.be/VIDEO_ID", mode="local", num_clips=3, captions=True, focus="key insights")
for short in result["shorts"]:
    print(short["score"], short["title"], short["clip_url"])
```

## Upload to YouTube automatically

Shorts Studio can upload your clips to **your own channel** (YouTube Data API v3, no extra software). Titles, descriptions and
hashtags come from the AI-written publish kit, and every upload is also tagged `#Shorts`. YouTube files a video under Shorts when
it is square or vertical and at most 3 minutes long, which is what the 9:16 clips are.

**One-time setup, about 5 minutes (the **YouTube** page walks you through it, one button per step, and ticks the steps off):**

1. In the [Google Cloud Console](https://console.cloud.google.com/projectcreate) create a project (any name) and turn on the **YouTube Data API v3**.
2. Set up the OAuth *consent screen* (called "Google Auth platform" now): audience **External**, any app name, your email; add your own Google account as a *test user*, then **publish the app ("In production")**. While it says *Testing*, Google expires the sign-in every 7 days. (An unverified app in production is limited to 100 users and shows an "isn't verified" warning, which is fine for your own app.)
3. Create an **OAuth client of type "Desktop app"** and download its JSON.
4. On the YouTube page **drop that JSON** on the drop zone (or choose it, or paste its text, or type the id and secret). The app saves it and **checks with Google that the key is accepted** (a mistyped secret or a client of the wrong type is reported at once). Then press **Connect YouTube**, sign in, press *Advanced → Go to … (unsafe)* on the warning (it is your app), and **keep every permission box ticked**.

Google may rename these screens; the steps stay the same. The client secret and the sign-in token are stored only on this computer
(`.env` and `output/state/youtube.json`) and are never shown again, logged, or put in the diagnostics zip.

**Using it**

* **Manual:** *Clips & upload* tab → tick clips (or **Select all**) → **Upload to YouTube** → choose visibility and *now* or *schedule* (the first one at a time you pick, then one every N hours; YouTube publishes scheduled videos by itself, even with your computer off). Open *Review the titles and descriptions* to edit any title, description or hashtags before they go.
* **Autopilot:** tick **🚀 Autopilot** under the link box, choose how many clips, a minimum score, visibility and whether to publish all at once or one every 6/12/24 hours, press **Run autopilot**. It finds the best moments, renders them with your current settings, writes the titles and uploads. You can leave the page open and walk away.
* **Safe by default:** uploads are **private** until you choose otherwise (check them in YouTube Studio first). Choosing *public* in Autopilot asks you to confirm.
* **Pause:** the **Pause uploads** switch on the YouTube page (and **Resume** in the alert that appears on every page) holds everything: queued clips stay queued, Autopilot still prepares clips but uploads nothing.
* **Overview:** the YouTube page lists everything *waiting*, *on YouTube* (with links to the video and to YouTube Studio) and *problems*, across all your videos, with today's usage ("3 of 10 uploads"), *Remove* / *Try again* per clip and **Stop all waiting uploads**.
* **Nothing is uploaded twice:** a clip that is already on YouTube is skipped (use *Upload again* on purpose). Uploads that cannot go now (your own daily limit, YouTube's quota, no internet) wait in a queue and **continue by themselves** while the app is open; *Remove from queue* and *Stop waiting uploads* take them out. A big upload that loses its connection resumes where it stopped.
* **Limits to know about:** *Most uploads per day* (Settings, default 10) is your own cap. YouTube also limits API use per Google project (the daily quota resets at midnight Pacific time) and may limit how many videos a channel can upload per day; when YouTube says "not now" the queue pauses until it is worth trying again. Google's documentation is inconsistent about how many API uploads a new project may do and whether an unaudited project's uploads are forced to *private*; the app does not assume either: if YouTube makes a video private against your choice, the clip says so and links to Studio.
* **Only upload what you may:** your own videos, or content you have the rights to. Copyrighted music or footage can get a Short blocked or a channel struck; reposting other people's videos is against YouTube's policies.

**Not verified against real Google yet:** the whole flow is tested against a faithful fake of Google's OAuth and upload endpoints (`tests/fake_google.py`), but it could not be run with a real account here. Do a first run with one clip set to *Private* and look at it in YouTube Studio.

## How it works (local mode)

1. **Read the link** — metadata only (~5 s). Captions, loudness and "most replayed" are fetched in parallel.
2. **Transcript** — the video's own captions (hand-made first, otherwise YouTube's automatic ones in the video's language); Whisper only if there are none.
3. **Rank** — an AI proposes moments in each 20-minute chunk, then **one global re-rank** compares them all; voice energy and replay data are blended in; edges snap to sentence boundaries; length is enforced.
4. **Fetch + render** — for each chosen moment only its time range is fetched (H.264, up to your quality), then reframed, captioned and encoded to 1080×1920 H.264 at 30 fps.

The code map, data contracts, caching layout and the measurements behind the design are in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Configuration

Everything has a screen in **Settings**; the same values live in `.env` (see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | write-only in the UI; never shown again |
| `GEMINI_MODEL` | `auto` | `auto` picks the best available Flash model and falls back to the others when one is busy or out of quota |
| `YTDLP_COOKIES_BROWSER` | — | `chrome` / `edge` / `firefox` ... for age-restricted, members-only or "confirm you're not a bot" videos |
| `YTDLP_COOKIES_FILE` | — | a `cookies.txt` exported from your browser, when reading the browser directly does not work |
| `YTDLP_PROXY` | — | `http://user:pass@host:port`, if your network needs one (write-only in the UI) |
| `LOCAL_OUTPUT_DIR` | `output` | Where clips and caches go |
| `LLM_CACHE` | `1` | Re-use saved AI answers (`0` = always ask again) |
| `LOCAL_CAPTION_FONT` | Arial / Nirmala UI | Font family for burned-in text |
| `LOCAL_WHISPER_MODEL` · `_BEAM_SIZE` · `_THREADS` · `_DEVICE` | `base` · `5` · `0` · `auto` | Only used when a video has no captions |
| `LOCAL_FACECAM_BG` | `white` | Colour behind a green-screen streamer (Create page) |
| `LOCAL_DELETE_UPLOADED` · `LOCAL_CACHE_DAYS` | `1` · `3` | Delete posted clips; remove saved data older than this (YouTube page, Settings -> Disk) |
| `LOCAL_MEME_DIR` | `Memes Assets/`, `memes/` | Use only this folder for meme clips |
| `LOCAL_CAPTION_SCRIPT` | `roman` | `roman` writes Hindi / Urdu subtitles in Roman letters, `original` keeps the script (Create page) |
| `LOCAL_ENCODER` | `auto` | `auto` uses the graphics chip (Quick Sync / NVENC / AMF) when it works, `cpu` forces x264 (Settings -> Health) |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` | — | your Google OAuth "Desktop app" client; the secret is write-only in the UI |
| `YOUTUBE_PRIVACY` | `private` | default visibility of uploads: `private`, `unlisted` or `public` |
| `YOUTUBE_CATEGORY` | `22` | YouTube category id (22 People & Blogs, 20 Gaming, 24 Entertainment ...) |
| `YOUTUBE_DAILY_UPLOADS` | `10` | your own cap per (Pacific) day; extra clips wait for the next day |
| `YOUTUBE_AUTO_RESUME` | `1` | waiting uploads continue by themselves while the app is open |
| `YOUTUBE_NOTIFY_SUBSCRIBERS` | `0` | tell your subscribers about each upload (off, so a batch does not spam them) |
| `MUAPI_API_KEY` | — | api mode |

The ranking prompts and thresholds are in `shorts_generator/highlights.py` (`VIRALITY_CRITERIA`, `HIGHLIGHT_SYSTEM_PROMPT`, `SIGNAL_INFO`, `CHUNK_SIZE_SECONDS`).

## Tests

```bash
python -m unittest discover -s tests -t .       # ~30 s, offline (never uses your real keys), includes real ffmpeg renders
```

## Troubleshooting

* **"The API key was rejected" / "AI key missing"** — open Settings, paste a key, press *Test connection*.
* **"Every Gemini model failed … quota"** — the free tier has a daily limit *per model*; the tool already tries every model. Wait for the reset, or add an OpenAI key in Settings. **Nothing is lost**: press *Try again* and it continues from the saved answers.
* **"Every Gemini model failed … overloaded"** — Google's servers are busy. The job **retries by itself** (after 1, 3, 7 and 15 minutes; the card shows the countdown) as long as it is still in the cheap stages (reading the link, transcript, ranking). Nothing is lost; if all five tries fail, press *Try again* later.
* **"YouTube wants a sign-in (bot check)"** / age-restricted / members-only — set *Videos that need a login* in Settings to the browser you are signed in to YouTube with.
* **"This video has no usable captions"** — install the optional Whisper: `pip install -r requirements-whisper.txt` (slow on a CPU).
* **Captions don't appear** — your ffmpeg needs libass (the standard "full" builds have it). The header shows "captions unavailable" when it doesn't.
* **A lot of disk used** — Settings → Disk used. Earlier versions kept the whole downloaded video (`output/source_*.mp4`); it is no longer needed and can be deleted there (or with `python main.py --clean x`).
* **"Something stopped working" with a YouTube link** — YouTube changes often and an old yt-dlp is the usual reason. Settings → Health → **Update yt-dlp** (or `pip install -U yt-dlp`).
* **Not sure what is wrong** — `python main.py --doctor` (add `--deep` to test the network and the AI key too). To report a bug, Settings → *Download diagnostics* gives a zip with the log, versions and settings; **keys, proxy and cookies are never included**. The log file is `output/logs/studio.log`.
* **"Your Google consent screen is still in Testing"** (an alert) — Google marks such sign-ins as expiring after 7 days; the app notices at sign-in. Publish the app (YouTube page, step 3: *Audience → Publish app → In production*) and press **Reconnect** once.
* **"YouTube sign-in expired"** — YouTube page → **Reconnect**. Queued clips wait and then go. If it keeps expiring after a week, the Google consent screen is still in *Testing*: publish it ("In production") and reconnect.
* **"Google hasn't verified this app"** — expected for your own app: *Advanced → Go to … (unsafe)*, and keep every permission box ticked (an unticked upload permission is detected and explained).
* **A clip says "YouTube made this video private"** — YouTube restricts some API projects; open the video in YouTube Studio and change the visibility there.
* **Uploads say "waiting"** — your daily cap or YouTube's quota is used up; they continue by themselves (the note under *Rendered clips* says when). *Settings → Most uploads per day* changes your own cap.
* **Slow rendering** — Settings -> Health shows the video encoder; if it says "x264 on the processor", update the graphics driver. 720p in "Picture quality" is faster still.
* **"Quality check failed"** — the clip was broken (wrong length, no sound, black picture) and was not kept; render it again, or pick a different quality.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

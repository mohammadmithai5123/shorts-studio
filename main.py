"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." --mode local --num-clips 3

Local mode never downloads the whole video: it reads the video's captions, ranks the best moments and
fetches only the clips it renders. Prefer the web UI for browsing candidates:  python app.py
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import __version__, generate_shorts


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", nargs="?", default="", help="YouTube (or other video site) link, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI does everything) or local (runs here; nothing is downloaded as a whole).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--language", default=None,
                        help="Caption / speech language code, e.g. 'en' or 'hi' (default: the video's own language)")
    parser.add_argument("--format", default="720", help="api mode only: resolution MuAPI downloads at (360/480/720/1080)")

    local = parser.add_argument_group("local mode")
    local.add_argument("--focus", default="", help='what to look for, e.g. "funny moments" or "key insights"')
    local.add_argument("--min-seconds", type=int, default=20, help="shortest clip (default: 20)")
    local.add_argument("--max-seconds", type=int, default=59, help="longest clip (default and most: 59, a Short stays under a minute)")
    local.add_argument("--quality", default="1080", choices=["480", "720", "1080", "1440", "2160"],
                       help="tallest source picture fetched per clip (default: 1080)")
    local.add_argument(
        "--layout", choices=["auto", "track", "split", "fit"], default="auto",
        help="auto (default) picks per clip; track = stable crop following a speaker; "
             "split = facecam on top + gameplay below; fit = whole frame over a blurred background.",
    )
    local.add_argument("--captions", action="store_true", help="burn big-type captions into each clip")
    local.add_argument("--caption-style", choices=["pop", "classic", "yellow", "box", "karaoke"], default=None,
                       help="look of the captions (default: pop with --edit, else classic)")
    local.add_argument("--edit", choices=["off", "clean", "viral"], default="off",
                       help="automatic editing: clean = jump cuts, zooms, animated captions, progress bar; viral = also popups, sounds, end card")
    local.add_argument("--title-card", action="store_true", help="show the clip's title big for its first seconds")
    local.add_argument("--watermark", default="", help='small corner text, e.g. "@mychannel"')
    local.add_argument("--no-normalize", action="store_true", help="keep the original loudness (default: -14 LUFS)")
    local.add_argument("--kit", action="store_true", help="also write a title, description and hashtags per clip")
    local.add_argument("--refresh", action="store_true", help="ignore the cached transcript and signals")
    local.add_argument("--doctor", action="store_true", help="check ffmpeg, yt-dlp, the AI key, disk space... and exit")
    local.add_argument("--deep", action="store_true", help="with --doctor: also test the network and the AI provider")
    local.add_argument("--clean", action="store_true",
                       help="show disk use of the output folder and delete old whole-video downloads, then exit")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument("--version", action="version", version=f"Shorts Studio {__version__}")
    args = parser.parse_args()

    if args.doctor:
        from shorts_generator.local import health

        report = health.doctor(deep=args.deep)
        print(health.format_report(report))
        return 0 if report["ok"] else 1

    if not args.url and not args.clean:
        parser.error("a video link or file path is required (or use --doctor / --clean)")

    from shorts_generator import logsetup

    log = logsetup.setup()
    log.info("CLI run: %s", " ".join(a for a in sys.argv[1:]))

    if args.clean:
        from shorts_generator.local import storage

        before = storage.usage()
        freed = storage.clean(["legacy_downloads"])
        mb = lambda n: f"{n / 1e6:,.0f} MB"
        print(f"clips {mb(before['clips'])} | cache {mb(before['cache'])} | LLM cache {mb(before['llm_cache'])} | "
              f"old whole-video downloads {mb(before['legacy_downloads'])}")
        print(f"removed old downloads: {mb(freed.get('legacy_downloads', 0))}")
        return 0

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
            layout=args.layout,
            captions=args.captions,
            focus=args.focus,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            caption_style=args.caption_style,
            edit=args.edit,
            title_card=args.title_card,
            watermark=args.watermark,
            normalize_audio=not args.no_normalize,
            quality=args.quality,
            kit=args.kit,
            refresh=args.refresh,
        )
    except Exception as e:
        log.exception("CLI run failed")
        print(f"\nFAILED: {e}", file=sys.stderr)
        print(f"(details: {logsetup.log_dir() / 'studio.log'})", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
    if result.get("transcript_source"):
        print(f"Transcript:    {result['transcript_source']}")
    print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("kit"):
            print(f"     post:   {s['kit'].get('title')}  {' '.join(s['kit'].get('hashtags', []))}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

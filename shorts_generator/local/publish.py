"""Publish kit: a title, description and hashtags for each clip, written by the LLM.

The kit is what you paste next to the video on YouTube Shorts / Reels / TikTok. One LLM call covers
all the clips of a request, and the answer is cached like every other LLM answer.
"""
import re
from typing import Callable, Dict, List, Sequence, Tuple

from .. import config
from ..highlights import _clock, _excerpt, _parse_json_loose

KIT_PROMPT = """You write the text that goes with short vertical videos (YouTube Shorts, Instagram Reels, TikTok).

Source video: "{video_title}"
{who}
Spoken language of the clips: {language}. Write in the language and script the speaker's audience uses; for mixed Hindi/English speech write natural Hinglish in Roman letters.{script_rule}

For EACH clip below write:
- "title": at most 60 characters, a headline nobody scrolls past. Lead with the DRAMA of this clip (trouble, a shock, a loss, a dare, a big reaction), name who it happens to, and leave the result open so the viewer has to watch to find out. Use the everyday words fans use and one emoji at the end. The style (for Hindi / Urdu audiences): "Ducky ke peeche lag gayi police 😱", "Ducky Supra chura ke phas gaya 😳", "Mummy ka superchat dekhte hi Carry ke hosh ud gaye 😂" (in English videos, the same style in English). Too weak, never write: a flat label ("Epic car chase", "Funny moment") or a sentence that already tells the whole thing ("Ducky ne car pakdi", "Carry ne kalesh dekha"). Exaggerate the feeling, never the facts: do not invent people or events. No hashtags, no quotation marks
- "description": one or two short lines that add context, ending with a light question or "comment karo" / "part 2?". Never ask people to follow or subscribe to anyone
- "hashtags": 6 to 8 hashtags, lowercase, no spaces, mixing broad reach tags and specific ones about the clip's topic

Clips:
{clips}

{edit_rules}Respond ONLY with valid JSON (no markdown): {{"clips":[{{"id":int,"title":"string","description":"string","hashtags":["#tag"]{edit_fields}}}]}}
Use the ids given above."""


EDIT_RULES = """ALSO plan the on-screen edit of each clip, using its timed lines (seconds from the start of the clip):
- "hook": 2 to 5 words, a big headline for the first seconds in the same cliffhanger style as the title, naming the person when you can, like "DUCKY KE PEECHE POLICE" (Roman letters, no emoji)
- "popups": 3 to 6 items {"at": seconds, "mood": one of "laugh", "shock", "fail", "sad", "hype", "awkward", "angry", "win", "emoji": "one emoji", "text": "optional, 1 to 3 Roman-letter words"} exactly at reactions, surprises and punchlines (a meme of that mood is shown there)
- "keywords": up to 6 words copied from the timed lines that should be highlighted in the captions

"""
EDIT_FIELDS = ',"hook":"string","popups":[{"at":1.5,"mood":"laugh","emoji":"x","text":""}],"keywords":["word"]'

MEME_RULES = """ALSO choose reaction memes for each clip from the list below ("id: when to use it"). The video pauses right after a line and the meme is shown, like a good editor does it.
- "memes": 0 to 3 items {"at": the second where the line that calls for it ENDS, "id": one id from the list}
- choose a meme only when it truly fits what was just said or happened; none is better than a wrong one
- keep them at least 6 seconds apart, not in the first 3 seconds, and do not use the same id twice
MEME LIST:
%s

"""
MEME_FIELDS = ',"memes":[{"at":7.5,"id":"id_from_the_list"}]'


STYLE = 2                    # how titles are written; clips whose saved text is older are asked again when rendered


def who_line(creator: str) -> str:
    creator = " ".join(str(creator or "").split()).replace('"', "")[:60]
    if creator:
        return (f'The video is from the channel "{creator}". When the person in a clip is that creator, name them in the title the way their fans '
                'do (a short first name or nickname).')
    return "Name the person only if the video title or what is said tells you who it is; never guess a name."


ROMAN_RULE = " Use only Roman (Latin) letters in everything you write, never Devanagari or Urdu script."


def _timed_lines(transcript: Dict, start: float, end: float, limit: int = 1800) -> str:
    lines = [f"[{max(0.0, s['start'] - start):.1f}] {s.get('text', '').strip()}"
             for s in transcript.get("segments", []) if s["end"] > start and s["start"] < end and s.get("text", "").strip()]
    return "\n".join(lines)[:limit]


def edit_notes(item: Dict, start: float, end: float, meme_ids: Sequence[str] = ()) -> Dict:
    """The AI's edit plan, checked. Popup and meme times are kept as video times ("t"), so a later trim cannot shift them.

    ``meme_ids``: what it was allowed to choose from. The "memes" key exists only when it was asked.
    """
    popups = []
    for p in item.get("popups") or []:
        if not isinstance(p, dict):
            continue
        try:
            at = float(p.get("at"))
        except (TypeError, ValueError):
            continue
        emoji, text = str(p.get("emoji") or "").strip()[:8], " ".join(str(p.get("text") or "").split())[:24]
        mood = str(p.get("mood") or "").strip().lower()[:12]
        if 0 <= at <= end - start and (emoji or text or mood):
            popups.append({"t": round(start + at, 2), "emoji": emoji, "text": text, **({"mood": mood} if mood else {})})
    notes = {"hook": " ".join(str(item.get("hook") or "").split())[:46],
             "popups": popups[:8],
             "keywords": [str(k).strip()[:30] for k in (item.get("keywords") or [])[:8] if str(k).strip()]}
    if meme_ids:
        picks = []
        for m in item.get("memes") or []:
            try:
                at, ident = float(m.get("at")), str(m.get("id") or "").strip()
            except (TypeError, ValueError, AttributeError):
                continue
            if ident in meme_ids and 0 <= at <= end - start + 0.5:
                picks.append({"t": round(start + at, 2), "id": ident})
        notes["memes"] = picks[:4]
    return notes


def make_publish_kit(project: Dict, transcript: Dict, ids: List[int], llm_fn: Callable[[str], str],
                     with_edit: bool = False, meme_menu: Sequence[Tuple[str, str]] = (), brand: str = "") -> Dict[int, Dict]:
    """{candidate id: {title, description, hashtags[, hook, popups, keywords]}} for the requested candidates.

    ``with_edit``: also ask for the on-screen edit (headline, popups, words to highlight) in the same single AI call.
    ``meme_menu``: (id, when to use) of the person's memes; the same call then picks memes by what is said.
    ``brand``: the person's own channel; it becomes the first hashtag.
    """
    lines = []
    for i in ids:
        c = project["candidates"][i]
        s, e = float(c["start_time"]), float(c["end_time"])
        lines.append(
            f"[id {i}] {_clock(s)}-{_clock(e)} ({e - s:.0f}s)\n"
            f"working title: {c.get('title', '')}\n"
            f"opening line: {c.get('hook_sentence', '')}\n"
            f"what is said: {_excerpt(transcript, s, e, limit=500)}"
            + (f"\ntimed lines:\n{_timed_lines(transcript, s, e)}" if with_edit else "")
        )
    prompt = KIT_PROMPT.format(
        video_title=project.get("title", ""),
        who=who_line((project.get("source") or {}).get("channel", "")),
        language=(transcript.get("language") or project.get("source", {}).get("language") or "the video's language"),
        script_rule=ROMAN_RULE if config.caption_script() == "roman" else "",
        clips="\n\n".join(lines),
        edit_rules=(EDIT_RULES + (MEME_RULES % "\n".join(f"{i}: {use}" for i, use in meme_menu) if meme_menu else "")) if with_edit else "",
        edit_fields=(EDIT_FIELDS + (MEME_FIELDS if meme_menu else "")) if with_edit else "",
    )
    last_error = "unknown"
    for _ in range(3):
        try:
            parsed = _parse_json_loose(llm_fn(prompt))
            raw = parsed.get("clips") if isinstance(parsed, dict) else parsed
            out: Dict[int, Dict] = {}
            for item in raw if isinstance(raw, list) else []:
                try:
                    cid = int(item.get("id"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if cid not in ids:
                    continue
                tags = []
                for t in item.get("hashtags") or []:
                    t = re.sub(r"[^\w]", "", str(t).lstrip("#"), flags=re.UNICODE)
                    if t:
                        tags.append("#" + t.lower())
                own = re.sub(r"[^\w]", "", brand, flags=re.UNICODE).lower()
                if own:
                    tags.insert(0, "#" + own)
                out[cid] = {
                    "style": STYLE,
                    "title": str(item.get("title") or "").strip().strip('"')[:100],
                    "description": str(item.get("description") or "").strip()[:500],
                    "hashtags": list(dict.fromkeys(tags))[:10],
                }
                if with_edit:
                    c = project["candidates"][cid]
                    out[cid].update(edit_notes(item, float(c["start_time"]), float(c["end_time"]), [i for i, _ in meme_menu]))
            if out:
                return out
            last_error = "no usable clips in the answer"
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
    raise RuntimeError(f"Could not write the publish kit: {last_error}")

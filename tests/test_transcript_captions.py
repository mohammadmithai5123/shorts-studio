import unittest

from shorts_generator.local import captions as C
from shorts_generator.local import transcript as T

# YouTube auto-captions ("rolling"): a line of words with offsets, then a 1-char newline event, then the next line.
JSON3 = {"events": [
    {"tStartMs": 300, "dDurationMs": 5000, "segs": [{"utf8": "yo"}, {"utf8": " whats", "tOffsetMs": 400}, {"utf8": " up", "tOffsetMs": 900}]},
    {"tStartMs": 2600, "dDurationMs": 2600, "segs": [{"utf8": "\n"}]},
    {"tStartMs": 2600, "dDurationMs": 5500, "segs": [{"utf8": "welcome"}, {"utf8": " back", "tOffsetMs": 500}]},
    {"tStartMs": 5300, "dDurationMs": 3000, "segs": [{"utf8": "welcome"}, {"utf8": " back", "tOffsetMs": 500}]},   # repeated line
    {"tStartMs": 9000, "dDurationMs": 2000, "segs": [{"utf8": "a hand made cue with no word timings"}]},
]}


class Json3(unittest.TestCase):
    def test_lines_become_non_overlapping_segments_with_word_timing(self):
        segs = T.parse_json3(JSON3)
        self.assertEqual([s["text"] for s in segs], ["yo whats up", "welcome back", "a hand made cue with no word timings"])
        for a, b in zip(segs, segs[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)
        self.assertEqual(segs[0]["words"][1]["start"], 0.7)          # tStart 0.3 s + 0.4 s offset
        self.assertAlmostEqual(segs[0]["words"][-1]["end"], segs[0]["end"], places=3)

    def test_hand_made_cue_spreads_its_words(self):
        seg = T.parse_json3(JSON3)[-1]
        self.assertEqual(len(seg["words"]), 8)
        self.assertTrue(all(w["end"] > w["start"] for w in seg["words"]))

    def test_vtt(self):
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello <c>there</c> world\n\n00:00:04.000 --> 00:00:05.500\nsecond line\n"
        segs = T.parse_vtt(vtt)
        self.assertEqual([s["text"] for s in segs], ["hello there world", "second line"])
        self.assertEqual((segs[1]["start"], segs[1]["end"]), (4.0, 5.5))

    def test_usable_needs_enough_speech(self):
        few = [{"start": 0, "end": 1, "text": "x"}] * 3
        self.assertFalse(T._usable(few, 600))
        many = [{"start": i, "end": i + 2, "text": "x"} for i in range(0, 100, 5)]
        self.assertTrue(T._usable(many, 600))


class Captions(unittest.TestCase):
    SEGS = [
        {"start": 10.0, "end": 12.0, "text": "one two three four five six",
         "words": [{"start": 10 + i * 0.3, "end": 10.3 + i * 0.3, "text": w} for i, w in enumerate("one two three four five six".split())]},
        {"start": 12.5, "end": 14.0, "text": "no word timings here"},
        {"start": 40.0, "end": 41.0, "text": "outside the clip"},
        {"start": 14.5, "end": 15.0, "text": "..."},
    ]

    def test_cards_are_short_inside_the_clip_and_never_overlap(self):
        ev = C.build_events(self.SEGS, 10.0, 16.0)
        texts = [" ".join(w[2] for w in words) for _, _, words in ev]
        self.assertNotIn("outside the clip", " ".join(texts))
        self.assertNotIn("...", texts)                                # punctuation-only is dropped
        self.assertTrue(all(len(t.split()) <= C.MAX_WORDS for t in texts))
        for (s1, e1, _), (s2, e2, _) in zip(ev, ev[1:]):
            self.assertLessEqual(e1, s2 + 1e-6)
        self.assertGreaterEqual(ev[0][0], 0.0)
        self.assertLessEqual(ev[-1][1], 6.0)

    def test_ass_styles(self):
        for style in C.STYLES:
            ass = C.build_ass(self.SEGS, 10.0, 16.0, 1080, 1920, style=style)
            self.assertIn("PlayResY: 1920", ass)
            self.assertIn("Dialogue:", ass)
        self.assertIn("\\k", C.build_ass(self.SEGS, 10.0, 16.0, 1080, 1920, style="karaoke"))
        self.assertIn("BorderStyle", C.build_ass(self.SEGS, 10.0, 16.0, 1080, 1920, style="box"))

    def test_title_card_and_watermark_without_captions(self):
        ass = C.build_ass([], 0, 20, 1080, 1920, captions=False, title="A big moment", watermark="@me", title_top=800)
        self.assertIn("A big moment", ass)
        self.assertIn("@me", ass)
        self.assertIn("Style: Title", ass)
        self.assertIsNone(C.build_ass([], 0, 20, 1080, 1920, captions=False))

    def test_devanagari_gets_a_font_that_has_the_glyphs(self):
        segs = [{"start": 0, "end": 2, "text": "यह एक टेस्ट है"}]
        self.assertIn("Nirmala UI", C.build_ass(segs, 0, 3, 1080, 1920))

    def test_ass_special_characters_are_removed(self):
        segs = [{"start": 0, "end": 2, "text": "evil {\\an9} text"}]
        self.assertNotIn("{\\an9}", C.build_ass(segs, 0, 3, 1080, 1920))


if __name__ == "__main__":
    unittest.main()

"""A long video is read in parts: one part with nothing in it (or a bad answer) must not lose all the others."""
import json
import unittest

import tests  # noqa: F401  (blank keys, temp output folder)
from shorts_generator import highlights as H

SEGMENTS = [{"start": float(t), "end": t + 9.0, "text": f"line {t}"} for t in range(0, 3600, 10)]
LONG = {"duration": 3600.0, "language": "en", "segments": SEGMENTS}          # read in 4 parts
GOOD = json.dumps({"highlights": [{"title": "t", "start_time": 30, "end_time": 70, "score": 80}]})
EMPTY = json.dumps({"highlights": []})


class Parts(unittest.TestCase):
    def run_with(self, answer_for_part, transcript=LONG):
        prompts = []

        def llm(prompt):
            if "Transcript sample:" in prompt:
                return json.dumps({"content_type": "vlog", "density": "low"})
            if "ranking" in prompt and "Candidates:" in prompt:
                return json.dumps({"ranking": []})
            body = prompt.split("Transcript:\n", 1)[1]
            part = 0
            for text in self.parts:
                if body.startswith(text):
                    part = self.parts.index(text)
            prompts.append((part, prompt))
            return answer_for_part(part, sum(1 for p, _ in prompts if p == part))

        self.parts = [H.build_transcript_text(c)[:400] for c in H.chunk_transcript(transcript)]
        return H.get_highlights(transcript, num_clips=10, llm_fn=llm), prompts

    def test_a_part_with_nothing_is_asked_twice_and_then_left(self):
        result, prompts = self.run_with(lambda part, n: EMPTY if part == 1 else GOOD)
        self.assertEqual(len(self.parts), 4)
        self.assertEqual(len([p for p, _ in prompts if p == 1]), 2)
        self.assertEqual(len(result["highlights"]), 3)

    def test_a_lazy_empty_answer_gets_a_second_chance(self):
        result, _ = self.run_with(lambda part, n: EMPTY if (part, n) == (0, 1) else GOOD)
        self.assertEqual(len(result["highlights"]), 4)

    def test_one_garbled_part_is_skipped_and_every_retry_is_a_new_question(self):
        result, prompts = self.run_with(lambda part, n: "not json at all" if part == 2 else GOOD)
        asked = [text for p, text in prompts if p == 2]
        self.assertEqual((len(asked), len(set(asked))), (3, 3))                 # a saved bad answer is never replayed
        self.assertEqual(len(result["highlights"]), 3)

    def test_most_parts_garbled_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "invalid output"):
            self.run_with(lambda part, n: GOOD if part == 0 else "nonsense")

    def test_a_short_video_still_reports_a_bad_answer(self):
        short = {"duration": 300.0, "language": "en", "segments": SEGMENTS[:30]}
        with self.assertRaisesRegex(RuntimeError, "invalid output"):
            self.run_with(lambda part, n: EMPTY, short)


if __name__ == "__main__":
    unittest.main()

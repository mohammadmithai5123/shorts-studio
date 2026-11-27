import json
import unittest

from shorts_generator import highlights as H
from shorts_generator.local import signals as S


def transcript(duration=4000.0, step=10.0):
    segs, t = [], 0.0
    while t < duration:
        segs.append({"start": t, "end": t + step, "text": f"line at {int(t)}"})
        t += step
    return {"duration": duration, "segments": segs}


class Chunking(unittest.TestCase):
    def test_chunk_timestamps_are_relative_so_later_chunks_survive_sanitising(self):
        chunks = H.chunk_transcript(transcript())
        self.assertGreater(len(chunks), 2)
        c = chunks[2]
        self.assertTrue(c["segments"][0]["start"] < 20)                 # relative to the chunk, not the video
        self.assertEqual(c["_offset"], 2 * (H.CHUNK_SIZE_SECONDS - H.CHUNK_OVERLAP_SECONDS))
        raw = [{"title": "x", "start_time": 500, "end_time": 540, "score": 80}]
        self.assertEqual(len(H._sanitize_highlights(raw, c["duration"])), 1)


class PromptOptions(unittest.TestCase):
    def test_focus_and_length_reach_the_prompt(self):
        seen = {}

        def llm(prompt):
            seen["p"] = prompt
            return json.dumps({"highlights": [{"title": "t", "start_time": 5, "end_time": 40, "score": 70}]})

        opts = H.Options(focus="the funniest reactions", min_seconds=15, max_seconds=30)
        H.call_highlight_api("[0.0s] hi", {}, 300.0, 3, llm_fn=llm, options=opts)
        self.assertIn("the funniest reactions", seen["p"])
        self.assertIn("between 15 and 30 seconds", seen["p"])

    def test_no_focus_means_no_brief(self):
        self.assertEqual(H.Options().focus_note(), "")


class Rerank(unittest.TestCase):
    def cands(self, n=12):
        return [{"title": f"c{i}", "start_time": i * 300.0, "end_time": i * 300.0 + 30, "score": 95, "hook_sentence": "h",
                 "virality_reason": "r"} for i in range(n)]

    def test_global_rerank_breaks_ties_and_floors_the_rest(self):
        t = transcript()

        def llm(prompt):
            return json.dumps({"ranking": [{"id": 3, "score": 91, "reason": "a"}, {"id": 5, "score": 70, "reason": "b"},
                                           {"id": 1, "score": 55, "reason": "c"}, {"id": 999, "score": 99, "reason": "bad id"}]})

        out = H.rerank_highlights(self.cands(), t, 3, {}, llm)
        self.assertEqual([o["score"] for o in out[:3]], [91, 70, 55])
        self.assertLessEqual(max(o["score"] for o in out[3:]), 54)
        self.assertEqual(len(out), 12)

    def test_garbage_answer_keeps_chunk_scores(self):
        cands = self.cands()
        out = H.rerank_highlights(cands, transcript(), 3, {}, lambda p: "not json")
        self.assertEqual([o["score"] for o in out], [95] * 12)

    def test_signals_are_blended_and_shown_to_the_llm(self):
        seen = {}

        def llm(prompt):
            seen["p"] = prompt
            return json.dumps({"ranking": [{"id": 0, "score": 90, "reason": "x"}, {"id": 1, "score": 90, "reason": "y"},
                                           {"id": 2, "score": 90, "reason": "z"}]})

        opts = H.Options(signals={"energy": lambda s, e: 100.0 if s < 300 else 0.0})
        out = H.rerank_highlights(self.cands(5), transcript(), 3, {}, llm, opts)
        self.assertIn("voice energy", seen["p"])
        by_title = {o["title"]: o for o in out}
        self.assertGreater(by_title["c0"]["score"], by_title["c1"]["score"])       # same LLM score, louder wins
        w = H.signal_weights(["energy"])["energy"]
        self.assertEqual(by_title["c0"]["score"], round((1 - w) * 90 + w * 100))
        self.assertEqual(by_title["c0"]["signals"], {"energy": 100})

    def test_pending_signals_are_resolved_and_failures_dropped(self):
        opts = H.Options(pending={"energy": lambda: (lambda s, e: 80.0), "replay": lambda: 1 / 0})
        sig = opts.resolve_signals()
        self.assertEqual(list(sig), ["energy"])
        self.assertEqual(opts.pending, {})

    def test_signal_weights_keep_the_llm_in_charge(self):
        w = H.signal_weights(["energy", "replay"])
        self.assertLessEqual(sum(w.values()), H.MAX_SIGNAL_WEIGHT + 1e-9)


class Edges(unittest.TestCase):
    def test_snap_moves_edges_to_sentence_boundaries(self):
        t = {"duration": 100.0, "segments": [{"start": 10.0, "end": 14.0, "text": "a"}, {"start": 14.0, "end": 20.0, "text": "b"},
                                              {"start": 20.0, "end": 26.0, "text": "c"}]}
        out = H.snap_to_sentences([{"start_time": 12.0, "end_time": 24.0, "score": 1}], t)
        self.assertAlmostEqual(out[0]["start_time"], 9.8)
        self.assertAlmostEqual(out[0]["end_time"], 26.4)

    def test_enforce_duration_trims_and_extends_on_sentence_ends(self):
        t = transcript(200.0, 5.0)
        long_clip = H.enforce_duration([{"start_time": 10.0, "end_time": 100.0}], t, 20, 60)[0]
        self.assertLessEqual(long_clip["end_time"] - 10.0, 60.0)
        self.assertEqual(long_clip["end_time"] % 5, 0)
        short_clip = H.enforce_duration([{"start_time": 10.0, "end_time": 15.0}], t, 20, 60)[0]
        self.assertGreaterEqual(short_clip["end_time"] - 10.0, 20.0)

    def test_dedupe_drops_overlaps_but_keeps_the_better_one(self):
        hs = [{"start_time": 0, "end_time": 30, "score": 80}, {"start_time": 5, "end_time": 35, "score": 90},
              {"start_time": 100, "end_time": 130, "score": 70}]
        kept = H.dedupe_highlights(hs)
        self.assertEqual([h["score"] for h in kept], [90, 70])


class Signals(unittest.TestCase):
    def test_window_percentile(self):
        series = [0.0] * 50 + [10.0] * 10 + [0.0] * 50
        pct = S.window_pct_fn(series)
        self.assertGreater(pct(50, 60), 90)
        self.assertLess(pct(0, 10), 10)
        self.assertEqual(pct(0, 1), 50.0)                              # too short to compare

    def test_heatmap_series_and_downsample(self):
        heat = [{"start_time": 0, "end_time": 5, "value": 0.2}, {"start_time": 5, "end_time": 10, "value": 1.0}]
        s = S.heatmap_series(heat, 10)
        self.assertEqual(list(s), [0.2] * 5 + [1.0] * 5)
        self.assertIsNone(S.heatmap_series(None, 10))
        self.assertEqual(len(S.downsample(list(range(1000)), 100)), 100)
        self.assertEqual(S.downsample(None), [])


if __name__ == "__main__":
    unittest.main()

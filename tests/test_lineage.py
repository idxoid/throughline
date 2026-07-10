import unittest

import throughline as tl
from throughline.modules.lineage import LineageLedger, LineageMiddleware, lines_of
from throughline.modules.policy import Policy, Transform


class LinesOf(unittest.TestCase):
    def test_str_list_dict_scalar(self):
        self.assertEqual(lines_of("a\nb"), ["a", "b"])
        self.assertEqual(lines_of(["x", 1]), ["x", "1"])
        self.assertEqual(lines_of({"k": "v", "n": 2}), ["k: v", "n: 2"])
        self.assertEqual(lines_of(42), ["42"])
        self.assertEqual(lines_of(None), [])


class LedgerAttribution(unittest.TestCase):
    def test_carry_modify_generate_drop(self):
        ledger = LineageLedger("r1")
        ledger.snapshot_source(
            "alpha stays exactly the same\n"
            "bravo line gets a small edit\n"
            "charlie line will be removed\n"
            "delta stays exactly the same")
        stats = ledger.evolve(
            "edit",
            "alpha stays exactly the same\n"
            "bravo line got a small edit\n"
            "TOTALLY NEW LINE\n"
            "delta stays exactly the same")
        self.assertEqual(stats["carry"], 2)      # alpha, delta
        self.assertEqual(stats["modify"], 1)     # bravo
        self.assertEqual(stats["generate"], 1)   # the new line
        self.assertEqual(stats["drop"], 1)       # charlie

        blame = ledger.blame()
        by_text = {entry["text"]: entry for entry in blame}
        self.assertEqual(by_text["alpha stays exactly the same"]["step"], "input")
        self.assertEqual(by_text["alpha stays exactly the same"]["op"], "source")
        self.assertEqual(by_text["bravo line got a small edit"]["op"], "modify")
        self.assertEqual(by_text["bravo line got a small edit"]["origin"], "input")
        self.assertEqual(by_text["TOTALLY NEW LINE"]["step"], "edit")
        self.assertEqual(by_text["TOTALLY NEW LINE"]["op"], "generate")

    def test_modify_links_parent(self):
        ledger = LineageLedger("r1")
        ledger.snapshot_source("the quick brown fox jumps")
        ledger.evolve("edit", "the quick brown fox JUMPED")
        entry = ledger.blame()[0]
        self.assertEqual(entry["op"], "modify")
        self.assertEqual(entry["step"], "edit")
        self.assertEqual(entry["origin"], "input")   # ancestry reaches the source
        chain = ledger.trace(entry["id"])
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[-1].text, "the quick brown fox jumps")

    def test_multi_step_ancestry(self):
        ledger = LineageLedger("r1")
        ledger.snapshot_source("one two three four five")
        ledger.evolve("s1", "one two three four five six")     # modify (similar)
        ledger.evolve("s2", "one two three four five six seven")
        entry = ledger.blame()[0]
        self.assertEqual(entry["step"], "s2")
        self.assertEqual(entry["origin"], "input")
        self.assertEqual(entry["depth"], 3)

    def test_unrelated_replacement_is_generate(self):
        ledger = LineageLedger("r1")
        ledger.snapshot_source("aaaa aaaa aaaa")
        ledger.evolve("rewrite", "zzzz yyyy xxxx")
        entry = ledger.blame()[0]
        self.assertEqual(entry["op"], "generate")
        self.assertEqual(entry["origin"], "rewrite")

    def test_stats_and_jsonl(self):
        import json
        ledger = LineageLedger("r1")
        ledger.snapshot_source("a\nb")
        ledger.evolve("s", "a\nc")
        stats = ledger.stats()
        self.assertEqual(stats["lines"], 2)
        self.assertEqual(stats["steps"], ["input", "s"])
        for line in ledger.to_jsonl().splitlines():
            record = json.loads(line)
            self.assertEqual(record["run_id"], "r1")

    def test_render_blame_contains_markers(self):
        ledger = LineageLedger("r1")
        ledger.snapshot_source("keep me")
        ledger.evolve("gen", "keep me\nbrand new")
        rendered = ledger.render_blame()
        self.assertIn("=", rendered)   # source marker
        self.assertIn("+", rendered)   # generate marker
        self.assertIn("gen", rendered)


class MiddlewareIntegration(unittest.TestCase):
    def test_flow_lineage_end_to_end(self):
        def compose(payload, ctx):
            return payload + "\nadded by compose"

        def edit(payload, ctx):
            return payload.replace("hello", "HELLO") + "\nsigned off"

        flow = tl.Flow([tl.as_step(compose, "compose"), tl.as_step(edit, "edit")],
                       middleware=[LineageMiddleware()])
        result = flow.run("hello world\nsecond line")
        ledger = result.lineage
        by_text = {e["text"]: e for e in ledger.blame()}
        self.assertEqual(by_text["second line"]["step"], "input")
        self.assertEqual(by_text["added by compose"]["step"], "compose")
        self.assertEqual(by_text["signed off"]["step"], "edit")
        self.assertEqual(by_text["HELLO world"]["step"], "edit")
        self.assertEqual(by_text["HELLO world"]["origin"], "input")

    def test_extract_targets_a_field(self):
        def build(payload, ctx):
            return {"question": payload, "answer": "line a\nline b"}

        def refine(payload, ctx):
            return {**payload, "answer": payload["answer"] + "\nline c"}

        flow = tl.Flow([tl.as_step(build, "build"), tl.as_step(refine, "refine")],
                       middleware=[LineageMiddleware(extract=lambda p: p["answer"]
                                                     if isinstance(p, dict) else p)])
        ledger = flow.run("q?").lineage
        by_text = {e["text"]: e for e in ledger.blame()}
        self.assertEqual(by_text["line a"]["step"], "build")
        self.assertEqual(by_text["line c"]["step"], "refine")

    def test_list_payload_lineage(self):
        flow = tl.Flow([tl.as_step(lambda docs: docs + ["extra doc"], "append")],
                       middleware=[LineageMiddleware()])
        ledger = flow.run(["doc one", "doc two"]).lineage
        by_text = {e["text"]: e for e in ledger.blame()}
        self.assertEqual(by_text["doc one"]["step"], "input")
        self.assertEqual(by_text["extra doc"]["step"], "append")

    def test_no_egress_sweep_without_a_transform(self):
        flow = tl.Flow([tl.as_step(lambda p: p + "\nmore", "grow")],
                       middleware=[LineageMiddleware()])
        ledger = flow.run("base").lineage
        self.assertNotIn("egress", ledger.stats()["steps"])


class EgressSweep(unittest.TestCase):
    """The run-end sweep: an inner middleware's on_run_end transform must
    land in the blame trail, attributed to the "egress" pseudo-step."""

    @staticmethod
    def _redact(checkpoint, value, ctx):
        if isinstance(value, str) and "sk-live-XYZ" in value:
            return Transform(value.replace("sk-live-XYZ", "[redacted]"),
                             "scrubbed a leaked key")
        return None

    @staticmethod
    def _emit(payload, ctx):
        return "answer line\ntoken sk-live-XYZ rides here"

    def test_egress_transform_lands_in_blame(self):
        flow = tl.Flow([tl.as_step(self._emit, "emit")],
                       middleware=[LineageMiddleware(),
                                   Policy(egress=[self._redact])])
        result = flow.run("q")
        self.assertEqual(result.output,
                         "answer line\ntoken [redacted] rides here")
        ledger = result.lineage
        self.assertNotIn("sk-live", "\n".join(ledger.current_lines()))
        by_text = {e["text"]: e for e in ledger.blame()}
        scrubbed = by_text["token [redacted] rides here"]
        self.assertEqual(scrubbed["step"], "egress")
        self.assertEqual(scrubbed["op"], "modify")
        self.assertEqual(scrubbed["origin"], "emit")  # ancestry survives
        self.assertEqual(by_text["answer line"]["step"], "emit")  # untouched
        self.assertEqual(ledger.stats()["steps"][-1], "egress")

    def test_lineage_inside_policy_misses_the_transform(self):
        # Documented cost of the wrong order: on_run_end runs innermost
        # first, so a ledger INSIDE Policy sweeps before the redaction and
        # keeps the raw text. List Lineage before Policy.
        flow = tl.Flow([tl.as_step(self._emit, "emit")],
                       middleware=[Policy(egress=[self._redact]),
                                   LineageMiddleware()])
        result = flow.run("q")
        self.assertEqual(result.output,
                         "answer line\ntoken [redacted] rides here")
        ledger = result.lineage
        self.assertNotIn("egress", ledger.stats()["steps"])
        self.assertIn("token sk-live-XYZ rides here", ledger.current_lines())


if __name__ == "__main__":
    unittest.main()

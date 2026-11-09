#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-slot statistics from /slots when a server runs with -np N > 1.

Fixtures are the /slots answer RECORDED on 2026-09-17 from a llama.cpp b10930 server
started with `-np 5 -c 150000`: slot 0 generating, slots 1-4 idle holding the cache of
their last request. On that build n_prompt_tokens is prompt PLUS generated so far
(server-context.cpp:540 pushes each sampled token into prompt.tokens), measured as
484 + 810 = 1294 on slot 0, which is why the occupancy below is 1294 and not 2104.

Run:  cd /home/lich/Desktop/llamagputop && python3 -m unittest test_llamagputop_slots -v
No dependencies beyond the standard library, like the program itself.
"""
import copy
import unittest
from unittest.mock import patch

import llamagputop as g

PORT = "65001"          # no llama-server on this port: the /proc walks find nothing
N_CTX = 30208

METRICS_OFF = '{"error":{"code":501,"message":"This server does not support metrics endpoint."}}'
PROPS = ('{"total_slots": 5, "model_path": "/x/model.gguf", '
         '"default_generation_settings": {"n_ctx": 30208}}')


def _slot(sid, task, npt, processed=0, decoded=0, remain=-1, busy=False):
    return {"id": sid, "n_ctx": N_CTX, "speculative": True, "is_processing": busy,
            "id_task": task, "n_prompt_tokens": npt,
            "n_prompt_tokens_processed": processed, "n_prompt_tokens_cache": 0,
            "params": {"n_predict": 3000, "reasoning_format": "auto"},
            "next_token": [{"has_next_token": busy, "has_new_line": False,
                            "n_remain": remain, "n_decoded": decoded}]}


def live_slots():
    """The recorded five-slot answer, a fresh copy each time."""
    return copy.deepcopy([
        _slot(0, 49926, 1294, processed=484, decoded=810, remain=2190, busy=True),
        _slot(1, 41320, 985),
        _slot(2, 51063, 1186),
        _slot(3, 38024, 237),
        _slot(4, 49920, 5074),
    ])


OCCUPIED = 1294 + 985 + 1186 + 237 + 5074      # 8776 — every slot's cache, counted once


class _Server:
    """Stand-in for LlamaProbe._get: answers from a dict, raises when told to."""
    def __init__(self, slots):
        self.slots = slots
        self.down = False

    def __call__(self, path, timeout=2.5):
        import json
        if self.down:
            raise TimeoutError()
        if path == "/metrics":
            return METRICS_OFF
        if path == "/slots":
            return json.dumps(self.slots)
        if path == "/props":
            return PROPS
        raise AssertionError(f"unexpected path {path}")


def _probe(slots):
    pr = g.LlamaProbe(PORT)
    srv = _Server(slots)
    pr._get = srv
    return pr, srv


def _tick(pr, clock, t):
    """One sample() at a fixed clock reading, so rates are exact and not timing-dependent."""
    clock[0] = t
    return pr.sample()


def _generated(slot, decoded):
    """Advance a busy slot the way the server does: n_decoded moves AND n_prompt_tokens
    grows with it (prompt.tokens takes every sampled token), so the fixture keeps the
    identity the real answer showed — n_prompt_tokens = processed + cache + decoded."""
    slot["next_token"][0]["n_decoded"] = decoded
    slot["n_prompt_tokens"] = (slot["n_prompt_tokens_processed"]
                               + slot["n_prompt_tokens_cache"] + decoded)


def _texts(rows):
    return ["".join(t for t, _ in r) for r in rows]


def render_state(slot_stats):
    """A server state as _llama_rows receives it, with the per-slot list attached."""
    d = g.LlamaProbe(PORT)._blank()
    d.update(alive=True, phase="generating", port=PORT, model="model", ctx=N_CTX,
             ctx_total=150000, slots=5, kv=OCCUPIED / 150000, kv_used=OCCUPIED,
             kv_cap=150000, active=1, queued=None, metrics_off=True, decoded=810,
             budget=3000, reasoning_format="auto", spec_type="draft-mtp",
             spec_head=None, spec_nmax="3")
    d["slot_stats"] = slot_stats
    return d


def stats_fixture():
    """Per-slot entries in the shape sample() publishes: one generating, one prefilling,
    one idle carrying a last rate, two idle without."""
    return [
        {"id": 0, "phase": "generating", "n_ctx": N_CTX, "used": 1294, "fill": 1294 / N_CTX,
         "decoded": 810, "budget": 3000, "processed": 484, "tg": 31.2, "pp": None,
         "tg_last": None, "pp_last": None},
        {"id": 1, "phase": "idle", "n_ctx": N_CTX, "used": 985, "fill": 985 / N_CTX,
         "decoded": 0, "budget": None, "processed": 0, "tg": None, "pp": None,
         "tg_last": 29.8, "pp_last": None},
        {"id": 2, "phase": "idle", "n_ctx": N_CTX, "used": 1186, "fill": 1186 / N_CTX,
         "decoded": 0, "budget": None, "processed": 0, "tg": None, "pp": None,
         "tg_last": None, "pp_last": None},
        {"id": 3, "phase": "prefill", "n_ctx": N_CTX, "used": 443, "fill": 443 / N_CTX,
         "decoded": 0, "budget": None, "processed": 443, "tg": None, "pp": 1800.0,
         "tg_last": None, "pp_last": None},
        {"id": 4, "phase": "idle", "n_ctx": N_CTX, "used": 5074, "fill": 5074 / N_CTX,
         "decoded": 0, "budget": None, "processed": 0, "tg": None, "pp": None,
         "tg_last": None, "pp_last": None},
    ]


# What _llama_rows printed for render_state([]) at width 120 BEFORE this feature existed,
# captured from the unmodified program on 2026-09-17. The single-slot render must stay
# byte-identical to it: with one slot the panel IS the slot.
GOLDEN_120 = [
    "status   generating   ctx 30208/slot · 150000 total (5 slots)   active 1, queued —",
    "prefill — (idle; needs --metrics for completed prefills) │ gen — (idle)",
    "metrics off — start with --metrics for reuse, queue and session averages",
    "kv █░░░░░░░░░░░░░░░ 5.9%  8776/150000 tok │ budget 810/3000",
    "reasoning auto",
    "head draft-mtp native (in model) n-max 3",
]


class BlankShape(unittest.TestCase):
    def test_blank_carries_slot_stats(self):
        # _blank() is the one door every sample() exit passes through: a key absent
        # there is a key nine render sites would have to invent a default for
        self.assertIn("slot_stats", g.LlamaProbe(PORT)._blank())
        self.assertEqual(g.LlamaProbe(PORT)._blank()["slot_stats"], [])


class PerSlotSample(unittest.TestCase):
    def test_one_entry_per_slot_with_its_phase(self):
        pr, _ = _probe(live_slots())
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            d = _tick(pr, clock, 0.0)
        sls = d.get("slot_stats") or []
        self.assertEqual(len(sls), 5, "one entry per slot")
        self.assertEqual([s["id"] for s in sls], [0, 1, 2, 3, 4])
        self.assertEqual([s["phase"] for s in sls],
                         ["generating", "idle", "idle", "idle", "idle"])
        self.assertEqual([s["n_ctx"] for s in sls], [N_CTX] * 5)

    def test_occupancy_counts_the_generated_tokens_once(self):
        # POSITIVE: the recorded slot — n_prompt_tokens already holds prompt + generated
        pr, _ = _probe(live_slots())
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            d = _tick(pr, clock, 0.0)
        sls = d.get("slot_stats") or []
        self.assertEqual(len(sls), 5)
        self.assertEqual(sls[0]["used"], 1294)
        self.assertAlmostEqual(sls[0]["fill"], 1294 / N_CTX)
        self.assertEqual([s["used"] for s in sls[1:]], [985, 1186, 237, 5074])
        # budget is n_decoded + n_remain on the generating slot, None where unlimited
        self.assertEqual(sls[0]["decoded"], 810)
        self.assertEqual(sls[0]["budget"], 810 + 2190)
        self.assertIsNone(sls[1]["budget"])

    def test_occupancy_on_an_old_build_shape(self):
        # NEGATIVE CONTROL: an older build reports the prompt alone in n_prompt_tokens,
        # n_decoded at the top level and an int `state` instead of is_processing. A
        # formula that simply trusts n_prompt_tokens would under-count this slot by 810.
        old = {"id": 0, "state": 3, "n_ctx": 4096, "id_task": 7, "n_prompt_tokens": 484,
               "n_prompt_tokens_processed": 484, "n_prompt_tokens_cache": 0,
               "n_decoded": 810, "params": {}, "next_token": {"n_remain": -1, "n_decoded": 810}}
        pr, _ = _probe([old, _slot(1, 8, 100)])
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            d = _tick(pr, clock, 0.0)
        sls = d.get("slot_stats") or []
        self.assertEqual(len(sls), 2)
        self.assertEqual(sls[0]["phase"], "generating")
        self.assertEqual(sls[0]["used"], 484 + 810)

    def test_rates_are_per_slot_and_reset_on_a_new_task(self):
        slots = live_slots()
        pr, srv = _probe(slots)
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            _generated(slots[0], 100)
            _tick(pr, clock, 0.0)
            _generated(slots[0], 130)
            d = _tick(pr, clock, 1.0)
            sls = d.get("slot_stats") or []
            self.assertEqual(len(sls), 5)
            # 30 tokens in 1.0 s on slot 0; the idle slots have no rate at all
            self.assertAlmostEqual(sls[0]["tg"], 30.0)
            self.assertIsNone(sls[1]["tg"])
            self.assertIsNone(sls[0]["pp"])
            # the aggregate keeps reading the same thing it did before the feature
            self.assertAlmostEqual(d["tg"], 30.0)
            # a NEW request lands on slot 0 (id_task changes, counter restarts): the
            # trail closes, the rate it reached is latched as the LAST one
            slots[0]["id_task"] = 50000
            _generated(slots[0], 5)
            d = _tick(pr, clock, 2.0)
            sls = d["slot_stats"]
            self.assertIsNone(sls[0]["tg"])
            self.assertAlmostEqual(sls[0]["tg_last"], 30.0)

    def test_stale_tick_keeps_slot_facts_and_marks_stale(self):
        slots = live_slots()
        pr, srv = _probe(slots)
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            _generated(slots[0], 100)
            _tick(pr, clock, 0.0)
            _generated(slots[0], 130)
            _tick(pr, clock, 1.0)
            srv.down = True
            d = _tick(pr, clock, 2.0)
        self.assertTrue(d["stale"])
        sls = d.get("slot_stats") or []
        self.assertEqual(len(sls), 5, "last good per-slot facts survive a silent server")
        self.assertEqual(sls[0]["used"], 484 + 130)


class AggregateKv(unittest.TestCase):
    def test_kv_used_counts_every_slot_once(self):
        # the aggregate bar summed n_prompt_tokens AND n_decoded on a build where the
        # first already contains the second: 2104 for a slot holding 1294
        pr, _ = _probe(live_slots())
        clock = [0.0]
        with patch("time.monotonic", lambda: clock[0]):
            d = _tick(pr, clock, 0.0)
        self.assertEqual(d["kv_used"], OCCUPIED)


class PerSlotRender(unittest.TestCase):
    def _new_rows(self, width):
        base = _texts(g._llama_rows(render_state([]), width))
        multi = _texts(g._llama_rows(render_state(stats_fixture()), width))
        return base, [r for r in multi if r not in base]

    def test_multi_slot_draws_one_line_per_slot(self):
        _base, new = self._new_rows(120)
        self.assertTrue(new, "no per-slot lines were drawn for a five-slot server")
        text = "\n".join(new)
        for needle in ("generating", "1294/30208", "985/30208", "1186/30208",
                       "443/30208", "5074/30208"):
            self.assertIn(needle, text)
        line0 = next(r for r in new if "1294/30208" in r)
        self.assertIn("generating", line0)
        self.assertIn("budget 810/3000", line0)
        self.assertIn("gen 31.2 t/s", line0)
        line1 = next(r for r in new if "985/30208" in r)
        self.assertIn("idle", line1)
        self.assertIn("29.8 t/s last", line1)
        line3 = next(r for r in new if "443/30208" in r)
        self.assertIn("prefill", line3)
        self.assertIn("processed 443 tok", line3)
        self.assertIn("prefill 1800 t/s", line3)
        self.assertTrue(new[0].startswith("slots"), new[0])

    def test_rows_fit_the_narrowest_terminal(self):
        # 62 columns is the program's own minimum; a row wider than the box is
        # truncated by _put with an ellipsis, and the rate sits at the end of the row
        width = 62
        inner = width - 4
        base = g._llama_rows(render_state([]), width)
        multi = g._llama_rows(render_state(stats_fixture()), width)
        new = [r for r in multi if r not in base]
        self.assertTrue(new)
        for r in new:
            self.assertLessEqual(g._cellw(r), inner, _texts([r])[0])

    def test_stale_state_draws_no_rate_per_slot(self):
        d = render_state(stats_fixture())
        d["stale"] = True
        base = _texts(g._llama_rows(render_state([]), 120))
        new = [r for r in _texts(g._llama_rows(d, 120)) if r not in base]
        self.assertTrue(new)
        self.assertIn("1294/30208", "\n".join(new))
        self.assertNotIn("t/s", "\n".join(new))

    def test_single_slot_render_is_unchanged(self):
        # REGRESSION GUARD (expected green before and after): with one slot the panel
        # must be byte-identical to what the program printed before this feature
        one = _texts(g._llama_rows(render_state(stats_fixture()[:1]), 120))
        none = _texts(g._llama_rows(render_state([]), 120))
        self.assertEqual(one, GOLDEN_120)
        self.assertEqual(none, GOLDEN_120)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for llamagputop.

Standard library only, like the program itself. Run them with

    python3 -m unittest discover -s tests -v

or run this file directly. No network is touched and no particular hardware is
needed: every reader is driven against a temporary directory that stands in for
sysfs, or against a stub feed.

Almost everything here comes in pairs, and the second half of each pair is the
half that matters. The program's promise is that a missing reading shows as a
dash with a reason and never as a zero, and the hard part of testing that is not
the missing case: it is the zero one. A fan parked at 0 rpm, a power-gated GPU
at 0 MHz, an idle server at 0 tokens per second and a freshly started card with
0 MiB allocated are all real readings that have to survive. A suite that checked
only the first half would pass a change that turns every zero into a dash, which
is the same defect facing the other way.

Each fixture also carries a control that proves it can express presence before
any absence is concluded from it. A fixture that silently produces nothing makes
every "this field is empty" assertion pass for the wrong reason.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "llamagputop.py")


def _load():
    spec = importlib.util.spec_from_file_location("llamagputop_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lgt = _load()


class Base(unittest.TestCase):
    """Shared temporary tree, and a reset of the module-level state.

    The program keeps its history, its last-good sensor cache and its per-port
    probes at module level, which is right for a monitor that runs for hours and
    wrong for tests that must not inherit each other's leftovers.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="llamagputop-test-")
        self._ttl = lgt._SANE_TTL
        lgt.HIST.clear()
        lgt._last_good.clear()
        lgt._probes.clear()
        lgt._port_seen.clear()
        lgt._last_pp.clear()
        lgt._model_ports.clear()
        lgt._CPU_E.update({"t": 0.0, "j": 0, "w": None})

    def tearDown(self):
        lgt._SANE_TTL = self._ttl
        shutil.rmtree(self.root, ignore_errors=True)

    # ---------------------------------------------------------------- fixtures
    def card(self, name, dev=None, hwmon=None, gt=None):
        """A directory that looks enough like a DRM card node to read from.

        `hwmon=None` means the node is absent entirely, which is a different
        state from a node whose optional files are missing, and both occur.
        """
        path = os.path.join(self.root, name)
        devdir = os.path.join(path, "device")
        os.makedirs(devdir, exist_ok=True)
        for fname, value in (dev or {}).items():
            self.write(os.path.join(devdir, fname), value)
        if hwmon is not None:
            hwdir = os.path.join(devdir, "hwmon", "hwmon0")
            os.makedirs(hwdir, exist_ok=True)
            for fname, value in hwmon.items():
                self.write(os.path.join(hwdir, fname), value)
        if gt is not None:
            gtdir = os.path.join(path, "gt", "gt0")
            os.makedirs(gtdir, exist_ok=True)
            for fname, value in gt.items():
                self.write(os.path.join(gtdir, fname), value)
        return path

    @staticmethod
    def write(path, value):
        with open(path, "w") as handle:
            handle.write(str(value))

    @staticmethod
    def rows(sample, width=96):
        return "\n".join("".join(text for text, _attr in row)
                         for row in lgt._gpu_rows(sample, width))


class StubNvtop:
    """Stands in for the memory feed thread.

    by_key maps a device name to (used_mib, total_mib, timestamp, utilisation),
    which is the shape the real reader builds from the tool's JSON output.
    """

    def __init__(self, rows):
        self.tool_ok = True
        self.answered = True
        self.by_key = {name: (used, total, time.monotonic(), util)
                       for name, (used, total, util) in rows.items()}


class StubEngine:
    """Stands in for the engine feed thread."""

    def __init__(self, freq, alive=True):
        self.tool_ok = True
        self.data = {"alive": alive, "ts": time.monotonic(),
                     "engines": {"RCS": 5.0}, "freq": freq}


class StubLlamaFeed:
    """Hands a fixed fleet to the collector so no HTTP request is made."""

    def __init__(self, fleet=()):
        self.data = [dict(entry) for entry in fleet]
        self.at = time.monotonic()


# =============================================================================
class SaneCache(Base):
    """A held reading has to expire, without losing the reason it is held."""

    def test_in_range_value_is_returned(self):
        self.assertEqual(lgt.sane("k", 60, 1, 130), 60)

    def test_rejected_value_with_no_history_is_absent(self):
        self.assertIsNone(lgt.sane("k", 0, 1, 130))

    def test_shipped_window_is_ten_seconds(self):
        self.assertEqual(lgt._SANE_TTL, 10.0)

    def test_value_held_past_the_window_expires(self):
        lgt._SANE_TTL = 0.3
        lgt.sane("k", 60, 1, 130)
        time.sleep(0.45)
        self.assertIsNone(lgt.sane("k", 0, 1, 130))

    def test_value_held_briefly_survives(self):
        """Holding across a bad tick is the point of the cache, not a bug."""
        lgt.sane("k", 60, 1, 130)
        time.sleep(0.05)
        self.assertEqual(lgt.sane("k", 0, 1, 130), 60)

    def test_intermittent_sensor_never_expires(self):
        """A sensor answering every other tick keeps its value indefinitely."""
        lgt._SANE_TTL = 0.3
        seen = []
        for i in range(6):
            seen.append(lgt.sane("k", 60 if i % 2 == 0 else 0, 1, 130))
            time.sleep(0.1)
        self.assertEqual(seen, [60] * 6)

    def test_a_fresh_reading_restarts_the_window(self):
        lgt._SANE_TTL = 0.3
        lgt.sane("k", 60, 1, 130)
        time.sleep(0.2)
        lgt.sane("k", 61, 1, 130)
        time.sleep(0.2)
        self.assertEqual(lgt.sane("k", 0, 1, 130), 61)

    def test_a_good_reading_always_beats_the_cache(self):
        lgt.sane("k", 60, 1, 130)
        self.assertEqual(lgt.sane("k", 42, 1, 130), 42)


# =============================================================================
class EnergyCaches(Base):
    """The two wattage caches carry a timestamp and never used to read it."""

    def gpu_watts(self, gap):
        """Two counter reads produce a rate, then the file vanishes for `gap`.

        The counter starts non-zero on purpose: the rate is only computed once a
        previous reading exists, so a fixture starting at zero would never
        produce a wattage at all and every assertion below would pass without
        testing anything.
        """
        lgt._SANE_TTL = 0.3
        path = self.card("energy", hwmon={"energy1_input": 1000})
        node = lgt._hwmon_of(path)
        gpu = lgt.IntelGpu(path, "Fixture", 0, None, None)
        gpu._power()
        time.sleep(0.45)
        self.write(os.path.join(node, "energy1_input"), 1000000)
        live = gpu._power()
        os.remove(os.path.join(node, "energy1_input"))
        time.sleep(gap)
        return live, gpu._power()

    def cpu_watts(self, gap):
        lgt._SANE_TTL = 0.3
        rapl = os.path.join(self.root, "rapl")
        os.makedirs(rapl, exist_ok=True)
        real = lgt.RAPL_CPU
        lgt.RAPL_CPU = rapl
        try:
            self.write(os.path.join(rapl, "energy_uj"), 1000)
            lgt.cpu_sample()
            time.sleep(0.45)
            self.write(os.path.join(rapl, "energy_uj"), 1000000)
            live = lgt.cpu_sample().get("power")
            os.remove(os.path.join(rapl, "energy_uj"))
            time.sleep(gap)
            return live, lgt.cpu_sample().get("power")
        finally:
            lgt.RAPL_CPU = real

    def test_gpu_fixture_produces_a_real_wattage(self):
        live, _ = self.gpu_watts(0.05)
        self.assertIsNotNone(live)
        self.assertGreater(live, 0.5)

    def test_gpu_wattage_expires(self):
        self.assertIsNone(self.gpu_watts(0.45)[1])

    def test_gpu_wattage_held_briefly_survives(self):
        self.assertIsNotNone(self.gpu_watts(0.05)[1])

    def test_cpu_fixture_produces_a_real_wattage(self):
        live, _ = self.cpu_watts(0.05)
        self.assertIsNotNone(live)
        self.assertGreater(live, 0.5)

    def test_cpu_wattage_expires(self):
        self.assertIsNone(self.cpu_watts(0.45)[1])

    def test_cpu_wattage_held_briefly_survives(self):
        self.assertIsNotNone(self.cpu_watts(0.05)[1])


# =============================================================================
class AmdReader(Base):
    """An absent sysfs file must not read as zero, and a real zero must stay."""

    BARE = {}
    ZEROS = {
        "gpu_busy_percent": 0,            # idle card
        "mem_busy_percent": 0,
        "mem_info_vram_used": 0,          # nothing allocated yet
        "mem_info_vram_total": 12884901888,
        "mem_info_gtt_used": 0,
        "mem_info_gtt_total": 8589934592,
        "pp_dpm_sclk": "0: 500Mhz *\n1: 2450Mhz\n",
        "pp_dpm_mclk": "0: 96Mhz *\n1: 1124Mhz\n",
    }
    ZEROS_HWMON = {
        "temp1_label": "edge", "temp1_input": 41000,
        "power1_average": 8000000,
        "in0_input": 806,
        "fan1_input": 0,                  # zero-rpm mode, a shipped feature
        "pwm1": 0,
    }
    FULL = {"gpu_busy_percent": 37, "mem_busy_percent": 12,
            "mem_info_vram_used": 3221225472, "mem_info_vram_total": 12884901888,
            "mem_info_gtt_used": 1073741824, "mem_info_gtt_total": 8589934592}
    FULL_HWMON = {"temp1_label": "edge", "temp1_input": 63000,
                  "power1_average": 142000000, "fan1_input": 1450, "pwm1": 128}

    def sample(self, tag, dev, hwmon=None):
        return lgt.AmdGpu(self.card(tag, dev=dev, hwmon=hwmon), "Fixture").sample()

    def bare(self):
        return self.sample("bare", self.BARE)

    def zeros(self):
        return self.sample("zeros", self.ZEROS, self.ZEROS_HWMON)

    def full(self):
        return self.sample("full", self.FULL, self.FULL_HWMON)

    # -- the fixture can express presence, so an absence below means something
    def test_fixture_reports_a_real_utilisation(self):
        self.assertEqual(self.full().get("util"), 37)

    def test_fixture_reports_a_real_memory_figure(self):
        self.assertEqual(self.full().get("vram_used"), 3072)

    # -- a card that exposes nothing
    def test_absent_utilisation_is_not_zero(self):
        self.assertIsNone(self.bare().get("util"))

    def test_absent_memory_is_not_zero(self):
        sample = self.bare()
        self.assertIsNone(sample.get("vram_total"))
        self.assertIsNone(sample.get("gtt_total"))
        self.assertIsNone(sample.get("mem_util"))

    def test_every_empty_field_carries_a_reason(self):
        needs = self.bare().get("needs") or {}
        for field in ("util", "vram", "temp", "power", "clocks"):
            self.assertIn(field, needs)

    def test_no_reason_names_a_binary_to_install(self):
        """On this vendor the answer is about the driver, not a missing tool."""
        joined = " ".join((self.bare().get("needs") or {}).values())
        for binary in ("nvidia-smi", "intel_gpu_top", "nvtop"):
            self.assertNotIn(binary, joined)

    def test_reasons_reach_the_panel(self):
        self.assertIn("no data", self.rows(self.bare()))

    def test_no_fan_row_is_invented(self):
        self.assertNotIn("fan ", self.rows(self.bare()))

    # -- a card whose readings are legitimately zero
    def test_idle_utilisation_stays_zero(self):
        self.assertEqual(self.zeros().get("util"), 0)
        self.assertEqual(self.zeros().get("mem_util"), 0)

    def test_unallocated_memory_stays_zero(self):
        sample = self.zeros()
        self.assertEqual(sample.get("vram_used"), 0)
        self.assertEqual(sample.get("vram_total"), 12288)

    def test_a_parked_fan_reads_zero_rather_than_unknown(self):
        self.assertEqual(self.zeros().get("fan_rpm"), 0)
        self.assertEqual(self.zeros().get("fan_pct"), 0)

    def test_a_parked_fan_is_visible_on_the_panel(self):
        self.assertIn("fan ", self.rows(self.zeros()))

    def test_a_spinning_fan_is_still_shown(self):
        self.assertIn("1450 rpm", self.rows(self.full()))

    def test_a_field_that_answered_gets_no_reason(self):
        needs = self.zeros().get("needs") or {}
        for field in ("util", "vram", "temp", "power"):
            self.assertNotIn(field, needs)


# =============================================================================
class IntelReader(Base):
    """Shared memory, a gated clock, and a fallback that never ran."""

    IGPU = "TigerLake-H GT1 (UHD Graphics)"
    DISCRETE = "Intel Arc A770 Graphics"

    def sample(self, tag, feed=None, engine=None, gt=None, hwmon=None):
        path = self.card(tag, gt=gt, hwmon=hwmon)
        return lgt.IntelGpu(path, self.IGPU, 0, engine, feed).sample()

    # -- an integrated GPU has no memory of its own
    def test_a_discrete_card_keeps_its_memory_pair(self):
        feed = StubNvtop({self.DISCRETE: (3000, 16384, 44.0)})
        card = self.card("arc")
        sample = lgt.IntelGpu(card, self.DISCRETE, 0, None, feed).sample()
        self.assertEqual(sample.get("vram_used"), 3000)
        self.assertEqual(sample.get("vram_total"), 16384)
        self.assertIsNone(sample.get("vram_shared"))
        self.assertIn("3000/16384", self.rows(sample))

    def test_module_memory_total_matches_an_independent_read(self):
        """Read here rather than taken from the module, which would prove nothing."""
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    self.assertEqual(lgt.RAM_TOTAL_MIB, int(line.split()[1]) // 1024)
                    return
        self.fail("no MemTotal in /proc/meminfo")

    def test_a_total_equal_to_system_memory_is_marked_shared(self):
        feed = StubNvtop({self.IGPU: (15018, lgt.RAM_TOTAL_MIB, 12.0)})
        sample = self.sample("igpu", feed=feed)
        self.assertTrue(sample.get("vram_shared"))
        self.assertIn(sample.get("vram_total"), (None, 0))
        self.assertEqual((sample.get("needs") or {}).get("vram"),
                         "memory is shared with system RAM")

    def test_a_shared_card_draws_no_memory_bar(self):
        feed = StubNvtop({self.IGPU: (15018, lgt.RAM_TOTAL_MIB, 12.0)})
        rendered = self.rows(self.sample("igpu-bar", feed=feed))
        self.assertNotIn("MiB", rendered)
        self.assertIn("shared with system RAM", rendered)

    def test_the_older_signature_still_routes_the_same_way(self):
        """A total with no used figure was the case the first guard covered."""
        feed = StubNvtop({self.IGPU: (None, lgt.RAM_TOTAL_MIB, 12.0)})
        self.assertTrue(self.sample("igpu-noused", feed=feed).get("vram_shared"))

    # -- a power-gated card really does report zero
    def test_a_gated_clock_reads_zero(self):
        sample = self.sample("gated", gt={"rps_act_freq_mhz": 0})
        self.assertEqual(sample.get("sclk"), 0)
        self.assertIn("core 0 MHz", self.rows(sample))

    def test_an_absent_clock_file_is_not_a_zero(self):
        sample = self.sample("noclock", gt={})
        self.assertIsNone(sample.get("sclk"))
        self.assertNotIn("core ", self.rows(sample))

    def test_a_normal_clock_is_unaffected(self):
        self.assertEqual(self.sample("clk", gt={"rps_act_freq_mhz": 1350}).get("sclk"),
                         1350)

    def test_an_out_of_range_clock_is_still_refused(self):
        self.assertIsNone(self.sample("wild", gt={"rps_act_freq_mhz": 99999}).get("sclk"))

    # -- the fallback that a present-but-empty key used to disable
    def test_sysfs_fills_in_when_the_feed_reports_no_frequency(self):
        sample = self.sample("fb0", engine=StubEngine(0),
                             gt={"rps_act_freq_mhz": 1350})
        self.assertEqual(sample.get("sclk"), 1350)

    def test_sysfs_fills_in_when_the_feed_frequency_is_absent(self):
        sample = self.sample("fbn", engine=StubEngine(None),
                             gt={"rps_act_freq_mhz": 1350})
        self.assertEqual(sample.get("sclk"), 1350)

    def test_a_live_feed_frequency_still_wins(self):
        sample = self.sample("fbw", engine=StubEngine(800),
                             gt={"rps_act_freq_mhz": 1350})
        self.assertEqual(sample.get("sclk"), 800)

    def test_a_parked_fan_reads_zero_here_too(self):
        sample = self.sample("fan", hwmon={"fan1_input": 0, "pwm1": 0,
                                           "temp1_input": 44000})
        self.assertEqual(sample.get("fan_rpm"), 0)


# =============================================================================
class StatusLine(Base):
    """The one-line output has to keep the same promise as the panel."""

    CPU = {"util": 12.0, "temp": 64, "power": 31.0, "rapl_present": True}
    MEM = {"free": 6717, "total": 31864}

    def line(self, gpu, cpu=None):
        return lgt._text_line([gpu], dict(cpu or self.CPU), self.MEM, [])

    def segment(self, out, prefix):
        """Assertions are scoped to one segment, never to the whole line.

        A substring over a line that concatenates six independent readings
        cannot say which reading it matched, so it is not an assertion about
        the field.
        """
        for part in out.split(" | ")[1:]:
            if part.startswith(prefix):
                return part
        self.fail("no segment starting with %r in %r" % (prefix, out))

    def gpu_segment(self, **fields):
        base = {"vendor": "Intel", "util": 1.0, "temp_main": 40, "power": 5.0}
        base.update(fields)
        return self.segment(self.line(base), base["vendor"])

    def test_absent_power_is_a_dash(self):
        seg = self.gpu_segment(power=None, needs={"power": "no energy counter"})
        self.assertIn("—W", seg)
        self.assertNotIn("0W", seg)

    def test_a_real_zero_power_stays_zero(self):
        self.assertIn("0W", self.gpu_segment(power=0.0))

    def test_absent_utilisation_is_a_dash(self):
        self.assertIn("util —%", self.gpu_segment(util=None))

    def test_a_real_zero_utilisation_stays_zero(self):
        self.assertIn("util 0%", self.gpu_segment(util=0.0))

    def test_absent_cpu_utilisation_is_a_dash(self):
        cpu = dict(self.CPU, util=None)
        seg = self.segment(self.line({"vendor": "Intel", "util": 1.0,
                                      "temp_main": 40, "power": 5.0}, cpu), "CPU")
        self.assertIn("CPU —%", seg)

    def test_a_real_zero_cpu_utilisation_stays_zero(self):
        cpu = dict(self.CPU, util=0.0)
        seg = self.segment(self.line({"vendor": "Intel", "util": 1.0,
                                      "temp_main": 40, "power": 5.0}, cpu), "CPU")
        self.assertIn("CPU 0%", seg)

    def test_absent_memory_used_is_a_dash(self):
        seg = self.gpu_segment(vram_used=None, vram_total=6144)
        self.assertIn("vram —/6144", seg)

    def test_a_real_zero_memory_used_stays_zero(self):
        self.assertIn("vram 0/6144", self.gpu_segment(vram_used=0, vram_total=6144))

    def test_no_memory_total_renders_a_bare_dash(self):
        self.assertIn("vram -", self.gpu_segment())

    def test_absent_temperature_stays_a_dash(self):
        self.assertIn("-C", self.gpu_segment(temp_main=None))

    def test_absent_cpu_power_stays_a_dash(self):
        cpu = dict(self.CPU, power=None)
        seg = self.segment(self.line({"vendor": "Intel", "util": 1.0,
                                      "temp_main": 40, "power": 5.0}, cpu), "CPU")
        self.assertIn("64C -", seg)


# =============================================================================
class PortFocus(Base):
    """The port argument selects one server, and says so when it is silent."""

    DISCOVERED = [
        {"pid": "101", "port": "7795", "host": "127.0.0.1", "model_hint": "a"},
        {"pid": "102", "port": "7797", "host": "127.0.0.1", "model_hint": "b"},
        {"pid": "103", "port": "7799", "host": "127.0.0.1", "model_hint": "c"},
        {"pid": "104", "port": "8181", "host": "127.0.0.1", "model_hint": "d"},
    ]

    def setUp(self):
        super().setUp()
        self._discover = lgt.discover_llama_servers
        self._sample = lgt.LlamaProbe.sample
        lgt.discover_llama_servers = lambda: [dict(s) for s in self.DISCOVERED]
        lgt.LlamaProbe.sample = lambda self: {"alive": True, "phase": "idle",
                                              "model": ""}

    def tearDown(self):
        lgt.discover_llama_servers = self._discover
        lgt.LlamaProbe.sample = self._sample
        super().tearDown()

    def ports(self, arg):
        lgt._probes.clear()
        return [entry["port"] for entry in lgt.sample_llama_fleet(arg)]

    def test_the_stub_discovers_the_whole_fleet(self):
        self.assertEqual(self.ports(None), ["7795", "7797", "7799", "8181"])

    def test_a_discovered_port_focuses_to_one(self):
        self.assertEqual(self.ports("7795"), ["7795"])

    def test_an_undiscovered_port_is_still_probed_alone(self):
        self.assertEqual(self.ports("65535"), ["65535"])

    def test_focusing_clears_the_multi_server_flag(self):
        lgt._probes.clear()
        self.assertEqual([e.get("multi") for e in lgt.sample_llama_fleet("7795")],
                         [False])

    def test_no_argument_keeps_the_multi_server_flag(self):
        lgt._probes.clear()
        self.assertEqual([e.get("multi") for e in lgt.sample_llama_fleet(None)],
                         [True] * 4)

    def test_an_integer_port_matches_the_string_ports(self):
        """Discovery yields strings; comparing the wrong types would match none."""
        self.assertEqual(self.ports(7795), ["7795"])


class LoneSilentServer(Base):
    """Skipping a silent server is right only while another one is answering."""

    DEAD = {"alive": False, "phase": "off", "stale": True, "port": "65535",
            "model": ""}
    ALIVE = {"alive": True, "phase": "idle", "port": "7795", "model": "m",
             "pp": None, "tg": None, "pp_last": None, "tg_last": None,
             "multi": True}
    CPU = {"util": 1.0, "temp": 50, "power": 10.0, "rapl_present": True}
    MEM = {"free": 1, "total": 2}

    def line(self, fleet):
        return lgt._text_line([], dict(self.CPU), self.MEM, fleet)

    def test_a_lone_silent_server_is_reported(self):
        out = self.line([self.DEAD])
        self.assertIn("65535", out)
        self.assertIn("did not answer", out)

    def test_a_silent_server_beside_a_live_one_is_skipped(self):
        out = self.line([self.ALIVE, self.DEAD])
        self.assertNotIn("65535", out)
        self.assertIn("7795", out)

    def test_an_empty_fleet_adds_nothing(self):
        self.assertNotIn("llama", self.line([]))


# =============================================================================
class TrendSeries(Base):
    """A dot on a trend row means a true zero and nothing else."""

    class FakeGpu:
        vendor = "NVIDIA"

        def __init__(self, payload):
            self.payload = payload

        def sample(self):
            return dict(self.payload)

    def series(self, key, values):
        for value in values:
            lgt.record(key, value)
        return list(lgt.HIST.get(key, []))

    def collect_util(self, payload, ticks=4):
        gpu = self.FakeGpu(dict(payload, name="Fixture", temp={}, temp_crit={}))
        for _ in range(ticks):
            lgt._collect([gpu], None, StubLlamaFeed())
        return list(lgt.HIST.get("gpu0_util", []))

    def test_real_values_are_kept(self):
        self.assertEqual(self.series("a", [10.0, 20.0]), [10.0, 20.0])

    def test_a_non_finite_value_is_refused(self):
        self.assertEqual(self.series("b", [5.0, float("nan")]), [5.0, 5.0])

    def test_an_absent_value_holds_the_previous_one(self):
        self.assertEqual(self.series("c", [62.0, None]), [62.0, 62.0])

    def test_an_absent_value_with_no_history_records_nothing(self):
        self.assertEqual(self.series("d", [None, None]), [])

    def test_a_card_with_no_source_charts_nothing(self):
        self.assertEqual(
            self.collect_util({"util": None, "needs": {"util": "needs nvidia-smi"}}),
            [])

    def test_a_real_zero_is_charted(self):
        self.assertEqual(self.series("e", [0.0, 0.0]), [0.0, 0.0])

    def test_a_card_reporting_zero_charts_zeros(self):
        self.assertEqual(self.collect_util({"util": 0.0}), [0] * 4)

    def test_a_card_reporting_a_rate_charts_it(self):
        self.assertEqual(self.collect_util({"util": 62.0}), [62.0] * 4)

    def test_a_real_zero_between_real_values_survives(self):
        self.assertEqual(self.series("f", [50.0, 0.0, 50.0]), [50.0, 0.0, 50.0])


class ServerSeries(Base):
    """Idle and silent are different facts about a server."""

    IDLE = {"alive": True, "phase": "idle", "port": "7795", "model": "m",
            "tg": None, "pp": None, "pp_last": None, "tg_last": None, "kv": None}

    def series(self, entry, ticks=3, port="7795"):
        for _ in range(ticks):
            lgt._collect([], None, StubLlamaFeed([entry]))
        return list(lgt.HIST.get("tg_series@%s" % port, []))

    def test_an_idle_server_charts_a_true_zero(self):
        self.assertEqual(self.series(self.IDLE), [0] * 3)

    def test_a_generating_server_charts_its_rate(self):
        busy = dict(self.IDLE, phase="generating", tg=28.7)
        self.assertEqual(self.series(busy), [28.7] * 3)

    def test_a_silent_server_charts_nothing(self):
        silent = dict(self.IDLE, phase="not answering", stale=True)
        self.assertEqual(self.series(silent), [])


class SeriesRetirement(Base):
    """A port that is gone takes all of its series with it."""

    DEAD, LIVE = 59991, 59992

    def retire(self):
        now = time.monotonic()
        for port in (self.DEAD, self.LIVE):
            lgt.record("kv_series@%s" % port, 42.0)
            lgt.record("tg_series@%s" % port, 1.0)
        lgt._port_seen[self.DEAD] = now - 300
        lgt._port_seen[self.LIVE] = now
        lgt._collect([], None, StubLlamaFeed())

    def test_the_existing_series_of_a_retired_port_are_dropped(self):
        self.retire()
        self.assertNotIn("tg_series@%s" % self.DEAD, lgt.HIST)

    def test_the_cache_series_of_a_retired_port_is_dropped_too(self):
        self.retire()
        self.assertNotIn("kv_series@%s" % self.DEAD, lgt.HIST)

    def test_a_live_port_keeps_its_series(self):
        self.retire()
        self.assertIn("kv_series@%s" % self.LIVE, lgt.HIST)


if __name__ == "__main__":
    unittest.main(verbosity=2)

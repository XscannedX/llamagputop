#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
llamagputop is a terminal GPU and llama.cpp inference monitor for Linux.

What makes it different from other monitors is that besides watching every GPU in the machine, it also reads a running llama.cpp server and shows what actually matters when you serve a model. You get to see the prefill and generation speed, KV cache fill, and speculative decoding metrics like draft head usage and acceptance rates. Those inference stats are tied to the model, so when the model changes they reset.

The design principles are simple. Read at the source. Everything comes from sysfs, proc and the driver, plus a couple of optional helpers used only when present. There are no Python dependencies beyond the standard library. No sensor is believed blindly. Out of range readings are dropped and the last good value is kept. The tool degrades honestly. A missing card or counter shows a dash with a reason, never a fake number.

It is highly portable. It discovers the hardware at runtime, so it runs on any Linux box, not just the one it was written on.

Copyright 2026 XscannedX <xscannedx@gmail.com>. MIT License.
"""
import glob
import math
import os
import re
import struct
import subprocess
import sys
import threading
import time
from collections import deque

# ------------------------------------------------------------------ small utils
def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def read_int(path, default=0):
    v = read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


_last_good = {}


def sane(key, value, lo=None, hi=None):
    """A sensor that answers is not a sensor that tells the truth. Out-of-range
    values are dropped in favour of the last good one; if there never was one,
    None — which the display turns into a dash, not a misleading zero."""
    if value is not None and (lo is None or value >= lo) and (hi is None or value <= hi):
        _last_good[key] = value
        return value
    return _last_good.get(key)


def which(name):
    return any(os.access(os.path.join(p, name), os.X_OK)
               for p in os.environ.get("PATH", "").split(os.pathsep) if p)


# ------------------------------------------------------------------ PCIe chain
# The real link a card has to the rest of the system is NOT the card's own node:
# both may sit behind a switch whose inner segment reads wider than the negotiated
# link to the chipset. What matters is the narrowest hop between the card and the
# root, and the root port itself — the only stable value. Re-read every tick; it
# rises under load and drops back at idle.
_PCIE_EFF = {2.5: 0.8, 5.0: 0.8, 8.0: 128 / 130, 16.0: 128 / 130,
             32.0: 128 / 130, 64.0: 242 / 256}


def _pcie_link(path):
    v = read(f"{path}/current_link_speed") or ""
    m = re.match(r"([\d.]+)\s*GT/s", v)
    w = read_int(f"{path}/current_link_width", 0)
    return (float(m.group(1)), w) if m and w else None


def _pcie_link_max(path):
    v = read(f"{path}/max_link_speed") or ""
    m = re.match(r"([\d.]+)\s*GT/s", v)
    w = read_int(f"{path}/max_link_width", 0)
    return (float(m.group(1)), w) if m and w else None


def pcie_chain(dev_path):
    nodes = []
    p = os.path.realpath(dev_path)
    for _ in range(8):
        cur = _pcie_link(p)
        if cur:
            nodes.append((cur, _pcie_link_max(p) or cur))
        parent = os.path.dirname(p)
        if not re.search(r"/\d{4}:[0-9a-f]{2}:", p) or parent == p:
            break
        p = parent
    if not nodes:
        return None
    band = lambda gw: gw[0] * gw[1] * _PCIE_EFF.get(gw[0], 128 / 130) / 8
    narrow = min(nodes, key=lambda n: band(n[0]))
    root = nodes[-1]
    return {"gts": root[0][0], "width": root[0][1], "gbs": band(root[0]),
            "narrow_gbs": band(narrow[0]), "bottleneck": band(narrow[0]) < band(root[0]) * 0.9,
            "narrow_gts": narrow[0][0], "narrow_width": narrow[0][1]}


def pcie_text(link):
    if not link:
        return ""
    t = f"PCIe {link['gts']:g} GT/s x{link['width']} · {link['gbs']:.1f} GB/s"
    if link["bottleneck"]:
        t += f" (link {link['narrow_gts']:g}x{link['narrow_width']})"
    return t


# ---------------------------------------------------------- PCI product naming
_PCI_IDS_PATHS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids",
                  "/usr/share/pci.ids")
_VENDOR = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}


def _name_from_lspci(pci_addr):
    """lspci -mm gives the marketing device string, e.g. 'Navi 22 [Radeon RX 6750
    XT]'. It is the most human-readable source and needs no ID database."""
    if not which("lspci"):
        return None
    try:
        out = subprocess.run(["lspci", "-mm", "-s", pci_addr], capture_output=True,
                             text=True, timeout=3).stdout.strip()
    except Exception:
        return None
    # fields are quoted: slot "class" "vendor" "device" ...
    fields = re.findall(r'"([^"]*)"', out)
    return fields[2] if len(fields) >= 3 else None


_pci_ids_cache = {}


def _name_from_pci_ids(vendor_id, device_id):
    """Fallback when lspci is missing: look the device up in the pci.ids database
    that ships with most distributions."""
    vid = vendor_id.replace("0x", "").lower()
    did = device_id.replace("0x", "").lower()
    path = next((p for p in _PCI_IDS_PATHS if os.path.isfile(p)), None)
    if not path:
        return None
    try:
        cur_vendor = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                if not line.startswith("\t"):
                    cur_vendor = line[:4]
                elif line.startswith("\t") and not line.startswith("\t\t") and cur_vendor == vid:
                    if line[1:5] == did:
                        return line[5:].strip()
    except Exception:
        return None
    return None


def gpu_product_name(dev_path, pci_addr, vendor_id):
    name = _name_from_lspci(pci_addr)
    if name:
        return re.sub(r"\s+", " ", name).strip()
    did = read(f"{dev_path}/device")
    if did:
        name = _name_from_pci_ids(vendor_id, did)
        if name:
            return name
    return f"{_VENDOR.get(vendor_id, 'Unknown')} GPU"


# ----------------------------------------------------------- Vulkan device order
_vulkan_map = None


def vulkan_device_map():
    """Map each GPU's PCI address to its Vulkan device index. vulkaninfo enumerates
    devices in the same order the ggml Vulkan backend does, so index N here is the
    'VulkanN' that appears on a llama.cpp `-dev`/`-ts` command line — which is how you
    tell which physical card llama assigned to which slot. Cached; {} without vulkaninfo
    or on any other backend (CUDA/ROCm/SYCL name their devices differently)."""
    global _vulkan_map
    if _vulkan_map is not None:
        return _vulkan_map
    _vulkan_map = {}
    if not which("vulkaninfo"):
        return _vulkan_map
    try:
        out = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return _vulkan_map
    dom = bus = dev = None
    for line in out.splitlines():
        m = re.search(r"pciDomain\s*=\s*(\d+)", line)
        if m:
            dom = int(m.group(1)); continue
        m = re.search(r"pciBus\s*=\s*(\d+)", line)
        if m:
            bus = int(m.group(1)); continue
        m = re.search(r"pciDevice\s*=\s*(\d+)", line)
        if m:
            dev = int(m.group(1)); continue
        m = re.search(r"pciFunction\s*=\s*(\d+)", line)
        if m and None not in (dom, bus, dev):
            addr = f"{dom:04x}:{bus:02x}:{dev:02x}.{int(m.group(1)):x}"
            if addr not in _vulkan_map:           # first occurrence = device order
                _vulkan_map[addr] = len(_vulkan_map)
            dom = bus = dev = None
    return _vulkan_map


# -------------------------------------------------------------------- AMD GPUs
# gpu_metrics is a small binary blob amdgpu exposes with data hwmon does not have
# (VRM temperatures, real memory clock, and the throttle reason). Its layout is
# versioned in the header, so we read the version and only decode what we know.
_AMD_THROTTLE = {
    0: "board power (PPT0)", 1: "power (PPT1)", 2: "power (PPT2)", 3: "power (PPT3)",
    16: "gpu current", 17: "soc current", 18: "mem current", 19: "vdd current",
    32: "gpu temp", 33: "core temp", 34: "mem temp", 35: "edge temp",
    36: "junction temp", 37: "soc temp", 38: "vrm gpu temp", 39: "vrm soc temp",
    40: "vrm mem temp", 46: "cpu prochot", 47: "gpu prochot",
}


def _amd_gpu_metrics(card):
    try:
        with open(f"{card}/device/gpu_metrics", "rb") as f:
            d = f.read()
    except Exception:
        return {}
    if len(d) < 8:
        return {}
    fmt_rev, content_rev = d[2], d[3]      # header: [size_lo, size_hi, format, content]
    out = {"metrics_version": f"{fmt_rev}.{content_rev}"}
    # v1.x (Navi2x/RDNA2) layout — the fields we use sit at stable offsets.
    if fmt_rev == 1 and len(d) >= 120:
        u16 = lambda o: struct.unpack_from("<H", d, o)[0]
        u32 = lambda o: struct.unpack_from("<I", d, o)[0]
        u64 = lambda o: struct.unpack_from("<Q", d, o)[0]
        good = lambda x: None if x in (0xFFFF, 0) else x
        out["mem_clock"] = good(u16(58))
        out["gfx_clock"] = good(u16(54))
        out["vr_gfx_mv"] = good(u16(10))          # gfx VRM voltage
        out["vr_soc_mv"] = good(u16(12))          # soc VRM voltage
        media = u16(20)                           # media engine activity — 0 is valid
        out["media_activity"] = None if media == 0xFFFF else media
        bits = u64(112)                           # indep_throttle_status, 64-bit
        if bits in (0, 0xFFFFFFFFFFFFFFFF):
            bits = u32(68)                         # older 32-bit field as fallback
        if bits == 0xFFFFFFFF:
            bits = 0
        out["throttle"] = [_AMD_THROTTLE.get(i, f"bit {i}") for i in range(64) if bits >> i & 1]
    return out


def _hwmon_of(card):
    hw = glob.glob(f"{card}/device/hwmon/hwmon*")
    return hw[0] if hw else None


def _hwmon_temp_labels(hw):
    out = {}
    for f in sorted(glob.glob(f"{hw}/temp*_label")) if hw else []:
        n = os.path.basename(f).replace("_label", "")
        out[read(f) or n] = n
    return out


class AmdGpu:
    vendor = "AMD"

    def __init__(self, card, name):
        self.card = card
        self.name = name
        self.dev = f"{card}/device"
        self.hw = _hwmon_of(card)
        self.temp_labels = _hwmon_temp_labels(self.hw)
        # per-sensor critical/emergency thresholds declared by the driver — better
        # than an invented number for colouring the temperatures
        self.temp_crit = {}
        for label, key in self.temp_labels.items():
            self.temp_crit[label] = (read_int(f"{self.hw}/{key}_crit") // 1000,
                                      read_int(f"{self.hw}/{key}_emergency") // 1000)
        self._guide = ("junction" if "junction" in self.temp_labels
                       else next(iter(self.temp_labels), None))

    def sample(self):
        d = {"vendor": "AMD", "name": self.name, "temp_crit": self.temp_crit}
        mib = lambda f: read_int(f"{self.dev}/{f}") // 1048576
        d["vram_used"] = mib("mem_info_vram_used")
        d["vram_total"] = mib("mem_info_vram_total")
        d["gtt_used"] = mib("mem_info_gtt_used")
        d["gtt_total"] = mib("mem_info_gtt_total")
        d["vram_free"] = (d["vram_total"] - d["vram_used"]) if d["vram_total"] else None
        d["util"] = read_int(f"{self.dev}/gpu_busy_percent")
        d["mem_util"] = read_int(f"{self.dev}/mem_busy_percent")

        def dpm(f):
            for line in (read(f"{self.dev}/{f}") or "").splitlines():
                if "*" in line:
                    m = re.search(r"(\d+)\s*Mhz", line, re.I)
                    if m:
                        return int(m.group(1))
            return None
        d["sclk"] = sane(f"{self.card}.sclk", dpm("pp_dpm_sclk"), 1, 6000)
        d["mclk"] = sane(f"{self.card}.mclk", dpm("pp_dpm_mclk"), 1, 6000)
        d["fclk"] = sane(f"{self.card}.fclk", dpm("pp_dpm_fclk"), 1, 6000)
        d["socclk"] = sane(f"{self.card}.socclk", dpm("pp_dpm_socclk"), 1, 6000)
        d["temp"] = {}
        for label, key in self.temp_labels.items():
            d["temp"][label] = sane(f"{self.card}.t.{label}",
                                    read_int(f"{self.hw}/{key}_input") // 1000, 1, 130)
        d["temp_main"] = d["temp"].get(self._guide) if self._guide else None
        if self.hw:
            d["power"] = sane(f"{self.card}.w",
                              read_int(f"{self.hw}/power1_average") / 1e6, 0.1, 800)
            d["power_cap"] = read_int(f"{self.hw}/power1_cap") // 1000000 or None
            # vddgfx has been seen at 6 mV while the card drew 8 W: below 400 it is
            # not a reading
            d["voltage"] = sane(f"{self.card}.mv", read_int(f"{self.hw}/in0_input"), 400, 1400)
            d["fan_rpm"] = read_int(f"{self.hw}/fan1_input") or None
            _pwm = read_int(f"{self.hw}/pwm1", -1)      # 0–255 duty cycle → percent
            d["fan_pct"] = round(_pwm * 100 / 255) if _pwm >= 0 else None
        d.update(_amd_gpu_metrics(self.card))
        if d.get("mem_clock"):
            d["mclk"] = d["mem_clock"]
        if d.get("gfx_clock"):
            d["sclk"] = d["gfx_clock"]
        d["pcie"] = pcie_chain(self.dev)
        return d


# ------------------------------------------------------------------ Intel GPUs
# The i915/xe drivers expose frequencies and an energy counter in sysfs; there is
# no wattmeter, so power is the derivative of the energy counter (same trick nvtop
# uses). Engine utilisation and VRAM are not in sysfs: intel_gpu_top has the
# engines, nvtop has the VRAM. Both are optional; without them those fields are
# dashes, everything else still works.
class _IntelEngineFeed(threading.Thread):
    def __init__(self, card_index):
        super().__init__(daemon=True)
        self.card_index = card_index
        self.data = {"engines": {}, "freq": 0, "alive": False, "ts": 0.0}
        self.stop = False

    def run(self):
        if not which("intel_gpu_top"):
            return
        while not self.stop:
            try:
                p = subprocess.Popen(
                    ["intel_gpu_top", "-d", f"drm:/dev/dri/card{self.card_index}",
                     "-l", "-s", "1000"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            except Exception:
                return
            names = []
            try:
                for line in p.stdout:
                    if self.stop:
                        break
                    # engine names in the header. Modern intel_gpu_top prints them bare
                    # ("RCS BCS VCS VECS CCS"); older builds used an "RCS/0" suffix — accept both.
                    t = re.findall(r"\b(RCS|BCS|VCS|VECS|CCS)(?:/\d+)?\b", line)
                    if t:
                        names = t
                        continue
                    cols = line.split()
                    if len(cols) < 6 or not re.match(r"^\d+$", cols[0]):
                        continue
                    try:
                        engines = {}
                        for i, nm in enumerate(names):
                            j = 4 + i * 3
                            if j < len(cols):
                                engines[nm] = float(cols[j])
                        self.data = {"engines": engines, "freq": int(cols[1]),
                                     "alive": True, "ts": time.monotonic()}
                    except (ValueError, IndexError):
                        continue
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
            time.sleep(3.0)


class _NvtopVramFeed(threading.Thread):
    """nvtop (built with CAP_PERFMON) is the only reader that sees Intel Arc VRAM.
    It is slow to spawn, so it runs in its own thread and the panel takes the last
    value without ever waiting."""
    def __init__(self):
        super().__init__(daemon=True)
        self.by_key = {}      # device_name substring -> (used_mib, total_mib, ts)
        self.stop = False

    def run(self):
        import json
        if not which("nvtop"):
            return
        while not self.stop:
            try:
                out = subprocess.run(["nvtop", "-s"], capture_output=True, text=True,
                                     timeout=8).stdout
                for g in json.loads(out):
                    nm = g.get("device_name", "")
                    mu, mt = g.get("mem_used"), g.get("mem_total")
                    gu = g.get("gpu_util")               # e.g. "44%"
                    util = None
                    if isinstance(gu, str):
                        try:
                            util = float(gu.rstrip("%").strip())
                        except ValueError:
                            util = None
                    self.by_key[nm] = (
                        int(mu) // 1048576 if mu else None,
                        int(mt) // 1048576 if mt else 0, time.monotonic(), util)
            except Exception:
                pass
            time.sleep(2.0)


class IntelGpu:
    vendor = "Intel"

    def __init__(self, card, name, card_index, engine_feed, vram_feed):
        self.card = card
        self.name = name
        self.dev = f"{card}/device"
        self.hw = _hwmon_of(card)
        self.gt = f"{card}/gt/gt0" if os.path.isdir(f"{card}/gt/gt0") else None
        self.card_index = card_index
        self.engine_feed = engine_feed
        self.vram_feed = vram_feed
        self._energy = {"t": 0.0, "j": 0, "w": None}

    def _power(self):
        if not self.hw:
            return None
        j = read_int(f"{self.hw}/energy1_input", -1)
        if j < 0:
            return self._energy["w"]
        t = time.monotonic()
        if self._energy["j"] and t > self._energy["t"]:
            dt = t - self._energy["t"]
            if dt > 0.4:
                self._energy["w"] = max(0.0, (j - self._energy["j"]) / dt / 1e6)
                self._energy["t"], self._energy["j"] = t, j
        else:
            self._energy["t"], self._energy["j"] = t, j
        return self._energy["w"]

    def sample(self):
        d = {"vendor": "Intel", "name": self.name}
        ef = self.engine_feed.data if self.engine_feed else {}
        if ef.get("alive") and time.monotonic() - ef.get("ts", 0) < 6:
            eng = ef["engines"]
            # utilisation = the busiest engine (RCS render / CCS compute are what LLM work hits)
            d["util"] = max(list(eng.values()) or [0])
            d["engines"] = eng
            d["sclk"] = ef.get("freq") or None
        if self.gt:
            d.setdefault("sclk", sane(f"{self.card}.f",
                                      read_int(f"{self.gt}/rps_act_freq_mhz"), 1, 4000) or None)
            d["freq_min"] = read_int(f"{self.gt}/rps_RPn_freq_mhz") or None
            d["freq_eff"] = read_int(f"{self.gt}/rps_RP1_freq_mhz") or None
            d["freq_max"] = read_int(f"{self.gt}/rps_RP0_freq_mhz") or None
            d["freq_req"] = read_int(f"{self.gt}/punit_req_freq_mhz") or None
            d["throttle"] = [t for f, t in
                             (("pl1", "PL1 power"), ("pl2", "PL2 power"), ("pl4", "PL4 current"),
                              ("thermal", "thermal"), ("ratl", "RATL avg temp"),
                              ("vr_tdc", "VRM current"), ("vr_thermalert", "VRM thermal"),
                              ("prochot", "prochot"))
                             if read_int(f"{self.gt}/throttle_reason_{f}")]
        st = read(f"{self.card}/power/runtime_status", "")
        d["suspended"] = st == "suspended"
        if self.hw:
            d["temp_main"] = sane(f"{self.card}.t",
                                  read_int(f"{self.hw}/temp1_input") // 1000, 1, 130)
            d["temp"] = {"gpu": d["temp_main"]} if d["temp_main"] is not None else {}
            d["fan_rpm"] = read_int(f"{self.hw}/fan1_input") or None
            _pwm = read_int(f"{self.hw}/pwm1", -1)      # 0–255 duty cycle → percent
            d["fan_pct"] = round(_pwm * 100 / 255) if _pwm >= 0 else None
            d["power"] = self._power()
            d["power_cap"] = read_int(f"{self.hw}/power1_max") // 1000000 or None
        d["pcie"] = pcie_chain(self.dev)
        # VRAM from nvtop if available
        if self.vram_feed:
            for nm, val in self.vram_feed.by_key.items():
                used, total, ts, nvutil = val
                if ("Arc" in nm or "DG2" in nm or "Intel" in nm) and time.monotonic() - ts < 8:
                    d["vram_used"], d["vram_total"] = used, total
                    if d.get("util") is None and nvutil is not None:
                        d["util"] = nvutil       # fallback when intel_gpu_top can't read the PMU
                    break
        return d


# ----------------------------------------------------------------- NVIDIA GPUs
# NVIDIA does not expose the useful counters in sysfs; nvidia-smi is the source.
# One background query per second feeds every NVIDIA card at once.
_NVSMI_FIELDS = ("index", "name", "utilization.gpu", "utilization.memory",
                 "memory.used", "memory.total", "temperature.gpu", "power.draw",
                 "power.limit", "clocks.sm", "clocks.mem")


class _LlamaFeed(threading.Thread):
    """llama.cpp is read over HTTP, and that read does not belong on the draw path.
    Every other reader in this file is either instant (sysfs, /proc) or already threaded
    (_IntelEngineFeed, _NvtopVramFeed, _NvidiaFeed); the llama probe alone blocked the
    key/redraw loop for up to servers × endpoints × timeout, which is why its numbers
    lagged behind every other stat on screen while the rest kept ticking once a second.
    MEASURED 2026-08-25: :8181 answered NOTHING for the 78 s of a CPU prefill (three 20 s
    timeouts back to back), so the panel could not have been kept live from the draw
    loop at any timeout setting. Here a silent server costs the interface nothing: the
    loop reads whatever snapshot is on hand and never waits for it."""
    def __init__(self, port=None):
        super().__init__(daemon=True)
        self.port = port
        self.data = []
        self.at = 0.0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                self.data = sample_llama_fleet(self.port)
                self.at = time.monotonic()
            except Exception:
                pass
            time.sleep(0.4)


class _NvidiaFeed(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.by_index = {}
        self.stop = False

    def run(self):
        if not which("nvidia-smi"):
            return
        query = "--query-gpu=" + ",".join(_NVSMI_FIELDS)
        while not self.stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    c = [x.strip() for x in line.split(",")]
                    if len(c) < len(_NVSMI_FIELDS):
                        continue
                    def num(x):
                        try:
                            return float(x)
                        except ValueError:
                            return None
                    self.by_index[int(c[0])] = {
                        "name": c[1], "util": num(c[2]), "mem_util": num(c[3]),
                        "vram_used": num(c[4]), "vram_total": num(c[5]),
                        "temp_main": num(c[6]), "power": num(c[7]), "power_cap": num(c[8]),
                        "sclk": num(c[9]), "mclk": num(c[10]), "ts": time.monotonic()}
            except Exception:
                pass
            time.sleep(1.0)


class NvidiaGpu:
    vendor = "NVIDIA"

    def __init__(self, name, index, feed):
        self.name = name
        self.index = index
        self.feed = feed

    def sample(self):
        d = {"vendor": "NVIDIA", "name": self.name}
        f = self.feed.by_index.get(self.index)
        if f and time.monotonic() - f.get("ts", 0) < 6:
            d.update({k: f.get(k) for k in
                      ("util", "mem_util", "vram_used", "vram_total", "temp_main",
                       "power", "power_cap", "sclk", "mclk")})
            d["temp"] = {"gpu": d.get("temp_main")} if d.get("temp_main") is not None else {}
        return d


# ------------------------------------------------------------- GPU discovery
def discover_gpus():
    """Enumerate every render node under /sys/class/drm and build the right kind of
    monitor for each driver. NVIDIA cards are correlated to nvidia-smi by index."""
    gpus, feeds = [], {}
    intel_vram = None
    nvidia_feed = None
    nvidia_seen = 0
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):        # connector, not a card
            continue
        dev = f"{card}/device"
        drv_link = f"{dev}/driver"
        if not os.path.exists(drv_link):
            continue
        driver = os.path.basename(os.path.realpath(drv_link))
        pci_addr = os.path.basename(os.path.realpath(dev))
        vendor_id = read(f"{dev}/vendor", "")
        name = gpu_product_name(dev, pci_addr, vendor_id)
        idx = int(re.search(r"card(\d+)", card).group(1))
        g = None
        if driver == "amdgpu":
            g = AmdGpu(card, name)
        elif driver in ("i915", "xe"):
            if intel_vram is None:
                intel_vram = _NvtopVramFeed()
                feeds["nvtop"] = intel_vram
            ef = _IntelEngineFeed(idx)
            feeds[f"intel{idx}"] = ef
            g = IntelGpu(card, name, idx, ef, intel_vram)
        elif driver in ("nvidia", "nvidia-drm"):
            if nvidia_feed is None:
                nvidia_feed = _NvidiaFeed()
                feeds["nvidia"] = nvidia_feed
            g = NvidiaGpu(name, nvidia_seen, nvidia_feed)
            nvidia_seen += 1
        if g is not None:
            g.pci_addr = pci_addr
            gpus.append(g)
    # label each card with the Vulkan device index llama.cpp would assign it
    vk = vulkan_device_map()
    for g in gpus:
        i = vk.get(getattr(g, "pci_addr", None))
        g.backend_dev = f"Vulkan{i}" if i is not None else None
    return gpus, feeds


# --------------------------------------------------------------------- the CPU
_CPU = {"tot": 0, "idle": 0, "util": None}
_CPU_E = {"t": 0.0, "j": 0, "w": None}


def _cpu_hwmon():
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if read(f"{h}/name") in ("k10temp", "coretemp", "zenpower"):
            return h
    return None


def _rapl_package():
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        if (read(f"{d}/name") or "").startswith("package") and os.path.isfile(f"{d}/energy_uj"):
            return d
    return None


HW_CPU = _cpu_hwmon()
# preferred CPU temperature label: the control/package sensor
T_CPU = None
if HW_CPU:
    for f in sorted(glob.glob(f"{HW_CPU}/temp*_label")):
        if (read(f) or "") in ("Tctl", "Tdie", "Package id 0", "Tccd1"):
            T_CPU = os.path.basename(f).replace("_label", "")
            break
    if not T_CPU:
        g = sorted(glob.glob(f"{HW_CPU}/temp*_input"))
        T_CPU = os.path.basename(g[0]).replace("_input", "") if g else None
RAPL_CPU = _rapl_package()
RAPL_MAX = read_int(f"{RAPL_CPU}/max_energy_range_uj", 0) if RAPL_CPU else 0

# every temperature the CPU sensor exposes (Tctl, Tccd1, Tccd2, Package, Core N...)
CPU_TEMPS = {}
if HW_CPU:
    for f in sorted(glob.glob(f"{HW_CPU}/temp*_label")):
        CPU_TEMPS[read(f) or os.path.basename(f)] = os.path.basename(f).replace("_label", "")

CPU_NAME = ""
NCPU = 0
for _r in (read("/proc/cpuinfo", "") or "").splitlines():
    if _r.lower().startswith("model name") and not CPU_NAME:
        CPU_NAME = re.sub(r"\s+\d+-Core Processor$", "", _r.split(":", 1)[1].strip())
    if _r.startswith("processor"):
        NCPU += 1


def cpu_sample():
    la = (read("/proc/loadavg", "0 0 0") or "0 0 0").split()
    d = {"name": CPU_NAME, "ncpu": NCPU, "util": _CPU["util"], "temp": None,
         "temps": {}, "freq": None, "power": None, "rapl_present": RAPL_CPU is not None,
         "load": la[:3], "loadavg": la[0] if la else "0"}
    if HW_CPU:
        for label, key in CPU_TEMPS.items():
            d["temps"][label] = sane(f"cpu.{label}",
                                     read_int(f"{HW_CPU}/{key}_input") // 1000, 1, 130)
    try:
        c = [int(x) for x in (read("/proc/stat", "") or "").splitlines()[0].split()[1:]]
        tot, idle = sum(c), c[3] + (c[4] if len(c) > 4 else 0)
        dt, di = tot - _CPU["tot"], idle - _CPU["idle"]
        if _CPU["tot"] and dt > 0:
            _CPU["util"] = max(0.0, min(100.0, 100.0 * (dt - di) / dt))
        _CPU["tot"], _CPU["idle"] = tot, idle
        d["util"] = _CPU["util"]
    except Exception:
        pass
    if HW_CPU and T_CPU:
        d["temp"] = sane("cpu.t", read_int(f"{HW_CPU}/{T_CPU}_input") // 1000, 1, 130)
    try:
        mhz = [float(r.split(":")[1]) for r in (read("/proc/cpuinfo", "") or "").splitlines()
               if r.lower().startswith("cpu mhz")]
        if mhz:
            d["freq"] = int(sum(mhz) / len(mhz))
    except Exception:
        pass
    if RAPL_CPU:
        j = read_int(f"{RAPL_CPU}/energy_uj", -1)
        if j >= 0:
            t = time.monotonic()
            if _CPU_E["j"] and t > _CPU_E["t"]:
                dt = t - _CPU_E["t"]
                if dt > 0.4:
                    dj = j - _CPU_E["j"]
                    if dj < 0 and RAPL_MAX:
                        dj += RAPL_MAX
                    _CPU_E["w"] = max(0.0, dj / dt / 1e6)
                    _CPU_E["t"], _CPU_E["j"] = t, j
            else:
                _CPU_E["t"], _CPU_E["j"] = t, j
        d["power"] = _CPU_E["w"]
    return d


# ------------------------------------------------------------------- memory
def mem_sample():
    m = {}
    for r in (read("/proc/meminfo", "") or "").splitlines():
        c = r.split()
        if len(c) >= 2 and c[0].endswith(":"):
            try:
                m[c[0][:-1]] = int(c[1]) // 1024
            except ValueError:
                pass
    d = {"free": m.get("MemAvailable", 0), "total": m.get("MemTotal", 0),
         "cached": m.get("Cached", 0), "buffers": m.get("Buffers", 0),
         "committed": m.get("Committed_AS", 0), "dirty": m.get("Dirty", 0),
         "shmem": m.get("Shmem", 0)}
    # swap split by destination: zram stays compressed IN ram, a swapfile leaves it
    zram_used = zram_total = disk_used = disk_total = 0
    for r in (read("/proc/swaps", "") or "").splitlines()[1:]:
        c = r.split()
        if len(c) < 4:
            continue
        used, total = int(c[3]) // 1024, int(c[2]) // 1024
        if "zram" in c[0]:
            zram_used += used; zram_total += total
        else:
            disk_used += used; disk_total += total
    d["swap_used"], d["swap_total"] = zram_used + disk_used, zram_total + disk_total
    d["zram_swap"], d["disk_swap"], d["disk_swap_total"] = zram_used, disk_used, disk_total
    # how much RAM zram actually holds those pages in (compressed)
    z_orig = z_compr = 0
    for z in glob.glob("/sys/block/zram*/mm_stat"):
        c = (read(z, "") or "").split()
        if len(c) >= 2:
            try:
                z_orig += int(c[0]) // 1048576
                z_compr += int(c[1]) // 1048576
            except ValueError:
                pass
    d["zram_orig"], d["zram_compr"] = z_orig, z_compr
    d["zram_ratio"] = (z_orig / z_compr) if z_compr else 0
    return d


# ----------------------------------------------------- llama.cpp server probe
def _port_of(cmd):
    """The --port a llama-server was started with (default 8080)."""
    for i, a in enumerate(cmd):
        if a == "--port" and i + 1 < len(cmd):
            return cmd[i + 1]
    return "8080"


def _host_of(cmd):
    """The --host it binds to, normalised to an address reachable from here."""
    for i, a in enumerate(cmd):
        if a == "--host" and i + 1 < len(cmd):
            return "127.0.0.1" if cmd[i + 1] in ("0.0.0.0", "::") else cmd[i + 1]
    return "127.0.0.1"


def discover_llama_servers():
    """Every running llama-server, keyed by the port it listens on — read from the
    process list, so a server is found on ANY port and ALL of them are found when
    several run at once, with nothing to configure."""
    out, seen = [], set()
    for p in glob.glob("/proc/[0-9]*"):
        if (read(f"{p}/comm", "") or "") != "llama-server":
            continue
        cmd = [c for c in (read(f"{p}/cmdline", "") or "").split("\x00") if c]
        port = _port_of(cmd)
        if port in seen:
            continue
        seen.add(port)
        model = ""
        if "-m" in cmd:
            try:
                model = os.path.basename(cmd[cmd.index("-m") + 1]).replace(".gguf", "")
            except IndexError:
                pass
        executable = os.path.basename(cmd[0]) if cmd else "llama-server"
        flavor = "llama.cpp" if executable == "llama-server" else executable
        out.append({"pid": os.path.basename(p), "port": port,
                    "host": _host_of(cmd), "model_hint": model, "flavor": flavor})
    return sorted(out, key=lambda s: s["port"])


def llama_spec_from_cmdline(port=None):
    """Which speculative head a server uses, read from its command line — no endpoint
    reports it. -md absent with an mtp type means the head is native. With `port`
    given, only the server on that port is inspected."""
    for p in glob.glob("/proc/[0-9]*"):
        if (read(f"{p}/comm", "") or "") != "llama-server":
            continue
        cmd = (read(f"{p}/cmdline", "") or "").split("\x00")
        if port is not None and _port_of([c for c in cmd if c]) != str(port):
            continue
        typ = head = nmax = None
        if "--spec-type" in cmd:
            try:
                typ = cmd[cmd.index("--spec-type") + 1]
            except IndexError:
                pass
        for flag in ("-md", "--model-draft", "--spec-draft-model"):
            if flag in cmd:
                try:
                    head = os.path.basename(cmd[cmd.index(flag) + 1]).replace(".gguf", "")
                    break
                except IndexError:
                    pass
        for flag in ("--spec-draft-n-max", "-n-max"):
            if flag in cmd:
                try:
                    nmax = cmd[cmd.index(flag) + 1]
                    break
                except IndexError:
                    pass
        return typ, head, nmax
    return None, None, None


def llama_ctx_from_cmdline(port=None):
    """The context a server was ALLOCATED, read from its -c. The authoritative source for
    the total, and the reason it is not derived instead: multiplying the per-slot context
    from /slots by total_slots is right on one build and wrong on another. MEASURED
    2026-08-25 across the four servers here — :7795 (-c 12288 -np 2) reports 6144 a slot,
    so per×slots recovers 12288 and agrees; :7797 (-c 4096, --reranking) reports 4096 a
    slot across FOUR slots, so the same arithmetic invents 16384 for a server that
    allocated 4096. The command line does not need the inference."""
    for p in glob.glob("/proc/[0-9]*"):
        if (read(f"{p}/comm", "") or "") != "llama-server":
            continue
        cmd = [c for c in (read(f"{p}/cmdline", "") or "").split("\x00") if c]
        if port is not None and _port_of(cmd) != str(port):
            continue
        for flag in ("-c", "--ctx-size"):
            if flag in cmd:
                try:
                    return int(cmd[cmd.index(flag) + 1])
                except (IndexError, ValueError):
                    return None
        return None
    return None


def llama_devices_from_cmdline(port=None):
    """The GPU backend devices a server was told to run on (-dev / --device), e.g.
    ['Vulkan0', 'Vulkan1']. An empty list means the flag was absent — which in llama.cpp
    means 'use every available device'. Read at runtime from the process, named after
    nothing: it just returns whatever labels the command line carries. Used to charge a
    server's energy line to the cards IT actually uses, not the whole box."""
    for p in glob.glob("/proc/[0-9]*"):
        if (read(f"{p}/comm", "") or "") != "llama-server":
            continue
        cmd = [c for c in (read(f"{p}/cmdline", "") or "").split("\x00") if c]
        if port is not None and _port_of(cmd) != str(port):
            continue
        for flag in ("-dev", "--device"):
            if flag in cmd:
                try:
                    return [x for x in cmd[cmd.index(flag) + 1].split(",") if x and x != "none"]
                except IndexError:
                    return []
        return []
    return []


# Every flag llama-server accepts, grouped for reading. This list is the ORDER and the
# LABELS — it is NOT the filter. Whatever it does not name still reaches the panel through
# the "other" group below, computed as the complement of what these entries consumed. That
# inversion is the whole point: the previous version showed a hardcoded subset, so a run
# started with --dry-multiplier, --pooling, --jinja or --load-mode simply had no line
# anywhere, and nothing said so. MEASURED 2026-08-25 on the four servers here — between 6
# and 8 settings shown out of 15 to 18 flags actually passed. A curated list also ages
# against llama.cpp: every flag added upstream would be invisible until someone edited
# this file, which is the same defect one release later.
_LLAMA_FLAG_GROUPS = (
    ("loading", (
        ("model", ("-m", "--model")), ("alias", ("-a", "--alias")),
        ("ctx", ("-c", "--ctx-size")),
        ("ngl", ("-ngl", "--gpu-layers", "--n-gpu-layers")),
        ("split", ("-sm", "--split-mode")), ("tensor-split", ("-ts", "--tensor-split")),
        ("devices", ("-dev", "--device")), ("main-gpu", ("-mg", "--main-gpu")),
        ("rpc", ("--rpc",)),
        ("flash-attn", ("-fa", "--flash-attn")),
        ("threads", ("-t", "--threads")), ("threads-batch", ("-tb", "--threads-batch")),
        ("batch", ("-b", "--batch-size")), ("ubatch", ("-ub", "--ubatch-size")),
        ("slots", ("-np", "--parallel")),
        ("mmap", ("--no-mmap", "--mmap")), ("mlock", ("--mlock",)),
        # the label of a negating switch is the CONCEPT, never the flag: "no-warmup off"
        # says warmup is ON, which is the opposite of what --no-warmup did
        ("load-mode", ("--load-mode",)), ("warmup", ("--no-warmup", "--warmup")),
        ("numa", ("--numa",)),
        ("override-tensor", ("-ot", "--override-tensor")),
        ("override-kv", ("--override-kv",)),
        ("lora", ("--lora", "--lora-scaled")),
        ("control-vector", ("--control-vector", "--control-vector-scaled")),
        ("rope", ("--rope-scaling", "--rope-freq-base", "--rope-freq-scale")),
        ("yarn", ("--yarn-orig-ctx", "--yarn-ext-factor", "--yarn-attn-factor",
                  "--yarn-beta-slow", "--yarn-beta-fast")),
    )),
    ("cache", (
        ("kv k", ("-ctk", "--cache-type-k")), ("kv v", ("-ctv", "--cache-type-v")),
        ("cache-ram", ("--cache-ram",)), ("kv-unified", ("--kv-unified", "-kvu")),
        ("cache-reuse", ("--cache-reuse",)), ("defrag", ("-dt", "--defrag-thold")),
        ("context-shift", ("--context-shift", "--no-context-shift")),
        ("cont-batching", ("-cb", "--cont-batching", "--no-cont-batching")),
        ("keep", ("--keep",)), ("swa-full", ("--swa-full",)),
    )),
    ("sampling", (
        ("temp", ("--temp", "--temperature")),
        ("dynatemp", ("--dynatemp-range", "--dynatemp-exp")),
        ("top-k", ("--top-k",)), ("top-p", ("--top-p",)), ("min-p", ("--min-p",)),
        ("typical", ("--typical", "--typical-p")), ("top-nsigma", ("--top-nsigma",)),
        ("xtc", ("--xtc-probability", "--xtc-threshold")),
        ("repeat", ("--repeat-penalty",)), ("repeat-last-n", ("--repeat-last-n",)),
        ("presence", ("--presence-penalty",)), ("frequency", ("--frequency-penalty",)),
        # DRY (Do not Repeat Yourself) — the repetition sampler that works on n-grams
        # instead of a flat token penalty. Five separate flags, none of which had a line.
        ("dry-multiplier", ("--dry-multiplier",)), ("dry-base", ("--dry-base",)),
        ("dry-allowed-length", ("--dry-allowed-length",)),
        ("dry-penalty-last-n", ("--dry-penalty-last-n",)),
        ("dry-breaker", ("--dry-sequence-breaker",)),
        ("mirostat", ("--mirostat",)), ("mirostat-lr", ("--mirostat-lr",)),
        ("mirostat-ent", ("--mirostat-ent",)),
        ("seed", ("-s", "--seed")), ("samplers", ("--samplers", "--sampling-seq")),
        ("ignore-eos", ("--ignore-eos",)), ("logit-bias", ("-l", "--logit-bias")),
        ("grammar", ("--grammar", "--grammar-file")), ("json-schema", ("-j", "--json-schema")),
        ("n-predict", ("-n", "--predict", "--n-predict")),
    )),
    ("reasoning", (
        ("format", ("--reasoning-format",)), ("budget", ("--reasoning-budget",)),
        ("effort", ("--reasoning-effort",)),
        ("jinja", ("--jinja",)),
        ("chat-template", ("--chat-template", "--chat-template-file")),
    )),
    ("role", (
        ("pooling", ("--pooling",)),
        ("embedding", ("--embedding", "--embeddings")),
        ("reranking", ("--reranking", "--rerank")),
    )),
    ("speculative", (
        ("type", ("--spec-type",)),
        ("head", ("-md", "--model-draft", "--spec-draft-model")),
        ("n-max", ("--spec-draft-n-max", "--draft-max", "--draft")),
        ("n-min", ("--spec-draft-n-min", "--draft-min")),
        # p-min is the adaptive cut-off: the draft stops proposing once its own
        # confidence drops below it (default 0.00 = never cut). One of the biggest,
        # most-tuned knobs — it reshapes the acceptance curve.
        ("p-min", ("--spec-draft-p-min", "--draft-p-min")),
        ("p-split", ("--spec-draft-p-split", "--draft-p-split")),
        ("draft-dev", ("-devd", "--device-draft")),
        ("draft-kv k", ("-ctkd", "--cache-type-k-draft", "--spec-draft-type-k")),
        ("draft-kv v", ("-ctvd", "--cache-type-v-draft", "--spec-draft-type-v")),
        ("draft-ngl", ("-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft")),
        ("draft-ctx", ("-cd", "--ctx-size-draft")),
    )),
    ("server", (
        ("host", ("--host",)), ("port", ("--port",)), ("path", ("--path",)),
        ("api-key", ("--api-key", "--api-key-file")),
        ("threads-http", ("--threads-http",)), ("timeout", ("-to", "--timeout")),
        ("metrics", ("--metrics",)), ("slots endpoint", ("--slots", "--no-slots")),
        ("props", ("--props",)), ("webui", ("--no-webui",)),
        ("ssl", ("--ssl-key-file", "--ssl-cert-file")),
    )),
)

# Values that must never be painted on a screen someone may screenshot. The tool is meant
# to be run in front of other people and its README carries a panel dump.
_SECRET_FLAGS = ("--api-key",)


def _mask(flag, value):
    if isinstance(value, str) and value and any(flag == s for s in _SECRET_FLAGS):
        return "•" * 8 + f" ({len(value)} chars)"
    return value


def _parse_cmdline_flags(cmd):
    """Command line to an ordered list of (flag, value) pairs, value True for a bare
    switch. A token starting with '-' is a NEW FLAG only when it is not a number: the
    previous parser had no such test, so `-np -1` read as a bare switch and the panel
    printed `slots on` for a server running with automatic parallelism (measured on
    :7799). `-1`, `-0.05` and `-1e-3` are values; `-np` and `--top-k` are flags."""
    def is_flag(tok):
        if not tok.startswith("-"):
            return False
        try:
            float(tok)
        except ValueError:
            return True
        return False

    out, i = [], 0
    while i < len(cmd):
        a = cmd[i]
        if is_flag(a):
            if i + 1 < len(cmd) and not is_flag(cmd[i + 1]):
                out.append((a, cmd[i + 1])); i += 2
            else:
                out.append((a, True)); i += 1
        else:
            i += 1
    return out


def llama_settings_from_cmdline(port=None):
    """The active server configuration, read from its command line and organized by theme.
    EVERY flag it was launched with appears: the themed groups above give the known ones a
    label and a place, and anything they do not name lands in `other`, verbatim, so a flag
    this file has never heard of is still on screen. Nothing that was left at its default
    appears (it is not on the command line). With `port` given, only that server is read.

    NOTE ON THE SAMPLING GROUP, because the word is narrower than it looks: what is here is
    the server's DEFAULT, applied only when a request does not carry its own. A client that
    sends temperature or repeat_penalty in the request body overrides it, and the panel
    cannot see that from /proc — the live values are in the slot, not the command line."""
    for p in glob.glob("/proc/[0-9]*"):
        if (read(f"{p}/comm", "") or "") != "llama-server":
            continue
        cmd = [c for c in (read(f"{p}/cmdline", "") or "").split("\x00") if c]
        if port is not None and _port_of(cmd) != str(port):
            continue
        return settings_from_cmd(cmd)
    return None


def settings_from_cmd(cmd):
    """The grouping itself, split out from the /proc walk so it can be exercised on a
    command line that no process is running — including flags this build has never seen."""
    pairs = _parse_cmdline_flags(cmd)
    base = lambda v: os.path.basename(v).replace(".gguf", "") if isinstance(v, str) else v
    used = set()                       # indices of pairs a themed entry claimed
    out = {}
    for group, entries in _LLAMA_FLAG_GROUPS:
        rows = []
        for label, aliases in entries:
            hits = [(k, (flag, val)) for k, (flag, val) in enumerate(pairs)
                    if flag in aliases]
            if not hits:
                continue
            shown = []
            for k, (flag, val) in hits:
                used.add(k)
                v = _mask(flag, val)
                if label in ("model", "head") or flag in ("--lora", "--lora-scaled",
                                                          "--control-vector"):
                    v = base(v)
                # a negating switch says what it turns OFF, so print the flag itself
                # rather than a bare "on" that reads as the opposite
                if v is True:
                    v = "off" if flag.startswith("--no-") else "on"
                shown.append(str(v))
            rows.append((label, " ".join(shown)))
        if rows:
            out[group] = rows
    # THE COMPLEMENT, and it is what makes this list an ordering instead of a filter
    leftover = [(flag, val) for k, (flag, val) in enumerate(pairs) if k not in used]
    if leftover:
        out["other"] = [(flag.lstrip("-"),
                         ("on" if val is True else str(_mask(flag, val))))
                        for flag, val in leftover]
    return out


# File capabilities put a binary in "secureexec" mode, which the kernel marks
# non-dumpable; from then on its fdinfo/maps need CAP_SYS_PTRACE and the per-process
# memory becomes unreadable. That is the usual reason a llama process shows dashes.
_CAP_NAMES = {14: "cap_ipc_lock", 19: "cap_sys_ptrace", 21: "cap_sys_admin",
              23: "cap_sys_nice", 24: "cap_sys_resource", 38: "cap_perfmon", 39: "cap_bpf"}


def _capabilities(status):
    for r in status.splitlines():
        if r.startswith("CapPrm:"):
            try:
                b = int(r.split()[1], 16)
            except (IndexError, ValueError):
                return []
            return [_CAP_NAMES.get(i, f"cap {i}") for i in range(64) if b >> i & 1]
    return []


def llama_processes():
    """Every llama.cpp process, with its resident memory and — read straight from
    fdinfo — how much of it is on the card (VRAM) versus spilled into host memory
    (GTT), which is the read-via-PCIe overflow that quietly tanks tokens/s."""
    out = []
    for p in glob.glob("/proc/[0-9]*"):
        name = read(f"{p}/comm", "") or ""
        if not name.startswith("llama-"):
            continue
        v = {"pid": os.path.basename(p), "name": name, "rss": 0, "model": "",
             "vram": 0, "gtt": 0, "read": False, "caps": [], "error": None, "gpu_accel": False}
        status = read(f"{p}/status", "") or ""
        for r in status.splitlines():
            if r.startswith("VmRSS:"):
                v["rss"] = int(r.split()[1]) // 1024
        cmd = (read(f"{p}/cmdline", "") or "").split("\x00")
        if "-m" in cmd:
            try:
                v["model"] = os.path.basename(cmd[cmd.index("-m") + 1]).replace(".gguf", "")
            except IndexError:
                pass
        try:
            fds = os.listdir(f"{p}/fd")
            for fd in fds:
                try:
                    target = os.readlink(f"{p}/fd/{fd}")
                    if "nvidia" in target or "renderD" in target:
                        v["gpu_accel"] = True
                        break
                except OSError:
                    pass
        except OSError:
            pass
        try:
            files = os.listdir(f"{p}/fdinfo")
        except OSError:
            v["caps"] = _capabilities(status)
            v["error"] = "oserror"
            out.append(v)
            continue
        for f in files:
            t = read(f"{p}/fdinfo/{f}", "") or ""
            if "drm-memory-vram" not in t and "drm-memory-local" not in t:
                continue
            for r in t.splitlines():
                if ":" not in r:
                    continue
                k, _, rest = r.partition(":")
                m = re.search(r"(\d+)", rest)
                if not m:
                    continue
                kb = int(m.group(1))
                k_strip = k.strip()
                if k_strip in ("drm-memory-vram", "drm-memory-local"):
                    v["vram"] = kb // 1024
                elif k_strip in ("drm-memory-gtt", "drm-memory-system"):
                    v["gtt"] = kb // 1024
            v["read"] = True
            break
        out.append(v)
    return sorted(out, key=lambda x: -x["rss"])


class LlamaProbe:
    """Reads a llama.cpp server: /metrics for the cumulative counters (t/s, KV,
    speculative acceptance), /slots for the live phase, /props for the model. The
    counters only advance when a request finishes, so the live phase comes from
    /slots. Speed histories are kept per model and reset when the model changes."""
    def __init__(self, port="8080", host="127.0.0.1"):
        self.port = port
        self.host = host or "127.0.0.1"
        self.state = {"alive": False}
        self.cnt = {}
        # None means NOT MEASURABLE and is drawn as a dash with the reason. 0.0 would be
        # a claim ("this server prefills at zero tokens a second") that no source ever
        # made — and with /metrics off, which is the default, that claim was the only
        # thing the prefill field ever showed.
        self.tg = self.pp = None
        self.tg_last = self.pp_last = None   # rate of the last COMPLETED request
        # 6 s of trail at the fastest refresh (0.5 s) needs 13 slots, not 12: the old cap
        # silently shortened the window at the very rate a user picks to watch it closely.
        self._dhist = deque(maxlen=32)       # (time, n_decoded) — generation
        self._phist = deque(maxlen=32)       # (time, n_prompt_processed) — prefill
        self._dlast = self._plast = 0
        self._busy = False
        self.ctx = self.ctx_total = self.slots = None   # server FACTS, survive a bad tick
        self.model = ""
        self.model_at = 0.0
        self._rf = None                    # reasoning format (from /slots, survives ticks)

    def _get(self, path, timeout=2.5):
        import urllib.request
        import urllib.error
        try:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}", timeout=timeout) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            return e.read().decode()
        except urllib.error.URLError as e:
            if isinstance(getattr(e, 'reason', None), TimeoutError):
                raise TimeoutError()
            raise

    def _backoff(self):
        """A server that stopped answering must not be drawn as a server answering
        ZEROS. Arm the 60 s countdown and hand back the LAST GOOD state, marked stale:
        ctx, kv and model stay on screen and only the two speeds go to "not measurable".
        Rebuilding the dict from defaults here is what made the context appear to reset."""
        # A FIXED sixty seconds was wrong in both directions at once: far too long a
        # penalty for one slow answer, and useless against a server that will be silent
        # for longer than that anyway. MEASURED 2026-08-25 on :8181 during a 78 s CPU
        # prefill — /slots did not answer AT ALL (three consecutive 20 s timeouts, then
        # 13.83 s the moment the prefill ended, zero replies under 1.2 s). No timeout
        # value would have helped, and the 60 s blackout simply hid the whole run. So:
        # retry immediately once, then back away geometrically, and never for long.
        # capped so the ladder below never computes 2**700 on a long outage just to throw
        # it away at the min(): the wait is bounded at 5 s from the fourth failure on
        self._fails = min(getattr(self, "_fails", 0) + 1, 8)
        # capped low ON PURPOSE: with the fan-out above, a probe that keeps failing no
        # longer slows anything else down, so there is nothing to buy by waiting longer —
        # and every second of wait is a second of missing the recovery
        self._timeout_until = time.monotonic() + (
            0 if self._fails < 2 else min(2 ** (self._fails - 1), 5))
        s = self._blank()
        s.update(self.state)               # keep whatever the last good tick knew
        s.update(phase="not answering", stale=True, pp=None, tg=None,
                 pp_last=self.pp_last, tg_last=self.tg_last)
        self.state = s
        return s

    # A prefill that lasts about as long as the poll interval CANNOT be measured by
    # polling: on :7795 a 4001-token prefill ran in 1.94 s and left two usable samples,
    # from which the slope reads ~790 t/s against the 2065 the server timed. So the live
    # prefill is published only once it has been running longer than this — a big prompt
    # on CPU takes a minute and is genuinely samplable, a GPU prefill is over before the
    # second tick. Below the threshold the completed-request figure from /metrics is used
    # instead, and it is exact (2068 against 2065.45, measured 2026-08-25). Publishing the
    # slope anyway would be manufacturing precision, which is the defect being fixed here
    # wearing better clothes.
    _PREFILL_MIN_SPAN = 2.0

    @staticmethod
    def _slope(hist, cur, prev, minspan=0.6, window=6.0, anchor_zero=False):
        """Tokens per second from a (time, counter) trail, or None while the trail is
        too short to say. Called only while the slot is BUSY — that restriction is the
        fix, not a detail. Feeding it a finished request's counter, which llama.cpp
        leaves standing in the slot, makes the window slide off the end of the work: the
        numerator shrinks while the denominator stays ~6 s, so the reading decays, and
        once the window is entirely past the request the delta is 0 and the guard stops
        updating — leaving the last decayed value frozen on screen. MEASURED 2026-08-25
        on :8181, whose own timings reported 29.93 t/s: 29.7, 27.8, 24.7, 21.1, 17.7,
        14.2, 12.7, 9.4, 5.4, 1.606, then 1.606 for as long as anyone watched."""
        now = time.monotonic()
        if cur < prev:                     # counter restarted: a new request began
            hist.clear()
        # WHERE THE TRAIL STARTS is not the same question for the two rates, and using one
        # answer for both is wrong in opposite directions. Prefill starts at zero tokens
        # processed, so the zero IS the anchor and dropping it loses the start of the work.
        # Generation also starts at zero decoded, but that zero sits at the END of prefill:
        # anchoring there would divide the generated tokens by generation time PLUS prefill
        # time and report a speed the model never ran at.
        if cur <= 0 and not anchor_zero:
            return None
        hist.append((now, cur))
        while len(hist) >= 2 and now - hist[0][0] > window:
            hist.popleft()
        t0, c0 = hist[0]
        if len(hist) >= 2 and now - t0 >= minspan and cur - c0 > 0:
            return (cur - c0) / (now - t0)
        return None

    def _blank(self):
        """The full SHAPE of a sample. Server FACTS (context, slot count, model) describe
        the SERVER and not this tick, so they are seeded from the last good sample: one
        failed request must not be able to make the panel read "ctx 0". Only the per-tick
        MEASUREMENTS start empty. Every exit from sample() returns this shape — handing
        back a partial dict pushes the job of inventing defaults onto nine render sites,
        which is how a missing key becomes a zero on screen."""
        return {"alive": False, "phase": "off", "pp": None, "tg": None,
             "pp_last": self.pp_last, "tg_last": self.tg_last, "kv": None,
             "ctx": self.ctx, "ctx_total": self.ctx_total, "slots": self.slots,
             "model": self.model, "spec": None, "tok_step": None,
             "spec_acc": 0, "spec_draft": 0, "spec_pos": [], "active": None, "queued": None,
             "pp_life": None, "tg_life": None, "spec_type": None, "spec_head": None,
             "spec_nmax": None, "spec_stale": False, "cache_hit": 0, "prompt_new": 0,
             "max_tok": None, "reuse": None, "budget": 0, "decoded": 0, "kv_used": 0,
             "metrics_off": False, "slots_off": False, "stale": False, "kv_cap": None,
             "active_metric": None}

    def sample(self):
        import json, time
        now = time.monotonic()
        timeout_until = getattr(self, '_timeout_until', 0)
        if now < timeout_until:
            s = self._blank()
            s.update(self.state)
            s.update(phase=f"not answering ({int(timeout_until - now) + 1}s)",
                     stale=True, pp=None, tg=None)
            self.state = s
            return s
        d = self._blank()
        metrics_ok = False
        try:
            m = {}
            raw = self._get("/metrics")
            # A build started without --metrics answers 501 with a JSON error body. That
            # is NOT "every counter reads zero", it is "no counter exists" — and reading
            # it as zeros is exactly what pinned prefill, active, queued and prompt-reuse
            # to 0 on every server here (all four, measured 2026-08-25).
            if raw.lstrip().startswith("{"):
                d["alive"] = True
                d["metrics_off"] = True
            else:
                for line in raw.splitlines():
                    if line.startswith("#") or " " not in line:
                        continue
                    k, _, v = line.partition(" ")
                    try:
                        m[k] = float(v)
                    except ValueError:
                        pass
                d["alive"] = True
                metrics_ok = bool(m)
                d["metrics_off"] = not metrics_ok
            if metrics_ok:
                d["active"] = int(m.get("llamacpp:requests_processing", 0))
                d["queued"] = int(m.get("llamacpp:requests_deferred", 0))
                # prompt-cache reuse (how well --cache-reuse pays off) and the largest
                # context ever seen against what -c allocated
                d["cache_hit"] = int(m.get("llamacpp:prompt_tokens_cached_total", 0))
                d["prompt_new"] = int(m.get("llamacpp:prompt_tokens_total", 0))
                d["max_tok"] = int(m.get("llamacpp:n_tokens_max", 0)) or None
                # a build that does not EXPORT the cache counter is not a build with zero
                # reuse: :8181 (ik_llama) exports 10 counters where mainline exports 15,
                # and reading the gap as a value would print "prompt reuse 0.0%" about a
                # thing nobody measured — the same defect as prefill, one field along
                _tot = d["cache_hit"] + d["prompt_new"]
                d["reuse"] = (d["cache_hit"] / _tot) if (
                    _tot and "llamacpp:prompt_tokens_cached_total" in m) else None
                draft = m.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
                acc = m.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
                d["spec_draft"], d["spec_acc"] = draft, acc
                d["spec"] = (acc / draft) if draft else None
                drafts = m.get("llamacpp:spec_decode_num_drafts_total", 0)
                if drafts:
                    i, pos = 0, []
                    while True:
                        v = m.get(f'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{{position="{i}"}}')
                        if v is None:
                            break
                        pos.append(v / drafts)
                        i += 1
                    d["spec_pos"] = pos
                    d["tok_step"] = 1.0 + acc / drafts
                C = ("llamacpp:prompt_tokens_total", "llamacpp:prompt_seconds_total",
                     "llamacpp:tokens_predicted_total", "llamacpp:tokens_predicted_seconds_total")
                c = {k: m.get(k, 0.0) for k in C}
                # a finished request moves these counters; their delta is the rate of the
                # request that just ended, which is the LAST rate, never the current one
                if self.cnt:
                    if c[C[0]] - self.cnt[C[0]] > 0 and c[C[1]] - self.cnt[C[1]] > 0:
                        self.pp_last = (c[C[0]] - self.cnt[C[0]]) / (c[C[1]] - self.cnt[C[1]])
                    if c[C[2]] - self.cnt[C[2]] > 0 and c[C[3]] - self.cnt[C[3]] > 0:
                        self.tg_last = (c[C[2]] - self.cnt[C[2]]) / (c[C[3]] - self.cnt[C[3]])
                self.cnt = c
                d["pp_life"] = c[C[0]] / c[C[1]] if c[C[1]] else None
                d["tg_life"] = c[C[2]] / c[C[3]] if c[C[3]] else None
                d["pp_last"], d["tg_last"] = self.pp_last, self.tg_last
        except TimeoutError:
            return self._backoff()
        except Exception:
            pass
        x = {}
        _am = d["active"]                  # what /metrics claimed, kept for comparison
        try:
            s = json.loads(self._get("/slots"))
            if not isinstance(s, list):
                # /slots disabled (--no-slots) or an error body: there is no live phase to
                # read. Say so — do NOT let ctx, kv and the phase fall through as zeros.
                d["slots_off"] = True
                raise ValueError("slots endpoint unavailable")
            x = s[0] if s else {}
            # EVERY SLOT, not slot 0. A server started with -np N hands a request to
            # whichever slot is free, so reading s[0] alone makes the panel say "idle"
            # while the server generates on another slot — and show that slot's LEFTOVER
            # n_decoded, which is the fossil that then froze on screen. MEASURED
            # 2026-08-25 on :8181 (-np 2): a request ran to completion on slot 1 at
            # 29.88 t/s while slot 0 sat at state 0 holding n_decoded from an older run.
            # Both generative servers on this box run -np 2, so about half of all work
            # was invisible.
            def _slot_busy(sl):
                # new builds: is_processing (bool). old builds: state (int),
                # 0=idle 1=started/prefill 2=prompt_done 3=gen
                stx = sl.get("state")
                return bool(sl.get("is_processing")) or (stx is not None and stx != 0)

            def _slot_next(sl):
                nx = sl.get("next_token")
                return nx[0] if isinstance(nx, list) and nx else nx if isinstance(nx, dict) else {}

            busy_slots = [sl for sl in s if _slot_busy(sl)]
            busy = bool(busy_slots)
            # a slot that is not working carries the residue of its last request: it must
            # count towards the KV fill (those cells ARE occupied) and never towards a rate
            dec = sum((_slot_next(sl).get("n_decoded") or sl.get("n_decoded") or 0)
                      for sl in busy_slots)
            d["decoded"] = dec
            # n_remain is the budget left (n_predict - generated), -1 when unlimited;
            # near its ceiling means the model is not stopping on its own
            _bn = _slot_next(busy_slots[0]) if busy_slots else {}
            resta = _bn.get("n_remain")
            _bdec = (_bn.get("n_decoded") or 0) if busy_slots else 0
            d["budget"] = (_bdec + resta) if isinstance(resta, int) and resta >= 0 else 0
            _nproc = sum((sl.get("n_prompt_tokens_processed") or 0) for sl in busy_slots)
            # the number of slots actually working is a LIVE reading of the queue depth,
            # and unlike llamacpp:requests_processing it needs no --metrics
            # DIRECT observation beats a counter here: the number of slots actually
            # working is read from the slots themselves, needs no --metrics, and cannot
            # disagree with the phase drawn one line above it
            d["active"] = len(busy_slots)
            d["active_metric"] = _am
            # the set of working slots changing mid-window would step the summed counter
            # and inflate one slope reading; close the trail instead of publishing it
            _ids = tuple(sorted(sl.get("id") for sl in busy_slots))
            if _ids != getattr(self, "_bids", ()):
                self._dhist.clear(); self._phist.clear()
                self._dlast = self._plast = 0
            self._bids = _ids
            # ---- live rates, and ONLY while the slot is busy ----------------------
            # The /metrics counters move only when a request finishes, so during a stream
            # the slot's own progress is the sole live source. When the slot goes idle
            # llama.cpp LEAVES the counters standing, so the trail must be closed here,
            # not left to age out: what stands after a request is the LAST rate, and it
            # is labelled as such rather than dressed up as the current one.
            if busy:
                r = self._slope(self._dhist, dec, self._dlast)
                if r is not None:
                    self.tg = r
                r = self._slope(self._phist, _nproc, self._plast,
                                minspan=self._PREFILL_MIN_SPAN, anchor_zero=True)
                if r is not None:
                    self.pp = r
            elif self._busy:               # the tick the request ended: latch and close
                if self.tg is not None:
                    self.tg_last = self.tg
                if self.pp is not None:
                    self.pp_last = self.pp
                self._dhist.clear(); self._phist.clear()
                self.tg = self.pp = None
            else:
                self.tg = self.pp = None
            self._dlast, self._plast, self._busy = dec, _nproc, busy
            d["phase"] = "generating" if (busy and dec > 0) else "prefill" if busy else "idle"
            nctx = x.get("n_ctx", 0) or 0
            if not isinstance(nctx, int) or nctx < 0:
                nctx = 0                       # a negative context is not a context
            if nctx:
                self.ctx = nctx            # per-SLOT context; the total comes from /props
                d["ctx"] = nctx
            # New builds have n_prompt_tokens*; old builds don't — fall back to
            # the decoded count alone (incomplete but honest). Summed over EVERY slot,
            # busy or not, because an idle slot still holds its conversation in the cache.
            occupied = 0
            for sl in s:
                _npt = sl.get("n_prompt_tokens", 0) or 0
                _npc = (sl.get("n_prompt_tokens_cache", 0) or 0) \
                     + (sl.get("n_prompt_tokens_processed", 0) or 0)
                _sd = _slot_next(sl).get("n_decoded") or sl.get("n_decoded") or 0
                occupied += (max(_npt, _npc) + _sd) if (_npt or _npc) else _sd
            d["kv_used"] = occupied
            # denominator: the WHOLE allocation when several slots share it, so the sum
            # above is compared against the capacity it was actually drawn from
            _cap = d["ctx_total"] or ((d["ctx"] or 0) * (d["slots"] or 1)) or d["ctx"]
            if _cap:
                d["kv"] = min(1.0, occupied / _cap)
                d["kv_cap"] = _cap
            d["alive"] = True
            d["spec_stale"] = busy
            self._fails = 0                # it answered: the retry ladder resets
        except TimeoutError:
            return self._backoff()
        except Exception:
            if not d["slots_off"]:
                d["stale"] = True          # kept last-good ctx/kv rather than zeroing them
        d["pp"], d["tg"] = self.pp, self.tg
        d["pp_last"], d["tg_last"] = self.pp_last, self.tg_last
        # reasoning format is in the slot data itself (both old and new builds).
        # GUARDED, and the guard was removed once and put back: `x` is seeded to {} above,
        # which covers "/slots failed", but NOT "/slots answered with a list whose items are
        # not objects" — there `.get` raises and takes the whole sample down with it, so a
        # server that is merely odd gets drawn as one that is not answering.
        try:
            _rf = (x.get("params", {}) or {}).get("reasoning_format") \
                  or x.get("reasoning_format")
        except Exception:
            _rf = None
        if _rf:
            self._rf = _rf
        d["reasoning_format"] = self._rf
        if d["alive"] and (not self.model or time.monotonic() - self.model_at > 15):
            self.model_at = time.monotonic()
            try:
                p = json.loads(self._get("/props"))
                self.model = (os.path.basename(p.get("model_path", "")).replace(".gguf", "")
                              or p.get("model_name", ""))
                # /slots reports the PER-SLOT context (-c divided by -np); the config panel
                # reads -c off the command line. Both are true and they are different
                # numbers under the same word — :7795 shows 6144 here and 12288 there.
                # Carry both so the panel can say which is which instead of picking one.
                _sl = p.get("total_slots")
                _per = (p.get("default_generation_settings") or {}).get("n_ctx")
                # the total comes from -c, never from per-slot × slots (see
                # llama_ctx_from_cmdline for the two builds that disagree)
                _tot = llama_ctx_from_cmdline(self.port) or p.get("n_ctx")
                if isinstance(_sl, int) and _sl > 0:
                    self.slots = _sl
                if _tot:
                    self.ctx_total = _tot
                if not self.ctx and _per:
                    self.ctx = _per
            except TimeoutError:
                return self._backoff()
            except Exception:
                pass
        d["model"] = self.model
        d["ctx"] = d["ctx"] or self.ctx
        d["ctx_total"], d["slots"] = self.ctx_total, self.slots
        if d["alive"]:
            d["spec_type"], d["spec_head"], d["spec_nmax"] = llama_spec_from_cmdline(self.port)
        self.state = d
        return d


def sample_llama_fleet(explicit_port=None):
    """Sample every running llama-server at once. A probe is kept per port and reused
    between ticks (so its rolling rate survives), created when a server appears and
    dropped when it exits. `explicit_port`, if given and not already discovered, is
    probed too. Returns one state dict per server, tagged with pid, port, and whether
    more than one server is present (which the UI uses to reveal per-server labels)."""
    servers = discover_llama_servers()
    if explicit_port and not any(s["port"] == str(explicit_port) for s in servers):
        servers.append({"pid": "?", "port": str(explicit_port),
                        "host": "127.0.0.1", "model_hint": ""})
        servers.sort(key=lambda s: s["port"])
    active, todo = set(), []
    for srv in servers:
        port = srv["port"]
        active.add(port)
        pr = _probes.get(port)
        if pr is None:
            pr = _probes[port] = LlamaProbe(port, srv["host"])
        todo.append((srv, pr))
    # Probe every server AT ONCE. Read in a row, ONE server that has stopped answering
    # costs every other server its own timeout — measured on this box, a silent :8181
    # held each pass for 2.5 s, so the three healthy servers refreshed six times slower
    # than they could. That coupling is also what forced the retry ladder to back away
    # for 16 s at a time, which then risks missing the moment the slow one recovers.
    # With the fan-out a failing probe costs nothing but its own thread, so the ladder
    # stays short. Probes are created above, in this thread: each one is then touched by
    # exactly one worker, and the _probes dict itself is never mutated concurrently.
    results = [None] * len(todo)

    def _one(i, pr):
        try:
            results[i] = pr.sample()
        except Exception:
            results[i] = None

    ths = [threading.Thread(target=_one, args=(i, pr), daemon=True)
           for i, (_s, pr) in enumerate(todo)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=10.0)
    out = []
    for (srv, pr), d in zip(todo, results):
        if d is None:                      # worker still stuck or it raised
            d = dict(pr._blank()); d.update(pr.state or {})
            d["phase"], d["stale"] = "not answering", True
        d["pid"], d["port"] = srv["pid"], srv["port"]
        d["flavor"] = srv.get("flavor", "llama.cpp")
        if not d.get("model"):
            d["model"] = srv["model_hint"]
        out.append(d)
    for k in [k for k in _probes if k not in active]:
        del _probes[k]
    for d in out:
        d["multi"] = len(out) > 1
    return out


# ------------------------------------------------------------------- --probe
def probe():
    """Print everything the data layer sees. Used to validate portability without
    the TUI."""
    gpus, feeds = discover_gpus()
    for f in feeds.values():
        f.start()
    print(f"GPUs discovered: {len(gpus)}")
    time.sleep(2.5)                              # let the feeds warm up
    cpu_sample(); time.sleep(0.5)                # util needs two samples
    for g in gpus:
        s = g.sample()
        vram = (f"{s.get('vram_used')}/{s.get('vram_total')} MiB"
                if s.get("vram_total") else "—")
        print(f"\n[{s['vendor']}] {s['name']}")
        print(f"   util={s.get('util')}%  vram={vram}  temp={s.get('temp_main')}C"
              f"  power={s.get('power')}W/{s.get('power_cap')}  "
              f"clk={s.get('sclk')}/{s.get('mclk')}MHz")
        if s.get("throttle"):
            print(f"   throttling: {', '.join(s['throttle'])}")
        if s.get("metrics_version"):
            print(f"   gpu_metrics v{s['metrics_version']}")
    c = cpu_sample()
    print(f"\n[CPU] {c['name']}")
    print(f"   util={c['util']:.0f}%  temp={c['temp']}C  freq={c['freq']}MHz  "
          f"power={c['power']}W (rapl:{'yes' if c['rapl_present'] else 'no'})  load={c['loadavg']}")
    m = mem_sample()
    print(f"\n[MEM] free {m['free']} / {m['total']} MiB  swap {m['swap_used']}/{m['swap_total']}")
    port = next((a for a in sys.argv[1:] if a.isdigit()), None)
    fleet = sample_llama_fleet(port)
    if not fleet:
        print("\n[llama.cpp] no server found")
    for ll in fleet:
        print(f"\n[llama.cpp :{ll['port']}] pid={ll['pid']} alive={ll['alive']} "
              f"model={ll['model'] or '—'} phase={ll['phase']} "
              f"pp={_n(ll.get('pp'))}/last {_n(ll.get('pp_last'))} "
              f"tg={_n(ll.get('tg'), 1)}/last {_n(ll.get('tg_last'), 1)} "
              f"ctx={ll.get('ctx')} tot={ll.get('ctx_total')} slots={ll.get('slots')} "
              f"metrics_off={ll.get('metrics_off')} spec={ll['spec']} "
              f"head={ll['spec_head']} type={ll['spec_type']}")
    for f in feeds.values():
        f.stop = True


# =========================================================== history & stats
HIST = {}
_probes = {}            # port -> LlamaProbe, reused across ticks and pruned when a server exits
_model_ports = {}       # port -> last model seen, to reset that server's median on a swap
_last_pp = {}           # port -> last prefill speed recorded, so only fresh prefills are logged
_port_seen = {}         # port -> time a live server was last seen (survives brief restarts)
_session = {"start": None, "t": None, "energy_j": 0.0}   # cumulative energy since launch


def record(key, value):
    """Append one sample to a series. NON-FINITE VALUES NEVER GET IN, and this is the only
    door, so no series can hold one. `float()` accepts the strings "nan" and "inf" without
    raising, and three readers parse external tools that way — intel_gpu_top, nvtop and
    nvidia-smi — so a sensor printing either would put it straight into the history; the
    `num()` helper on the nvidia side only catches ValueError, which those do not raise.
    Downstream a NaN raised ValueError inside the sparkline and an inf raised OverflowError
    in the step ladder, either of which kills the whole interface. The existing `sane()`
    guard does not cover this path: it is applied to clocks, temperatures, watts and
    voltage, not to utilisation or VRAM, which are exactly what is charted.
    A bad sample HOLDS THE PREVIOUS ONE rather than being dropped: this is a fixed-cadence
    series where one column is one tick, so a missing sample still has to occupy its
    column, and holding is what the tool already does elsewhere for an out-of-range read."""
    s = HIST.setdefault(key, deque(maxlen=3600))
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        value = s[-1] if s else 0
    s.append(value)


def median_dev(key, window=60):
    """Median and standard deviation of the RECENT samples (last `window`), not the
    whole history — over thousands of samples the median is so stable it looks
    frozen, and what you want on screen is generation right now."""
    import statistics
    s = HIST.get(key)
    if not s or len(s) < 2:
        return None, None
    from itertools import islice
    v = list(islice(s, max(0, len(s) - window), len(s)))
    if len(v) < 2:
        return None, None
    return statistics.median(v), statistics.pstdev(v)


def _extremes(key):
    """min, mean, max over the whole recorded history of a series — the spread the
    summary tiles and the trend peaks are drawn from. None when nothing is recorded."""
    s = HIST.get(key)
    if not s:
        return None, None, None
    v = list(s)
    return min(v), sum(v) / len(v), max(v)


# =============================================================== curses render
import curses

OK = WARN = CRIT = DIM = SER = 0
BLOCKS = "▏▎▍▌▋▊▉█"


def _bar(value, total, width, attr):
    if width <= 0:
        return [("", 0)]
    if not total:
        return [("░" * width, DIM)]
    p = max(0.0, min(1.0, value / total)) * width
    full = int(p)
    s = "█" * full
    if full < width:
        frac = p - full
        s += BLOCKS[int(frac * 8)] if frac > 0.06 else "░"
        s += "░" * (width - full - 1)
    return [(s[:width], attr)]


def _wwidth(s):
    """Display width of a string in terminal cells. Nearly every glyph here is narrow
    — box drawing, blocks, sparklines, ·°±→… all take one cell — but the warning sign
    ⚠ (U+26A0) is emoji-class and most terminals draw it two cells wide. Counting it
    as one is what lands a box border a column early. Combining marks and the emoji
    variation selector take no cell. Over-counting a hair is safe (it only opens a
    gap); under-counting corrupts the borders, so ⚠ is treated as the wider case."""
    n = 0
    for ch in s:
        o = ord(ch)
        if o == 0xFE0F or 0x300 <= o <= 0x36F:
            continue
        n += 2 if o == 0x26A0 else 1
    return n


def _wtrim(s, width):
    """Trim a string to at most `width` display cells (not characters)."""
    out, n = [], 0
    for ch in s:
        cw = _wwidth(ch)
        if n + cw > width:
            break
        out.append(ch)
        n += cw
    return "".join(out)


def _cellw(cell):
    """Display width of a cell — a list of (text, attr) segments drawn as a unit."""
    return sum(_wwidth(t) for t, _ in cell)


def _flow(cells, width, sep=" │ "):
    """Pack a list of cells (each a segment list) into as few rows as possible without
    any row exceeding `width` cells, greedily, keeping order. This is what lets a
    section lay its metrics out HORIZONTALLY on a wide terminal — util, vram, temp and
    clock share one line — and reflow onto more lines as the terminal narrows, instead
    of one metric per line wasting the width. Cells are divided by a dim separator so a
    dense row still reads as distinct fields. Empty cells are skipped."""
    cells = [c for c in cells if c and _cellw(c) > 0]
    rows, cur, cw, sw = [], [], 0, _wwidth(sep)
    for cell in cells:
        cwidth = _cellw(cell)
        if cur and cw + sw + cwidth > width:
            rows.append(cur)
            cur, cw = [], 0
        if cur:
            cur.append((sep, DIM))
            cw += sw
        cur.extend(cell)
        cw += cwidth
    if cur:
        rows.append(cur)
    return rows


def _put(w, y, x, segs, limit):
    """Draw a list of (text, attr) segments, clipping at `limit` columns. Column
    bookkeeping is by display width, not character count, so a wide glyph advances
    the cursor by what it actually occupies and the borders stay aligned."""
    h, W = w.getmaxyx()
    limit = min(limit, W - 1)
    for text, attr in segs:
        if x >= limit:
            break
        if not text:
            continue          # an empty segment (e.g. a full sparkline's zero-width pad)
        tw = _wwidth(text)     # is a no-op, NOT a reason to stop drawing the rest of the row
        if x + tw > limit:
            text = _wtrim(text, max(0, limit - x - 1)) + "…"
            tw = _wwidth(text)
        try:
            w.addnstr(y, x, text, limit - x, attr)
        except curses.error:
            pass
        x += tw
    return x


def _box(w, y, x, width, height, title, note=""):
    top = "╭─ " + title + " "
    note = f" {note} " if note else ""
    fill = max(0, width - len(top) - len(note) - 2)
    _put(w, y, x, [(top + "─" * fill, DIM)], x + width)
    if note:
        _put(w, y, x + len(top) + fill, [(note, DIM)], x + width)
    _put(w, y, x + width - 2, [("─╮", DIM)], x + width + 1)
    for i in range(1, height - 1):
        _put(w, y + i, x, [("│", DIM)], x + 1)
        _put(w, y + i, x + width - 1, [("│", DIM)], x + width)
    _put(w, y + height - 1, x, [("╰" + "─" * (width - 2) + "╯", DIM)], x + width + 1)


def _temp_attr(t):
    return CRIT if (t or 0) >= 90 else WARN if (t or 0) >= 80 else OK


def _temp_attr_th(t, thresholds):
    """Colour a temperature by the driver-declared critical/emergency points when
    available, falling back to fixed 80/90°C."""
    if t is None:
        return DIM
    crit, emerg = thresholds or (0, 0)
    if emerg and t >= emerg:
        return CRIT
    if crit and t >= crit:
        return WARN
    return _temp_attr(t)


def _gtt_status(gtt, vram_free):
    """High GTT is not automatically 'model evicted from VRAM'. If the free VRAM
    would fit the GTT, the kernel never had a reason to evict — it is host memory by
    choice (pinned transfer buffers, the embedding table) and -ts won't move it."""
    if gtt < 1000:
        return None
    if vram_free is not None and vram_free >= gtt:
        return (f"{gtt} MiB host memory, not eviction", WARN if gtt >= 3000 else OK)
    return (f"⚠ {gtt} MiB read via PCIe (spilled)", CRIT if gtt >= 3000 else WARN)


def _util_attr(u):
    return CRIT if (u or 0) >= 95 else WARN if (u or 0) >= 80 else OK


def _gpu_rows(s, width, idx=0):
    """One GPU, packed horizontally: the activity bars (util, vram) lead, then the
    utilisation average, temperatures, clocks, power, voltage and fan flow onto the
    same line while they fit and wrap onto the next when they don't. The GTT/eviction
    verdict, the PCIe link and any throttle reasons are full sentences, so they keep
    their own lines below."""
    inner = width - 4
    bw = max(8, min(14, inner // 6))
    cells = []
    u = s.get("util")
    cells.append([("util ", DIM)] + _bar(u or 0, 100, bw, _util_attr(u))
                 + [(f" {u:>3.0f}%" if u is not None else " —", 0)])
    _mn, avg, _mx = _extremes(f"gpu{idx}_util")
    if avg is not None:
        cells.append([("avg ", DIM), (f"{avg:.0f}%", DIM)])
    if s.get("mem_util") is not None:
        cells.append([("mem ", DIM), (f"{s['mem_util']:.0f}%", 0)])
    if s.get("media_activity") is not None:
        cells.append([("media ", DIM), (f"{s['media_activity']}%", 0)])
    vt, vu = s.get("vram_total"), s.get("vram_used")
    if vt:
        fr = (vu or 0) / vt
        c = CRIT if fr >= 0.92 else WARN if fr >= 0.80 else OK
        cells.append([("vram ", DIM)] + _bar(vu or 0, vt, bw, c)
                     + [(f" {int(vu or 0)}/{int(vt)} MiB", 0)])
    temps = s.get("temp") or {}
    crit = s.get("temp_crit") or {}
    for label, v in temps.items():
        if v is not None:
            cells.append([(f"{label} ", DIM), (f"{v}°C", _temp_attr_th(v, crit.get(label)))])
    if s.get("sclk"):
        cells.append([("core ", DIM), (f"{s['sclk']} MHz", 0)])
    if s.get("mclk"):
        cells.append([("vmem ", DIM), (f"{s['mclk']} MHz", 0)])
    if s.get("fclk"):
        cells.append([("fclk ", DIM), (f"{s['fclk']} MHz", 0)])
    if s.get("socclk"):
        cells.append([("soc ", DIM), (f"{s['socclk']} MHz", 0)])
    if s.get("power") is not None:
        cap = f"/{s['power_cap']:.0f}" if s.get("power_cap") else ""
        cells.append([("power ", DIM), (f"{s['power']:.0f}{cap} W", 0)])
    if s.get("voltage"):
        cells.append([("vdd ", DIM), (f"{s['voltage']} mV", 0)])
    if s.get("vr_gfx_mv") or s.get("vr_soc_mv"):
        cells.append([("vrm ", DIM),
                      (f"gfx {s.get('vr_gfx_mv') or '—'}/soc {s.get('vr_soc_mv') or '—'} mV", DIM)])
    if s.get("fan_rpm"):
        pct = f" ({s['fan_pct']}%)" if s.get("fan_pct") is not None else ""
        cells.append([("fan ", DIM), (f"{s['fan_rpm']} rpm{pct}", 0)])
    if s.get("freq_max"):
        cells.append([("range ", DIM),
                      (f"{s.get('freq_min')}–{s.get('freq_max')} eff {s.get('freq_eff')} MHz", DIM)])
    rows = _flow(cells, inner)
    if s.get("gtt_used") is not None and s.get("gtt_total"):
        st = _gtt_status(s["gtt_used"], s.get("vram_free"))
        if st:
            rows.append([("gtt   ", DIM), (st[0], st[1])])
    if s.get("suspended"):
        rows.append([("state ", DIM), ("runtime suspended", DIM)])
    if s.get("throttle"):
        rows.append([("⚠ ", WARN), ("throttling: " + ", ".join(s["throttle"]), WARN)])
    return rows


def _cpu_rows(c, width):
    """The CPU packed onto as few lines as the width allows — utilisation (with its
    session average), frequency, every temperature the sensor exposes, package power
    and the load average, flowing horizontally instead of one metric per line."""
    inner = width - 4
    bw = max(8, min(14, inner // 6))
    u = c.get("util")
    cells = [[("use ", DIM)] + _bar(u or 0, 100, bw, _util_attr(u)) + [(f" {u or 0:>3.0f}%", 0)]]
    _mn, avg, _mx = _extremes("cpu_util")
    if avg is not None:
        cells.append([("avg ", DIM), (f"{avg:.0f}%", DIM)])
    if c.get("freq"):
        cells.append([("freq ", DIM), (f"{c['freq']} MHz", 0)])
    for label, v in (c.get("temps") or {}).items():
        if v is not None:
            cells.append([(f"{label} ", DIM), (f"{v}°C", _temp_attr(v))])
    if c.get("power") is not None:
        cells.append([("power ", DIM), (f"{c['power']:.0f} W", 0)])
    elif c.get("rapl_present"):
        cells.append([("power ", DIM), ("— needs root", DIM)])
    la = c.get("load") or []
    if la:
        cells.append([("load ", DIM), (" ".join(la), 0), (" 1·5·15m", DIM)])
    return _flow(cells, inner)


def _mem_rows(m, width):
    inner = width - 4
    bw = max(8, min(14, inner // 6))
    used = m["total"] - m["free"]
    c = CRIT if m["free"] < 1200 else WARN if m["free"] < 2500 else OK
    cells = [[("ram ", DIM)] + _bar(used, m["total"] or 1, bw, c)
             + [(f" {m['free']} free / {m['total']} MiB", 0)]]
    cells.append([("cached ", DIM), (f"{m['cached']} MiB", 0)])
    cells.append([("buffers ", DIM), (f"{m['buffers']}", 0)])
    cells.append([("dirty ", DIM), (f"{m['dirty']}", 0)])
    cells.append([("committed ", DIM), (f"{m['committed']} MiB", 0)])
    if m["swap_total"]:
        cells.append([("swap ", DIM), (f"{m['swap_used']}/{m['swap_total']} MiB", 0)])
        if m.get("zram_orig"):
            cells.append([("zram ", DIM),
                          (f"{m['zram_orig']}→{m['zram_compr']} MiB ({m['zram_ratio']:.1f}x)", 0)])
        if m.get("disk_swap_total"):
            cells.append([("on-disk ", DIM), (f"{m['disk_swap']}/{m['disk_swap_total']} MiB", 0)])
    return _flow(cells, inner)


_power_peak = [0.0]


def _fmt_dur(seconds):
    """Seconds as a compact h/m/s string for the session clock."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def system_power(gpus_data, cpu):
    """Total watts the machine draws right now, summed from every meter it actually exposes —
    one term per GPU the tool discovered, plus the CPU package via RAPL. Nothing about the
    hardware is named or assumed: it adds whatever discover_gpus() returned, so the same code
    is right on a one-GPU laptop and on a four-GPU server, and picks up a new card for free.
    Returns (watts, missing): `missing` names any component that HAS a meter the tool could
    not read (the CPU when RAPL is root-only and not yet opened), so a figure covering only
    part of the box is never passed off as the whole-system draw."""
    total, missing = 0.0, []
    for s in gpus_data:
        if s.get("power") is not None:
            total += s["power"]
    if cpu.get("power") is not None:
        total += cpu["power"]
    elif cpu.get("rapl_present"):
        missing.append("CPU (needs root)")
    return total, missing


def _power_rows(gpus_data, cpu, llamas, width):
    """Electrical overview laid out horizontally: each readable watt meter as a compact
    cell (the point is the whole picture, not one reading), then a prominent TOTAL with
    its session average and peak, and — when a server is generating — the fleet's energy
    efficiency: how many tokens per second we get for each kilowatt drawn (t/s per kW,
    which is the same number as tokens per kJ)."""
    inner = width - 4
    total, missing = system_power(gpus_data, cpu)
    parts = []
    for s in gpus_data:
        if s.get("power") is not None:
            parts.append((s["vendor"], s["power"], s.get("power_cap")))
    if cpu.get("power") is not None:
        parts.append(("CPU", cpu["power"], None))
    if not parts:
        return [[("no readable power sensors (GPU hwmon / CPU RAPL, the latter often "
                  "needs root)", DIM)]]
    _power_peak[0] = max(_power_peak[0], total)
    cells = []
    for label, wv, cap in parts:
        span = cap or 300
        col = CRIT if wv >= span * 0.95 else WARN if wv >= span * 0.80 else OK
        cap_txt = f"/{cap:.0f}" if cap else ""
        cells.append([(f"{label} ", DIM), (f"{wv:.0f}{cap_txt} W", col)])
    rows = _flow(cells, inner)
    _pmn, pavg, ppk = _extremes("power_series")
    tot = [("total ", DIM), (f"{total:.0f} W", curses.A_BOLD)]
    if pavg:
        tot += [("  avg ", DIM), (f"{pavg:.0f} W", 0)]
    tot += [("  peak ", DIM), (f"{_power_peak[0]:.0f} W", 0)]
    rows.append(tot)
    # session clock + cumulative energy drawn since launch, then live efficiency
    line = []
    if _session["start"] is not None:
        wh = _session["energy_j"] / 3600.0
        etxt = f"{wh / 1000:.2f} kWh" if wh >= 1000 else f"{wh:.1f} Wh"
        line += [("session ", DIM), (_fmt_dur(time.monotonic() - _session["start"]), 0),
                 ("  energy used ", DIM), (etxt, curses.A_BOLD)]
    sum_tg = sum((d.get("tg") or 0) for d in (llamas or []) if d.get("alive"))
    if sum_tg > 0 and total > 0:
        eff = sum_tg / (total / 1000.0)                # t/s per kW  ==  tokens/kJ
        if line:
            line.append(("      ", 0))
        # whole-system efficiency: EVERY server's tokens over the TOTAL system draw — as
        # opposed to the per-session tok/kJ shown in the llama panel. If a component's meter
        # could not be read the total is short by it, so say so instead of overstating.
        line += [("system efficiency ", DIM), (f"{eff:.0f} t/s per kW", OK),
                 (f"  ({sum_tg:.1f} t/s / {total:.0f} W total", DIM)]
        if missing:
            line.append((" · excludes " + ", ".join(missing), WARN))
        line.append((")", DIM))
    if line:
        rows.append(line)
    return rows


def _llama_config_rows(cfg, width):
    """The active server settings, organized by theme (only what differs from the
    defaults reaches the command line). Each theme's settings wrap onto as many lines as
    the width needs, indented under the label, so nothing is ever clipped off the edge."""
    inner = width - 4
    lab = 12
    rows = []
    for group, items in cfg.items():
        shown = [(k, v) for k, v in items if v not in (None, "", False)]
        if not shown:
            continue
        cells = [[(f"{k} ", DIM), ("on" if v is True else str(v), 0)] for k, v in shown]
        for i, r in enumerate(_flow(cells, inner - lab)):
            prefix = [(f"{group:<{lab}}", SER)] if i == 0 else [(" " * lab, 0)]
            rows.append(prefix + r)
        # A sampler value on the command line is what the server falls back to, NOT what
        # the model is running at: a client that puts temperature or repeat_penalty in the
        # request body wins, and /proc cannot see that. Measured 2026-08-25 against a
        # server whose slot went from the values one client sent to another client's the
        # moment the second one called. True and read as false is the expensive kind of
        # wrong, and the correction costs one dim line that only appears when it applies.
        if group == "sampling":
            rows.append([(" " * lab, 0),
                         ("↑ server defaults — a request carrying its own overrides them",
                          DIM)])
    return rows or [[("server started with all defaults", DIM)]]


def _llama_proc_rows(procs, width):
    """Per-process resident memory and the VRAM/GTT split from fdinfo — non-dumpable
    binaries (file capabilities) can't be read, and it says why."""
    if not procs:
        return [[("no llama.cpp process running", DIM)]]
    rows = []
    for p in procs[:6]:
        name = (p["model"] or p["name"])[:26]
        seg = [(f"{p['pid']:>7} ", DIM), (f"{name:<26}", 0), (f" RSS {p['rss']}M", DIM)]
        if not p["read"]:
            if p.get("error") == "oserror":
                if p["caps"]:
                    why = f"{p['caps'][0]} → non-dumpable, needs root"
                elif os.geteuid() != 0:
                    why = "owned by another user — run llamagputop with sudo"
                else:
                    why = "permission denied"
            else:
                why = "not reported by driver (NVIDIA?)" if p.get("gpu_accel") else "CPU only (no VRAM)"
            seg += [("   VRAM ", DIM), ("—", DIM), (f"  ({why})", DIM)]
        else:
            seg += [("   VRAM ", DIM), (f"{p['vram']}M", 0)]
            if p["gtt"] > 200:
                seg += [("   ⚠ GTT ", CRIT), (f"{p['gtt']}M in host RAM", CRIT)]
            elif p["gtt"]:
                seg += [("   GTT ", DIM), (f"{p['gtt']}M", DIM)]
        rows.append(seg)
    return rows


def _col_segs(text, attr, colw):
    """One tile column: text in `attr`, padded to exactly colw cells and always
    leaving at least one blank so adjacent columns never touch."""
    t = _wtrim(text, colw - 1)
    pad = colw - _wwidth(t)
    return [(t, attr)] + ([(" " * pad, 0)] if pad > 0 else [])


def _val_segs(val, unit, colw):
    """A tile's value line: bold value, dim unit, padded to colw."""
    used = _wwidth(val) + 1 + _wwidth(unit)
    segs = [(val, curses.A_BOLD), (f" {unit}", DIM)]
    if used < colw:
        segs.append((" " * (colw - used), 0))
    return segs


def _tiles_rows(gpus_data, cpu, mem, servers, width):
    """The handful of numbers to read first, as columns of label / bold value / note.
    There is a PREFILL and a GEN tile for EVERY running server (each with its own
    min·avg·max, labelled by port when more than one runs); VRAM names the tighter card;
    POWER carries the session peak; a spill tile appears only when VRAM has overflowed to
    host memory. Laid out horizontally, as many columns as the width fits."""
    free, which = None, ""
    for s in gpus_data:
        if s.get("vram_total"):
            fr = s["vram_total"] - (s.get("vram_used") or 0)
            if free is None or fr < free:
                free, which = fr, s["vendor"]
    total_w = sum((s.get("power") or 0) for s in gpus_data) + (cpu.get("power") or 0)
    gtt = max([(s.get("gtt_used") or 0) for s in gpus_data], default=0)
    multi = len(servers) > 1

    def speed(label, key, cur, last, dec):
        mn, mean, mx = _extremes(key)
        live = cur is not None
        val = _n(cur, dec) if live else _n(last, dec)
        # the "last" marker rides on the UNIT, not on the note: the note is 26 cells wide
        # and min·avg·max already fills it, so a marker put there is the first thing
        # trimmed off — and a past reading that loses its marker is a past reading being
        # read as the current one, which is the defect this whole pass is about
        unit = "t/s last" if (not live and last is not None) else "t/s"
        if mn is None:
            return (label, val, unit, "" if live or last is not None else "no measurement yet")
        return (label, val, unit,
                f"min·avg·max {mn:.{dec}f}·{mean:.{dec}f}·{mx:.{dec}f}")

    tiles = []
    for d in servers:
        port = d["port"]
        sfx = f" :{port}" if multi else ""
        tiles.append(speed("PREFILL" + sfx, f"pp_gen@{port}", d.get("pp"), d.get("pp_last"), 0))
        tiles.append(speed("GEN" + sfx, f"tg_gen@{port}", d.get("tg"), d.get("tg_last"), 1))
    if free is not None:
        tiles.append((f"VRAM FREE·{which}", f"{free}", "MiB", "tighter card"))
    _pmn, _pav, ppk = _extremes("power_series")   # recorded in _collect, always current
    tiles.append(("POWER", f"{total_w:.0f}", "W", f"peak {(ppk or total_w):.0f} W"))
    if gtt >= 1000:
        tiles.append(("SPILL→HOST", f"{gtt}", "MiB", "read over PCIe"))

    colw = 27
    per = max(1, width // colw)
    rows = []
    for i in range(0, len(tiles), per):
        grp = tiles[i:i + per]
        lab, val, note = [], [], []
        for (l, v, u, n) in grp:
            lab += _col_segs(l, DIM, colw)
            val += _val_segs(v, u, colw)
            note += _col_segs(n, DIM, colw)
        rows += [lab, val, note]
    return rows


_SPARK = "·▁▂▃▄▅▆▇█"


def _nice_step(x):
    """The 1-2-5 step ladder every plotting axis uses. Snapping the ends of a scale to a
    round step is what stops it from twitching: without it the low and high move a little
    on EVERY tick, and a mapping that shifts every tick is a graph that flickers instead of
    scrolling."""
    if not (isinstance(x, (int, float)) and math.isfinite(x)) or x <= 0:
        return 1.0
    e = math.floor(math.log10(x))
    f = x / 10 ** e
    return (1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10) * 10 ** e


def _sparkline(key, width, maxv):
    """A STRIP CHART of the last `width` samples: one column per tick, the newest on the
    right, sliding left by exactly one column each time. Returns the scale it was drawn
    against, which the caller prints.

    WHY IT SLIDES INSTEAD OF RE-BUCKETING. The previous version folded the WHOLE history
    into `width` columns, so every column stood for a bucket whose boundaries moved as the
    history grew: the picture did not scroll, it re-quantised in place. MEASURED
    2026-08-26 over 900 ticks, using "is this frame the previous one shifted left by one
    column" as the definition of scrolling: the folded version satisfied it 7.9% of the
    time and 60.8% of its frames were IDENTICAL to the one before — frozen, then a jump.
    One column per tick satisfies it 98.7%.

    WHY THE SCALE IS SNAPPED. Fitting to the window keeps the detail (a series in a narrow
    band uses 7 of 9 glyphs instead of 3), but recomputing the exact min and max every tick
    moves the mapping constantly. Snapping both ends to a 1-2-5 step holds the mapping
    still until the range really changes: on the same run, 91.3% -> 98.7% scrolling with no
    loss of detail. A slow drift of the ends was tried first and measured WORSE than doing
    nothing (82.2%, and 56.6% on a bursty series) — a bound that creeps every tick is the
    flicker, not the cure.

    A percentage keeps its absolute 0..maxv range: 60% drawn full-height would lie about
    the quantity. Glyphs: `·` is a true zero and nothing else; on a fitted row the bottom
    of the range is `▁`, since a dot there reads as "nothing happened" when what happened
    is "the lowest value in view".
    """
    s = HIST.get(key)
    if not s or width <= 0:
        return [(" " * max(0, width), DIM)], None, None
    # defence in depth: record() already refuses non-finite values, but the render must
    # not be the thing that dies if one ever arrives by another door
    v = [x if isinstance(x, (int, float)) and math.isfinite(x) else 0
         for x in list(s)[-width:]]
    pad = width - len(v)                       # a young series fills in from the right
    if maxv:                                   # absolute scale — percentages
        lo, hi = 0.0, float(maxv)
        t = "".join(_SPARK[int(max(0, min(8, x / hi * 8)))] for x in v)
    else:
        lo, hi = float(min(v)), float(max(v))
        if hi > lo:
            step = _nice_step((hi - lo) / 8.0)
            lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        span = hi - lo
        if span <= 0:                          # genuinely flat: draw it flat, do not invent
            t = ("·" if hi == 0 else "▄") * len(v)
        else:
            t = "".join("·" if x == 0 else
                        _SPARK[1 + int(max(0, min(7, round((x - lo) / span * 7))))]
                        for x in v)
    return [(" " * max(0, pad), DIM), (t, SER)], lo, hi


def _trend_rows(gpus_data, cpu, llamas, width):
    """Rolling recent history of the metrics that actually move, so you can see WHEN
    something changed — not just its value now. Every series is live even between
    requests: per-GPU utilisation and VRAM fill, CPU utilisation, total power and free
    RAM, with each server's generation and prefill speed on top. Each line is the
    sparkline plus the current value and the session peak."""
    ports = sorted(_port_seen)          # live now, or seen within the last two minutes
    multi = len(ports) > 1
    series = []
    for port in ports:
        sfx = f" :{port}" if multi else ""
        # only chart a speed that has actually occurred on this server, so idle
        # embedding / rerank servers don't fill the panel with flat-zero lines
        _m, _a, tgpk = _extremes(f"tg_series@{port}")
        _m, _a, pppk = _extremes(f"pp_series@{port}")
        if tgpk:
            series.append((f"gen t/s{sfx}", f"tg_series@{port}", None, 1))
        if pppk:
            series.append((f"prefill t/s{sfx}", f"pp_series@{port}", None, 0))
    for i, s in enumerate(gpus_data):
        series.append((f"{s['vendor']} util %", f"gpu{i}_util", 100, 0))
    for i, s in enumerate(gpus_data):
        if s.get("vram_total"):
            series.append((f"{s['vendor']} vram %", f"gpu{i}_vram", 100, 0))
    series.append(("cpu util %", "cpu_util", 100, 0))
    series.append(("power W", "power_series", None, 0))
    series.append(("ram free MiB", "ram_series", None, 0))
    # the label column fits the LONGEST label (e.g. "prefill t/s :8080"), so every
    # sparkline and value lines up in one clean column no matter how many servers run
    labw = max([15] + [_wwidth(lab) + 1 for lab, *_ in series])
    sw = max(16, width - labw - 38)
    rows = []
    for lab, key, mx, dec in series:
        hs = HIST.get(key)
        cur = hs[-1] if hs else 0
        bar, lo, hi = _sparkline(key, sw, mx)
        seg = [(f"{lab:<{labw}}", DIM)] + bar + [(f" {cur:>7.{dec}f}", 0)]
        # PEAK IS OVER THE WHOLE SESSION, and it is back on every row. Replacing it with
        # the bar's own range was a bad trade: on a speed series the floor is almost always
        # zero (idle), so "0.0…35.9" is "peak 35.9" with noise in front of it, and the one
        # number worth keeping across a long run is the highest it ever went.
        _mn, _avg, peak = _extremes(key)
        if peak is not None:
            seg += [(f"  peak {peak:.{dec}f}", DIM)]
        # the bar shows the LAST `sw` ticks, so its floor is not the session's. Say where
        # that floor is whenever it is not zero: without it a narrow band high above zero
        # is drawn full-height and reads as if it had started from nothing.
        if not mx and lo:
            seg += [(f"  base {lo:.{dec}f}", DIM)]
        rows.append(seg)
    return rows


def _n(v, dec=0):
    """A number, or an em dash. The whole llama panel used to print 0 for "no source
    said anything", which is the one thing this program's own docstring forbids."""
    return "—" if v is None else f"{v:.{dec}f}"


def _ctx_text(d):
    """/slots reports the per-SLOT context, the command line carries the total, and with
    -np 2 they differ by a factor of two under the same word (measured on :7795, 6144
    against 12288). Show both whenever they differ, so neither panel silently contradicts
    the other."""
    per, tot, sl = d.get("ctx"), d.get("ctx_total"), d.get("slots")
    if not per:
        return f"ctx {tot}" if tot else "ctx —"
    if tot and tot != per:
        return f"ctx {per}/slot · {tot} total" + (f" ({sl} slots)" if sl and sl > 1 else "")
    return f"ctx {per}" + (f" ×{sl} slots" if sl and sl > 1 else "")


def _why(d, which):
    """Why a speed is unavailable — a dash with no reason just moves the puzzle."""
    if d.get("stale"):
        return "server did not answer"
    if d.get("slots_off"):
        return "slots endpoint off"
    if which == "pp" and d.get("metrics_off"):
        return "idle; needs --metrics for completed prefills"
    return "idle"


def _llama_rows(d, width):
    inner = width - 4
    rows = []
    ET = 9
    ph = d.get("phase", "?")
    port = d.get("port", "")
    rows.append([(f"{'status':<{ET}}", DIM), (ph, OK if ph == "generating" else 0),
                 (f"   {_ctx_text(d)}", DIM),
                 (f"   active {_n(d.get('active'))}, queued {_n(d.get('queued'))}", DIM)]
                + ([("   ⚠ stale", WARN)] if d.get("stale") else []))
    # prefill, generation (each live + windowed median±stddev) and the session lifetime
    # average (always available from the cumulative counters), packed onto one line
    speed = []
    # THREE distinct states, and collapsing them is what made this panel lie: a LIVE rate
    # (the request running right now), the LAST rate (the request that just finished —
    # true, but past, so it says so), and NOT MEASURABLE (no source exists on this build).
    for label, cur, last, mkey, dec in (("prefill", d.get("pp"), d.get("pp_last"), "pp", 0),
                                        ("gen", d.get("tg"), d.get("tg_last"), "tg", 1)):
        mp, dp = median_dev(f"{mkey}_gen@{port}")
        if cur is not None:
            cell = [(f"{label} ", DIM), (f"{cur:.{dec}f}", 0), (" t/s", DIM)]
        elif last is not None:
            cell = [(f"{label} ", DIM), (f"{last:.{dec}f}", DIM), (" t/s last", DIM)]
        else:
            cell = [(f"{label} ", DIM), ("—", DIM), (f" ({_why(d, mkey)})", DIM)]
        if mp is not None:
            cell.append((f" (med {mp:.{dec}f}±{dp:.{dec}f})", DIM))
        speed.append(cell)
    if d.get("pp_life") or d.get("tg_life"):
        speed.append([("session ", DIM),
                      (f"{_n(d.get('pp_life'))}/{_n(d.get('tg_life'), 1)}", 0),
                      (" t/s avg", DIM)])
    elif d.get("metrics_off"):
        speed.append([("metrics off", WARN),
                      (" — start with --metrics for reuse, queue and session averages", DIM)])
    rows += _flow(speed, inner)
    # kv fill, how deep the context has ever gone, the generation budget, prompt reuse
    if d.get("kv") is not None:
        kv = d["kv"]
        c = CRIT if kv >= 0.95 else WARN if kv >= 0.85 else OK
        bw = max(8, min(16, inner // 6))
        kv_cells = [[("kv ", DIM)] + _bar(kv, 1.0, bw, c)
                    + [(f" {kv * 100:.1f}%  {d.get('kv_used', 0)}/"
                        f"{d.get('kv_cap') or d.get('ctx') or 0} tok", 0)]]
        ctx, seen = d.get("ctx") or 0, d.get("max_tok") or 0
        if ctx and seen:
            q = seen / ctx
            kv_cells.append([("max seen ", DIM), (f"{seen}", WARN if q < 0.25 else 0),
                             (f" ({q * 100:.0f}% of alloc)", DIM)])
        bg = d.get("budget", 0)
        if d.get("phase") == "generating" and bg:
            q = d.get("decoded", 0) / bg
            kv_cells.append([("budget ", DIM),
                             (f"{d.get('decoded', 0)}/{bg}", WARN if q > 0.9 else 0)])
        if d.get("reuse") is not None:
            ri = d["reuse"]
            kv_cells.append([("prompt reuse ", DIM), (f"{ri * 100:.1f}%", OK if ri > 0.3 else 0)])
        rows += _flow(kv_cells, inner)
    elif d.get("reuse") is not None:
        ri = d["reuse"]
        rows.append([("prompt   ", DIM), ("reused ", DIM), (f"{ri * 100:.1f}%", OK if ri > 0.3 else 0)])
    # reasoning format detected at runtime from the slot data, not the cmdline
    rf = d.get("reasoning_format")
    if rf and rf not in ("none", ""):
        rows.append([("reasoning ", DIM), (rf, OK)])
    # speculative head + how well it drafts, on one line; per-position beneath
    spec = []
    if d.get("spec_type") not in (None, "", "none") or d.get("spec_head"):
        typ = d.get("spec_type") or "?"
        head = d.get("spec_head") or ("native (in model)" if "mtp" in typ else "—")
        cell = [("head ", DIM), (typ, OK), (" ", 0), (head, 0)]
        if d.get("spec_nmax"):
            cell += [(" n-max ", DIM), (str(d["spec_nmax"]), 0)]
        spec.append(cell)
    if d.get("spec") is not None:
        sp = d["spec"]
        c = CRIT if sp < 0.20 else WARN if sp < 0.50 else OK
        cell = [("draft ", DIM), (f"{sp * 100:.1f}%", c), (" accepted", DIM)]
        if d.get("tok_step"):
            ts = d["tok_step"]
            cell += [("  ", 0), (f"{ts:.2f}", OK if ts >= 1.5 else WARN), (" tok/step", DIM)]
        if d.get("spec_stale"):
            cell += [("  ", 0), ("⚠ frozen", WARN)]
        spec.append(cell)
    rows += _flow(spec, inner)
    if d.get("spec_pos"):
        rows.append([("per pos  ", DIM),
                     (" ".join(f"{v * 100:.0f}%" for v in d["spec_pos"]), 0)])
    w = d.get("power_w") or 0
    _tg = d.get("tg") or 0
    if d.get("phase") == "generating" and _tg > 0 and w > 0:
        rows.append([(f"{'energy':<{ET}}", DIM),
                     (f"{_tg:.1f} t/s at {w:.0f} W on its GPUs = ", DIM),
                     (f"{_tg / w * 1000:.0f} tok/kJ", OK),
                     ("   this session (system draw is in power)", DIM)])
    return rows


_scroll = [0]


def draw(w, gpus_data, cpu, mem, llamas, cfgs, procs):
    w.erase()
    h, W = w.getmaxyx()
    if W < 62 or h < 8:
        _put(w, 0, 0, [("terminal too small", 0)], W)
        _put(w, 1, 0, [(f"need at least 62x8, have {W}x{h}", DIM)], W)
        w.refresh()
        return
    # header (fixed)
    alive = [d for d in llamas if d.get("alive")]
    primary = next((d for d in alive if d.get("phase") == "generating"), None) \
        or (alive[0] if alive else None)
    title = f" llamagputop · {len(gpus_data)} GPU" + ("s" if len(gpus_data) != 1 else "")
    if len(alive) > 1:
        title += f" · llama: {len(alive)} servers"
    elif primary and primary.get("model"):
        title += f" · llama: {primary['model']}"
    clock = time.strftime(" %H:%M:%S ")
    _put(w, 0, 0, [(title[:W - len(clock) - 1].ljust(W - len(clock) - 1) + clock,
                    curses.A_REVERSE)], W)
    try:
        w.chgat(0, 0, W, curses.A_REVERSE)
    except curses.error:
        pass
    # build every block, then render them into an off-screen pad we scroll through
    bw = W - 2
    multi = len(llamas) > 1
    blocks = [("summary", _tiles_rows(gpus_data, cpu, mem, alive, bw), "")]
    for i, s in enumerate(gpus_data):
        name = f"{s['vendor']} · {s['name']}"
        if s.get("backend_dev"):
            name += f" ({s['backend_dev']})"
        blocks.append((name, _gpu_rows(s, bw, i), pcie_text(s.get("pcie"))))
    blocks.append((f"CPU · {cpu.get('name', '')}", _cpu_rows(cpu, bw),
                   f"{cpu.get('ncpu', 0)} threads"))
    blocks.append(("memory", _mem_rows(mem, bw), ""))
    blocks.append(("power", _power_rows(gpus_data, cpu, llamas, bw), ""))
    # one panel per server; the port/pid identifiers appear only when several run
    for d in llamas:
        port = d["port"]
        flavor = d.get("flavor", "llama.cpp")
        ltitle = f"{flavor} :{port} · {d.get('model', '')}" if multi else flavor
        lnote = f"pid {d.get('pid', '')}" if multi else ""
        blocks.append((ltitle, _llama_rows(d, bw), lnote))
        cfg = cfgs.get(port)
        if cfg is not None:          # {} is "no flags at all", which the panel still says
            blocks.append((f"config :{port}" if multi else "server config",
                           _llama_config_rows(cfg, bw), ""))
    if llamas:
        blocks.append(("llama processes", _llama_proc_rows(procs, bw), ""))
    # the bar is a WINDOW now, not the whole session, so the note says its resolution:
    # a reader who does not know how much time a column covers cannot read the shape
    blocks.append(("trend", _trend_rows(gpus_data, cpu, llamas, bw),
                   f"{_interval():g}s per column · peak is the whole session"))
    total = sum(len(r) + 2 for _, r, _ in blocks)
    body_top, body_bot = 1, h - 2
    body_h = body_bot - body_top + 1
    maxscroll = max(0, total - body_h)
    _scroll[0] = max(0, min(_scroll[0], maxscroll))
    pad = curses.newpad(max(total, body_h) + 1, W)
    y = 0
    for btitle, rows, note in blocks:
        height = len(rows) + 2
        _box(pad, y, 1, bw, height, btitle, note)
        for i, seg in enumerate(rows):
            _put(pad, y + 1 + i, 3, seg, 1 + bw - 1)
        y += height
    # footer with scroll hints
    more = ""
    if _scroll[0] > 0:
        more += "↑more "
    if _scroll[0] < maxscroll:
        more += "↓more "
    foot = f" q quit · ↑↓ PgUp/PgDn scroll · +/- rate {_interval():g}s · z reset   {more}"
    _put(w, h - 1, 0, [(foot.ljust(W - 1), curses.A_REVERSE)], W)
    try:
        w.chgat(h - 1, 0, W, curses.A_REVERSE)
    except curses.error:
        pass
    # Stage the screen, then paint. stdscr holds the header, the footer and a blank
    # body; the pad holds the body. Both are staged (noutrefresh) and a single
    # doupdate paints them together — with the pad staged LAST so it wins the body
    # region. Doing pad.refresh() before stdscr.refresh() (the obvious order) is the
    # trap: stdscr's erased body would then wipe the pad back out on the next update.
    w.noutrefresh()
    try:
        pad.noutrefresh(_scroll[0], 0, body_top, 0, body_bot, W - 1)
    except curses.error:
        pass
    curses.doupdate()


# ------------------------------------------------------------------- cadence
_RATES = [0.5, 1.0, 2.0, 5.0]
_rate = [1]


def _interval():
    return _RATES[_rate[0]]


def _collect(gpus, explicit_port=None, lfeed=None):
    gpus_data = [g.sample() for g in gpus]
    for g, s in zip(gpus, gpus_data):
        s["backend_dev"] = getattr(g, "backend_dev", None)
        s["pci_addr"] = getattr(g, "pci_addr", None)
    for i, s in enumerate(gpus_data):
        record(f"gpu{i}_util", s.get("util") or 0)
        if s.get("vram_total"):
            record(f"gpu{i}_vram", 100.0 * (s.get("vram_used") or 0) / s["vram_total"])
    cpu = cpu_sample()
    mem = mem_sample()
    # the TUI hands in a feed thread and never waits; --once/--line have no loop to
    # protect and read directly
    llamas = lfeed.data if lfeed is not None else sample_llama_fleet(explicit_port)
    # hardware trend series — always live, so the trend panel is never blank
    total_w, _ = system_power(gpus_data, cpu)      # single source of truth for the whole-box draw
    record("power_series", total_w)
    record("ram_series", mem["free"])
    record("cpu_util", cpu.get("util") or 0)
    # cumulative energy this session: integrate power over each elapsed tick (resets
    # naturally on relaunch, since these are per-process globals)
    now = time.monotonic()
    if _session["start"] is None:
        _session["start"] = now
    if _session["t"] is not None:
        _session["energy_j"] += total_w * (now - _session["t"])
    _session["t"] = now
    procs = llama_processes() if llamas else []
    cfgs = {}
    for d in llamas:
        port = d["port"]
        # charge THIS server's energy line to the power of the GPUs it runs on (its -dev
        # list; all cards when -dev is absent) — not the whole-box draw, which is the power
        # panel's job. So the per-session figure and the system figure are genuinely different
        # numbers: the session one omits the CPU and any card this server does not use.
        _devs = llama_devices_from_cmdline(port)
        _gp = [s for s in gpus_data if (not _devs) or (s.get("backend_dev") in _devs)]
        if _devs and not _gp:                 # -dev set but unmappable (no vulkaninfo) → all cards
            _gp = gpus_data
        d["power_w"] = sum((s.get("power") or 0) for s in _gp)
        cfgs[port] = llama_settings_from_cmdline(port) if d.get("alive") else None
        # each server's speed histories are tied to its model: reset on a swap
        model = d.get("model") or ""
        if model and model != _model_ports.get(port):
            # reset only the MEDIAN histories, so median / min·avg·max describe the new
            # model. The TREND sparklines (…_series) are deliberately NOT reset — they
            # roll continuously across model swaps, which is exactly what a "recent
            # history" view wants, and is why the trend no longer blanks when the
            # campaign moves to the next model.
            for k in (f"pp_gen@{port}", f"tg_gen@{port}"):
                HIST.pop(k, None)
            _last_pp.pop(port, None)
            _model_ports[port] = model
        # A PREFILL IS AN EVENT, NOT A STATE, and that distinction decides whether its
        # trend line exists at all. Generation spans many ticks, so charting the live rate
        # draws it correctly. A prefill on a GPU is over before the second tick — measured
        # 2026-08-25, 4001 tokens in 1.94 s — so the live slope is almost always absent and
        # a series fed only from it stays at zero forever, peaks at zero, and is dropped by
        # the "only chart a speed that has actually occurred" test below. The line was
        # therefore invisible on exactly the servers fast enough to be worth watching.
        # So: chart the live rate while one is running, and the COMPLETED rate on the tick
        # it first lands — one spike per prefill, which is what actually happened.
        pp_live, pp_done = d.get("pp"), d.get("pp_last")
        fresh = pp_done if (pp_done and abs(pp_done - _last_pp.get(port, 0.0)) > 1e-6) else None
        pp_now = pp_live if pp_live is not None else fresh
        if d.get("alive"):
            # a series records 0 while idle — that is a true statement about the server
            # (it is producing nothing) and it is what keeps the trend line continuous
            record(f"tg_series@{port}", d.get("tg") or 0)
            record(f"pp_series@{port}", pp_now or 0)
        # the MEDIAN window, unlike the trend, must only ever see real measurements
        if d.get("phase") == "generating" and (d.get("tg") or 0) > 0:
            record(f"tg_gen@{port}", d["tg"])
        if pp_now and abs(pp_now - _last_pp.get(port, 0.0)) > 1e-6:
            record(f"pp_gen@{port}", pp_now)
            _last_pp[port] = pp_now
    # remember every port that currently has a live server, so the trend keeps its line
    # (and its rolling history) while the server briefly restarts between runs. A port
    # gone for more than two minutes is genuinely finished and forgotten.
    for d in llamas:
        if d.get("alive"):
            _port_seen[d["port"]] = now
    for port in [p for p, t in list(_port_seen.items()) if now - t > 120]:
        for k in (f"pp_gen@{port}", f"tg_gen@{port}", f"pp_series@{port}", f"tg_series@{port}"):
            HIST.pop(k, None)
        _port_seen.pop(port, None)
        _model_ports.pop(port, None)
        _last_pp.pop(port, None)
    return gpus_data, cpu, mem, llamas, cfgs, procs


def _tui(w, gpus, feeds, port):
    global OK, WARN, CRIT, DIM, SER
    curses.curs_set(0)
    w.nodelay(True)
    w.keypad(True)
    DIM = curses.A_DIM
    if not os.environ.get("NO_COLOR") and curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        for i, c in enumerate((curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED,
                               curses.COLOR_CYAN), 1):
            curses.init_pair(i, c, -1)
        OK, WARN = curses.color_pair(1), curses.color_pair(2)
        CRIT = curses.color_pair(3) | curses.A_BOLD
        SER = curses.color_pair(4)
    else:
        OK, WARN, CRIT, SER = 0, curses.A_BOLD, curses.A_REVERSE, 0
    for f in feeds.values():
        f.start()
    lfeed = _LlamaFeed(port)
    lfeed.start()
    cpu_sample()
    data = _collect(gpus, port, lfeed)
    # Input and data collection are decoupled. getch half-blocks (timeout) so keys —
    # including the multi-byte arrow/PgUp/End escape sequences — are assembled and
    # answered within 100 ms; scrolling redraws IMMEDIATELY from the data already in
    # hand. A full _collect() (which includes a blocking HTTP read of the server) runs
    # only on its own cadence, never in the keypress path — that coupling was what made
    # scrolling stutter and the screen jump. Bare ESC no longer quits, so an arrow key
    # whose escape prefix arrives alone can't kill the program.
    w.timeout(100)
    last_collect = time.monotonic()
    dirty = True
    while True:
        if dirty:
            try:
                draw(w, *data)
            except curses.error:
                pass
            dirty = False
        k = w.getch()
        if k in (ord("q"), ord("Q")):
            break
        elif k in (curses.KEY_DOWN, ord("j")):
            _scroll[0] += 1; dirty = True
        elif k in (curses.KEY_UP, ord("k")):
            _scroll[0] = max(0, _scroll[0] - 1); dirty = True
        elif k == curses.KEY_NPAGE:
            _scroll[0] += 10; dirty = True
        elif k == curses.KEY_PPAGE:
            _scroll[0] = max(0, _scroll[0] - 10); dirty = True
        elif k == curses.KEY_HOME:
            _scroll[0] = 0; dirty = True
        elif k == curses.KEY_END:
            _scroll[0] = 10 ** 6              # clamped to the bottom in draw()
            dirty = True
        elif k in (ord("+"), ord("=")):
            _rate[0] = max(0, _rate[0] - 1); dirty = True
        elif k == ord("-"):
            _rate[0] = min(len(_RATES) - 1, _rate[0] + 1); dirty = True
        elif k in (ord("z"), ord("Z")):
            HIST.clear(); dirty = True
        elif k == curses.KEY_RESIZE:
            dirty = True
        now = time.monotonic()
        if now - last_collect >= _interval():
            data = _collect(gpus, port, lfeed)
            last_collect = now
            dirty = True
    lfeed.stop = True
    for f in feeds.values():
        f.stop = True
    return


def _text_line(gpus_data, cpu, mem, llamas, cfgs=None, procs=None):
    parts = []
    total_w, _ = system_power(gpus_data, cpu)
    for s in gpus_data:
        vram = f"{s.get('vram_used') or 0}/{s.get('vram_total') or 0}" if s.get("vram_total") else "-"
        parts.append(f"{s['vendor']} util {s.get('util') or 0:.0f}% vram {vram} "
                     f"{s.get('temp_main') or '-'}C {s.get('power') or 0:.0f}W")
    parts.append(f"CPU {cpu.get('util') or 0:.0f}% {cpu.get('temp') or '-'}C "
                 f"{('%.0fW' % cpu['power']) if cpu.get('power') is not None else '-'}")
    parts.append(f"RAM {mem['free']}/{mem['total']}")
    parts.append(f"PWR {total_w:.0f}W")
    multi = len(llamas or []) > 1
    for d in (llamas or []):
        if not d.get("alive"):
            continue
        sp = f" spec {d['spec'] * 100:.0f}%" if d.get("spec") is not None else ""
        tag = f":{d['port']} " if multi else ""
        # a LOG line has no room for a column that says "this one is from the request
        # that just ended" — so it says it inline. Writing the last rate bare is the same
        # defect the panel was carrying: a past number wearing the present tense.
        def _sp(live, last, dec):
            if live is not None:
                return _n(live, dec)
            return f"{_n(last, dec)}(last)" if last is not None else "—"
        parts.append(f"llama {tag}{d.get('phase')} "
                     f"pp {_sp(d.get('pp'), d.get('pp_last'), 0)} "
                     f"tg {_sp(d.get('tg'), d.get('tg_last'), 1)}{sp}")
    return time.strftime("%H:%M:%S") + " | " + " | ".join(parts)


def _maybe_enable_rapl():
    """CPU wattage comes from the RAPL energy counter, which most kernels expose only to
    root (the Platypus side-channel mitigation). If it is unreadable and we are on a
    terminal, offer to open it for this session with one sudo call — the same thing the
    old build did. The README's udev rule makes it permanent instead."""
    if not RAPL_CPU or read_int(f"{RAPL_CPU}/energy_uj", -1) >= 0:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    try:
        ans = input("CPU power needs root — enable RAPL reading for this session "
                    "via sudo? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        return
    try:
        subprocess.run(["sudo", "chmod", "-R", "a+r", "/sys/devices/virtual/powercap"],
                       timeout=120)
    except Exception as e:
        print(f"could not enable RAPL ({e}); continuing without CPU power")


def main():
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print("llamagputop — a terminal GPU + llama.cpp inference monitor for Linux.\n")
        print("Usage: llamagputop.py [PORT] [--once | --line | --probe]\n")
        print("  PORT      focus one llama.cpp port; omit to auto-detect EVERY running")
        print("            server (each gets its own panel)")
        print("  --once    print one status line and exit    (for scripts)")
        print("  --line    print a status line every tick    (for logging)")
        print("  --probe   dump the detected hardware and exit")
        print("  -h, --help  this help\n")
        print("TUI keys:  q quit   ↑↓ PgUp/PgDn Home/End scroll   +/- refresh rate   z reset history")
        print("Honors NO_COLOR. No dependencies beyond the Python standard library;")
        print("optional helpers used when present: lspci, intel_gpu_top, nvtop, nvidia-smi.")
        return
    port = next((a for a in argv if a.isdigit()), None)
    gpus, feeds = discover_gpus()
    if "--probe" in argv:
        probe()
        return
    if "--once" in argv or "--line" in argv:
        for f in feeds.values():
            f.start()
        time.sleep(2.5)
        _collect(gpus, port)          # prime the derivatives
        try:
            if "--once" in argv:
                time.sleep(1.1)
                print(_text_line(*_collect(gpus, port)))
            else:
                while True:
                    print(_text_line(*_collect(gpus, port)), flush=True)
                    time.sleep(_interval())
        except KeyboardInterrupt:
            pass
        finally:
            for f in feeds.values():
                f.stop = True
        return
    _maybe_enable_rapl()
    curses.wrapper(_tui, gpus, feeds, port)


if __name__ == "__main__":
    main()

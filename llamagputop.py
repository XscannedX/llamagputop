#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
llamagputop is a terminal GPU and llama.cpp inference monitor for Linux.

What makes it different from other monitors is that besides watching every GPU in the machine, it also reads a running llama.cpp server and shows what actually matters when you serve a model. You get to see the prefill and generation speed, KV cache fill, and speculative decoding metrics like draft head usage and acceptance rates. Those inference stats are tied to the model, so when the model changes they reset.

The design principles are simple. Read at the source. Everything comes from sysfs, proc and the driver, plus a couple of optional helpers used only when present. There are no Python dependencies beyond the standard library. No sensor is believed blindly. Out of range readings are dropped and the last good value is kept. The tool degrades honestly. A missing card or counter shows a dash with a reason, never a fake number.

It is highly portable. It discovers the hardware at runtime, so it runs on any Linux box, not just the one it was written on.

Copyright 2026 XscannedX <xscannedx@gmail.com>. MIT License.
"""
import atexit
import glob
import json
import math
import os
import re
import selectors
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

# How long a held reading stays believable. Holding the last good value across one bad
# tick is the point of sane(); holding it forever is a different thing. With no age at
# all, a single in-range reading meant that a sensor which later died kept replaying
# that number for the rest of the run, styled exactly like a live one. A temperature
# frozen at its last value with nothing on screen saying so is the worst way for an
# overheat display to fail, because the number still looks reasonable.
# Ten seconds: the feed readers further down trust their own caches for six and eight,
# and sane() reads sysfs once per refresh, so ten consecutive refusals are not a blip.
# A module constant rather than a literal, because it is a knob worth turning.
_SANE_TTL = 10.0


def sane(key, value, lo=None, hi=None):
    """A sensor that answers is not a sensor that tells the truth. Out-of-range values are
    dropped in favour of the last good one; if there never was one, or the last one has
    aged past _SANE_TTL, None, which the display turns into a dash, not a misleading
    zero, and not a number the sensor stopped producing minutes ago."""
    if value is not None and (lo is None or value >= lo) and (hi is None or value <= hi):
        _last_good[key] = (value, time.monotonic())
        return value
    held = _last_good.get(key)
    if held is None or time.monotonic() - held[1] > _SANE_TTL:
        return None
    return held[0]


def norm_name(s):
    """Product names for the SAME card differ only in punctuation between sources: pci.ids
    gives 'TigerLake-H GT1 [UHD Graphics]' and nvtop gives that string with round brackets.
    Comparing on a vendor substring instead ('Arc', 'DG2', 'Intel') was measured on
    2026-08-27 to miss integrated graphics entirely, the nvtop name contains none of the
    three, so its utilisation figure was thrown away while it sat there for the taking.
    A vendor word is a proxy for identity; the product name IS the identity. Letters and
    digits only, so bracket style and spacing cannot separate two names for one card."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def which(name):
    return any(os.access(os.path.join(p, name), os.X_OK)
               for p in os.environ.get("PATH", "").split(os.pathsep) if p)


# ------------------------------------------------------------------ PCIe chain
# The real link a card has to the rest of the system is not the card's own node:
# both may sit behind a switch whose inner segment reads wider than the negotiated
# link to the chipset. What matters is the narrowest hop between the card and the
# root, and the root port itself, the only stable value. Re-read every tick; it
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


# PCIe generation from the signalling rate. The kernel prints the rate itself in
# current_link_speed and max_link_speed (pci_speed_string() in drivers/pci/probe.c,
# emitted by current_link_speed_show / max_link_speed_show in drivers/pci/pci-sysfs.c),
# so the generation is a lookup and never a guess; a rate this table does not know
# renders as "gen?" rather than as the nearest plausible number.
_PCIE_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}

_PCI_ADDR = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


def _is_pci_node(path):
    return bool(_PCI_ADDR.match(os.path.basename(path)))


def _is_root_port(path):
    """A PCI node whose parent directory is not a PCI address: here the three root ports
    sit directly under pci0000:00, the host bridge."""
    return _is_pci_node(path) and not _is_pci_node(os.path.dirname(path))


# Where the kernel says whether anything can count AER errors. Module constants so a test
# can point them at a fake tree.
_PCIE_SERVICES_DIR = "/sys/bus/pci_express/devices"
_ACPI_TABLES_DIR = "/sys/firmware/acpi/tables"


def _aer_live(node):
    """(live, why): whether the AER counters under `node` can increment at all, and when
    they cannot, the reason to print instead of their zeros.
    A counter that nothing can increment is a structural zero, the fake number this
    program's own docstring forbids, and on THIS machine every aer_dev_* counter is one.
    Who increments them (drivers/pci/pcie/aer.c, 6.18.26): the root port receives the
    ERR_COR / ERR_NONFATAL / ERR_FATAL messages, the AER port service handles its
    interrupt, find_source_device() locates the device that sent the message and
    aer_print_error() -> pci_dev_aer_stats_incr() bumps THAT device's counters. The only
    other path is firmware-first, aer_recover_work_func() -> pci_print_aer(), fed by APEI
    GHES, which needs an ACPI HEST table. So two conditions, either one enough:
      native: the AER service bound to the root port above the device. Port services are
        devices named "<port>:pcie%03x" of the service bit (drivers/pci/pcie/portdrv.c,
        dev_set_name(device, "%s:pcie%03x", ...)), and portdrv.h numbers them PME 1,
        AER 2, HP 4, DPC 8, BWCTRL 0x10, so the AER service is <root port>:pcie002.
      firmware-first: /sys/firmware/acpi/tables/HEST exists.
    MEASURED on this machine: /sys/bus/pci_express/devices holds only 0000:00:02.0,
    00:04.0, 00:09.0, 00:0b.0, 00:15.0 and 00:15.1 as :pcie010 (the Bandwidth
    Controller), no :pcie002 anywhere, and the tables directory holds APIC BGRT DSDT FACP
    FACS FPDT HPET MCFG and no HEST. Cause of the first: the boot log says "acpi
    PNP0A08:00: _OSC: OS supports [ExtendedConfig Segments MSI EDR HPX-Type3]" then
    "_OSC: not requesting OS control; OS requires [ExtendedConfig ASPM ClockPM MSI]": the
    kernel command line carries pcie_aspm=off, drivers/pci/pcie/aspm.c sets
    aspm_support_enabled = false for "off", and drivers/acpi/pci_root.c refuses to
    request control unless ACPI_PCIE_REQ_SUPPORT (ExtendedConfig|ASPM|ClockPM|MSI) is all
    supported, so the platform keeps every PCIe feature, AER included. Besides that, the
    root ports' `aer` attribute directory is empty (the GPUs' holds the ratelimit files)
    and aer.c's pci_aer_init() sets things up only for a device with the AER extended
    capability, so the RD990 root ports most likely lack the capability altogether; a
    direct confirmation would need `lspci -vvv` as root, which this tool does not run.
    The root port is the topmost PCI node above `node` (the node itself when its parent
    is not a PCI node), so behind an on-card switch the walk passes the upstream port."""
    rp = os.path.realpath(node)
    while _is_pci_node(os.path.dirname(rp)):
        rp = os.path.dirname(rp)
    name = os.path.basename(rp)
    if (os.path.exists(f"{_PCIE_SERVICES_DIR}/{name}:pcie002")
            or os.path.exists(f"{_ACPI_TABLES_DIR}/HEST")):
        return True, None
    return False, (f"not counted: no AER service on root port {name} ({name}:pcie002 absent) "
                   f"and no ACPI HEST table")


# Baseline of the correctable AER total per device, taken on the first read of a run. The
# kernel's counters are cumulative since boot (drivers/pci/pcie/aer.c), so for a monitoring
# session the number that matters is how many arrived since it started, not how many the
# machine has ever seen: a link that took 300 corrected errors during last week's
# re-seating and none since is a healthy link.
_AER_BASE = {}


def pcie_aer(dev_path):
    """The Advanced Error Reporting counters of one PCI device, or None when the kernel
    exposes none for it. Three files under the device node, aer_dev_correctable,
    aer_dev_nonfatal and aer_dev_fatal (Documentation/ABI/testing/sysfs-bus-pci-devices-
    aer), each one "Name count" per line and closed by a TOTAL_ERR_COR /
    TOTAL_ERR_NONFATAL / TOTAL_ERR_FATAL line. The TOTAL line is the total used here, on
    purpose: the ABI says "since multiple errors may be reported using a single ERR_COR
    message, thus TOTAL_ERR_COR at the end of the file may not match the actual total of
    all the errors in the file", so summing the named lines would be a different number
    from the one the kernel counts. MEASURED on this machine: all three files present on
    each of the three NVIDIA cards, every counter 0 (and unable to move, see _aer_live),
    and the root ports above them carry no aer_dev_* file at all.
    The caller reads BOTH ends of the link, because the ABI's header warns that "if an
    endpoint is causing problems, the AER counters may increment at its link partner
    (e.g. root port) because the errors may be "seen" / reported by the link partner and
    not the problematic endpoint itself (which may report all counters as 0 as it never
    saw any problems)". On a root port (or an RCEC) the kernel adds
    aer_rootport_total_err_cor / _fatal / _nonfatal, one number each, "Total number of
    ERR_COR messages reported to rootport" and likewise for the other two: message
    totals for everything below the port, attributed to no single device, read here as
    rootport_cor / rootport_fatal / rootport_nonfatal when the files exist.
    A correctable error is the link retrying a packet: harmless one at a time, and a
    steady trickle is the first sign of a marginal riser or a slot that needs re-seating,
    which is why the per-session delta is kept beside the boot-time total. A non-fatal or
    fatal error is a different kind of fact and colours the row on its own."""
    p = os.path.realpath(dev_path)
    files = {"correctable": "aer_dev_correctable", "nonfatal": "aer_dev_nonfatal",
             "fatal": "aer_dev_fatal"}
    if not any(os.path.isfile(f"{p}/{f}") for f in files.values()):
        return None
    out = {"correctable": 0, "nonfatal": 0, "fatal": 0, "nonzero": []}
    for key, fn in files.items():
        for line in (read(f"{p}/{fn}") or "").splitlines():
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            name, n = parts[0], int(parts[1])
            if name.startswith("TOTAL_ERR_"):
                out[key] = n
            elif n > 0:
                out["nonzero"].append(f"{name} {n}")
    for key, fn in (("rootport_cor", "aer_rootport_total_err_cor"),
                    ("rootport_fatal", "aer_rootport_total_err_fatal"),
                    ("rootport_nonfatal", "aer_rootport_total_err_nonfatal")):
        v = read_int(f"{p}/{fn}", None)
        if v is not None:
            out[key] = v
    base = _AER_BASE.get(p)
    # a total that went down means the counter was reset (driver reload, a device reset):
    # the old baseline would report a negative delta, so it is taken again
    if base is None or base > out["correctable"]:
        _AER_BASE[p] = base = out["correctable"]
    out["correctable_session"] = out["correctable"] - base
    return out


def _pcie_card_node(dev_path):
    """(node, via_switch): the PCI node whose link is the one that goes into the slot,
    and whether it was found by climbing. Two rules, layered:
      1. A device whose parent is the root port negotiates with that root port, so the
         device itself is the card. VERIFIED on this machine: the three NVIDIA cards
         (vendor 0x10de) sit directly under root ports 0000:00:02.0, 00:04.0 and 00:0b.0
         (vendor 0x1002, class 0x060400, parent pci0000:00), and every value the panel
         showed before this rule existed is unchanged by it.
      2. Some cards carry a PCIe switch of their own (AMD Navi, Intel Arc): GPU function
         -> downstream port -> upstream port -> root port. The GPU function then reports
         the card's INTERNAL link, whatever the on-card switch trained it to (2.5 GT/s x1
         on the Arc-like shape in the tests, 16 GT/s x16 on the Navi-like one), which says
         nothing about the slot; what went into the slot is the upstream port's link.
         Heuristic: climb from the device
         while the parent is a PCI node that is NOT the root port and has the same PCI
         `vendor` as the device, on the reasoning that an on-card switch is the GPU
         vendor's silicon while a switch on the board (a PLX, 0x10b5) is not. Never climb
         into or past the root port; a missing vendor file stops the climb. NOT verified
         on real switched hardware, none is installed here, which is why the panel says
         "matched by vendor" whenever it applied.
    The four topologies the tests cover: direct attach (this machine), an on-card switch
    all one vendor (Arc-like, Navi-like: the climb stops at the upstream port), a board
    switch of another vendor (the climb stops at the GPU, the slot is the switch's port),
    and a GPU of the same vendor as its root port (a Radeon on this AMD board: the walk
    must not enter the root port, so the device stays the card).
    A root port is a PCI node whose parent directory is not a PCI address (here
    pci0000:00, the host bridge)."""
    p = os.path.realpath(dev_path)
    vendor = read(f"{p}/vendor")
    node, via = p, False
    while vendor:
        parent = os.path.dirname(node)
        if not _is_pci_node(parent) or _is_root_port(parent):
            break
        if read(f"{parent}/vendor") != vendor:
            break
        node, via = parent, True
    return node, via


def _pcie_negotiation(dev_path):
    """What the card negotiated against what each end of its link can do. pcie_chain()
    answers a different question, the narrowest hop on the way to the root, and reports
    one link; this reports the card's own link three ways: as it is now, as the card is
    capable of (max_link_speed / max_link_width on the card node) and as the slot is
    capable of (the same attributes on the card node's parent, the port it is plugged
    into, when that parent is a PCI node at all). The card node is the device itself on a
    direct attach and the on-card switch's upstream port behind one, see
    _pcie_card_node, which also explains why. Each pair of maxima answers a question a
    single reading cannot:
      width_limited_by_slot: the slot has fewer lanes than the card, a mechanical fact
        (a x16 card in a x4 slot) and never a fault;
      speed_limited_by_host: the port's generation is below the card's, also a fact of
        the board;
      width_degraded: the negotiated width is below what BOTH ends support, which is the
        one condition that is a fault, a dirty contact, a bad riser or a cable, and is
        the only flag rendered as a warning;
      speed_below_both: the negotiated rate is below both maxima. Not a fault on its own:
        a driver may lower the rate while the card idles and raise it under load
        (nvidia-smi --help-query-gpu on pcie.link.gen.gpucurrent: "may be reduced when
        the GPU is not in use"), so this is dim, and the caller re-reads it every tick.
        MEASURED here: the RTX 2060 idles at gen1 x4 and comes up under load; the two
        P100 (P0, persistence mode) hold gen2 x16 while idle. So it depends on the card
        and its power state, and the flag under load is the one worth a look.
    MEASURED on this machine: the RTX 2060 (device max 8.0 GT/s x16) sits in a slot whose
    port reports max 5.0 GT/s x4 and idles at 2.5 GT/s x4: limited by the slot in both
    width and speed, and below both in speed, with nothing degraded. The two P100 (max
    8.0 x16) sit under 5.0 x16 root ports and run 5.0 x16: speed limited by the host, and
    nothing else. Empty when the card node has no link attributes."""
    p, via = _pcie_card_node(dev_path)
    cur = _pcie_link(p)
    if not cur:
        return {}
    card_max = _pcie_link_max(p) or cur
    parent = os.path.dirname(p)
    slot_max = _pcie_link_max(parent) if _is_pci_node(parent) else None
    live, why = _aer_live(p)
    max_gts = [card_max[0]] + ([slot_max[0]] if slot_max else [])
    max_width = [card_max[1]] + ([slot_max[1]] if slot_max else [])
    return {"card_gts": cur[0], "card_width": cur[1],
            "card_max_gts": card_max[0], "card_max_width": card_max[1],
            "slot_max_gts": slot_max[0] if slot_max else None,
            "slot_max_width": slot_max[1] if slot_max else None,
            "gen": _PCIE_GEN.get(cur[0]), "card_max_gen": _PCIE_GEN.get(card_max[0]),
            "slot_max_gen": _PCIE_GEN.get(slot_max[0]) if slot_max else None,
            "width_limited_by_slot": bool(slot_max) and slot_max[1] < card_max[1],
            "speed_limited_by_host": bool(slot_max) and slot_max[0] < card_max[0],
            "width_degraded": cur[1] < min(max_width),
            "speed_below_both": cur[0] < min(max_gts),
            "via_switch": via,
            "aer": pcie_aer(p),
            # the slot port's counters too (see pcie_aer on why the link partner may be
            # the one that counts); None here on this machine, whose root ports expose no
            # aer_dev_* files
            "aer_port": pcie_aer(parent) if _is_pci_node(parent) else None,
            "aer_live": live, "aer_why": why}


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
    link = {"gts": root[0][0], "width": root[0][1], "gbs": band(root[0]),
            "narrow_gbs": band(narrow[0]), "bottleneck": band(narrow[0]) < band(root[0]) * 0.9,
            "narrow_gts": narrow[0][0], "narrow_width": narrow[0][1]}
    # the card's own negotiation rides on the same dict: the seven keys above keep their
    # meaning and their readers (pcie_text), the new ones are read only by _pcie_rows
    link.update(_pcie_negotiation(dev_path))
    return link


def pcie_text(link):
    if not link:
        return ""
    t = f"PCIe {link['gts']:g} GT/s x{link['width']} · {link['gbs']:.1f} GB/s"
    if link["bottleneck"]:
        t += f" (link {link['narrow_gts']:g}x{link['narrow_width']})"
    return t


# ---------------------------------------------------------- pci product naming
_PCI_IDS_PATHS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids",
                  "/usr/share/pci.ids")
_VENDOR = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}


# The device names of pci.ids, {vendor: {device: name}}, read ONCE (round 7c, Fabrizio: «3d
# ok»). Discovery used to run `lspci -mm -s <addr>` once per card, four programs at every
# start here, with this database as the fallback scanned from the top for each card. The
# database is static data: one pass at the first name, kept for the run. MEASURED on the
# file here (/usr/share/misc/pci.ids, 1.4 MB, the hwdata path is a symlink to it): 2378
# vendors, 18685 device lines, 2.9 MiB peak, 0.27 s for the pass; and for all four cards
# lspci -mm printed exactly the pci.ids string (pci-names fixture, 17/09), so the names on
# this machine are unchanged. Elsewhere they can differ: lspci also reads the udev hwdb and
# can print a subsystem name where pci.ids has none or another; the Intel join to nvtop's
# name goes through norm_name() (letters and digits only), which a punctuation difference
# cannot break, a different word could.
_pci_ids_cache = {}


def _pci_ids_db():
    """The first existing _PCI_IDS_PATHS file parsed into {vendor: {device: name}}, cached
    after the first call; {} when no database exists or it cannot be read. Only the vendor
    lines (no tab) and the device lines (one tab) are kept: a subsystem line (two tabs) is
    never a device, and the class list that closes the file (its lines start with `C `,
    which no hex vendor id can) ends the pass."""
    if "db" in _pci_ids_cache:
        return _pci_ids_cache["db"]
    db, cur = {}, None
    path = next((p for p in _PCI_IDS_PATHS if os.path.isfile(p)), None)
    if path:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("C "):
                        break
                    if line.startswith("#") or not line.strip():
                        continue
                    if not line.startswith("\t"):
                        cur = db.setdefault(line[:4].lower(), {})
                    elif not line.startswith("\t\t") and cur is not None:
                        cur[line[1:5].lower()] = line[5:].strip()
        except OSError:
            db = {}                      # unreadable midway: no half database
    _pci_ids_cache["db"] = db
    return db


def _name_from_pci_ids(vendor_id, device_id):
    """The device's name in pci.ids, None when the database or the entry is missing."""
    vid = vendor_id.replace("0x", "").lower()
    did = device_id.replace("0x", "").lower()
    return _pci_ids_db().get(vid, {}).get(did)


def gpu_product_name(dev_path, vendor_id):
    did = read(f"{dev_path}/device")
    if did:
        name = _name_from_pci_ids(vendor_id, did)
        if name:
            return name
    return f"{_VENDOR.get(vendor_id, 'Unknown')} GPU"


# ----------------------------------------------------------- Vulkan device order
# The N of the `VulkanN` a llama.cpp process names on its -dev/-ts command line is the
# position of the card in ggml-vulkan's device list, and that list is the end of a chain of
# three programs, none of which this tool runs. The previous reader ran vulkaninfo at every
# start. MEASURED (17/09 12:58, strace under a llama-bench Vulkan load on the W9170): the child
# opened /dev/dri/renderD128 O_RDWR and issued 60 DRM_IOCTL_AMDGPU_INFO, 4 VERSION, 2 GET_CAP
# and syncobj create/wait/destroy in 18 ms, a DRM client with driver state on a busy card;
# the same call wakes the idle RTX 2060 (CHANGELOG, Measurements, 3/3 runs). Fabrizio: «ok
# vulkaninfo da sysfs». So the map comes from FILES ONLY: the cards' sysfs identity, the
# loader's manifest files and the environment of the process that matters, with the rules
# of the three programs reproduced from their sources (every line cited was opened):
#  1. WHICH CARDS AN INSTALLED DRIVER EXPOSES. A Vulkan ICD claims the cards bound to its
#     kernel driver: RADV opens PCI render nodes of vendor 0x1002 and accepts only the amdgpu
#     kernel driver (mesa-25.2.8 radv_physical_device.c:2488-2494, 2164-2185); NVK does the
#     same for nouveau; ANV/hasvk for i915/xe, on disjoint generations, so one of the two
#     exposes a card; the proprietary NVIDIA ICD the cards bound to `nvidia` (no source:
#     MEASURED on this machine, three cards, three devices). A card whose driver has no
#     installed ICD (the `radeon` driver, say) is no Vulkan device at all. lvp is llvmpipe,
#     a CPU device with no PCI address. A card claimed by TWO installed ICDs (RADV and AMDVLK)
#     appears twice, and ggml keeps one by deviceUUID with a priority table that may re-append
#     the survivor at the end (ggml-vulkan.cpp:7845-7867, 7914-7917): not reproducible here.
#  2. THE LOADER (libvulkan 1.3.275, KhronosGroup/Vulkan-Loader tag v1.3.275, loader/
#     loader_linux.c; the installed Ubuntu binary carries the strings of this code, so the
#     sort is compiled in). Every ICD's devices are pooled and sorted: type first, discrete 10
#     > integrated 5 > virtual 3 > other 2 > CPU 1 (:38-56), then devices with
#     VK_EXT_pci_bus_info before those without, then domain, bus, device, function ascending
#     (:75-123). VK_LOADER_DEVICE_SELECT=vid:did marks the FIRST match in ICD order as default
#     and it sorts first (:207-232, :77-81): with one match the result is known, with two
#     (this machine's P100 pair) it is not. VK_LOADER_DISABLE_SELECT non-zero leaves the ICD
#     order (loader.c:6040-6050), which no file describes. Manifests: VK_DRIVER_FILES (or the
#     deprecated VK_ICD_FILENAMES) replaces the search and VK_ADD_DRIVER_FILES adds to it
#     (loader.c:3025-3031); otherwise $XDG_CONFIG_HOME|~/.config, $XDG_CONFIG_DIRS|/etc/xdg,
#     /etc, $XDG_DATA_HOME|~/.local/share, $XDG_DATA_DIRS|/usr/local/share:/usr/share, each
#     with vulkan/icd.d (loader.c:2958-3018, 3170-3183; vk_loader_platform.h:135-146), then
#     VK_LOADER_DRIVERS_SELECT/DISABLE on the file name (loader.c:3610-3631) with the filter
#     grammar of loader_environment.c:182-244 and 393-440. Implicit layers come from the same
#     directories under implicit_layer.d, and one is enabled unless its manifest's
#     disable_environment variable is present, or VK_LOADER_LAYERS_DISABLE names it and
#     VK_LOADER_LAYERS_ALLOW does not, or its enable_environment variable does not hold
#     EXACTLY the manifest's value (loader.c:937-994).
#  3. DEVICE-SELECT (mesa-25.2.8 src/vulkan/device-select-layer/device_select_layer.c). The
#     layer picks a default device and moves it first; the rest keep the loader's order
#     (:604-611). The default: MESA_VK_DEVICE_SELECT=vid:did, the LAST match (:272-288,
#     :471-475); else DRI_PRIME when it is not a plain number, as vid:did and then as the
#     tag pci-dddd_bb_dd_f (:451-461, :477-499, :290-317); else the X11/Wayland screen device,
#     BUT only when the application enabled VK_KHR_xcb_surface or VK_KHR_wayland_surface at
#     instance creation (:149-161), which ggml-vulkan never does (ggml-vulkan.cpp:7743-7766):
#     for llama this step does not exist and DISPLAY changes nothing; else the first card in
#     loader order whose /sys/bus/pci/devices/<addr>/boot_vga reads 1 (:376-398, :510-517;
#     the kernel creates that file on VGA-class devices alone, pci-sysfs.c:1754-1761); else
#     the first non-CPU device (:400-413, :518-534). DRI_PRIME=n>0 then replaces a default at
#     index 0 or 1 by the n-th OTHER non-CPU device (:415-432, :535-547). A trailing `!` on
#     the selection, or MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1, exposes the default
#     alone (:474, :620-625). Any other implicit layer that is enabled (the NVIDIA Optimus
#     layer by __NV_PRIME_RENDER_OFFLOAD=1, the present layer by NVPRESENT_ENABLE_SMOOTH_
#     MOTION=1) sits in the same chain and is not modelled, and so is an explicit layer forced
#     on by VK_INSTANCE_LAYERS: no map, and the reason names it.
#  4. GGML (llama.cpp b10977 ggml/src/ggml-vulkan/ggml-vulkan.cpp). GGML_VK_VISIBLE_DEVICES
#     names indices into the list above, in that order, of any type, and an index past the
#     end aborts the process (:7806-7821). Without it, every discrete or integrated device
#     that passes ggml_vk_device_is_supported, 16-bit storage buffers (:7838, :20089-20101),
#     a driver answer no file holds, so the map ASSUMES it and says so; with no GPU at all
#     the first non-CPU device (:7930-7937). `VulkanN` is the position in that list (:7299,
#     :7969, :19992).
# DECLARED NOT MODELLED, beyond the reasons the map returns: CDNA and unknown-generation AMD
# parts, which RADV refuses (radv_is_gpu_supported, :2122-2132); an APU's integrated GPU,
# which RADV reports integrated from has_dedicated_vram (:1462-1465) and this table takes as
# discrete, sysfs having no such flag; Intel and nouveau generations their ICD refuses;
# virtio, asahi and gfxstream devices. On a card of one of those kinds the index is wrong,
# not absent. And an ICD without VK_EXT_pci_bus_info: RADV (:730), hasvk (:316), NVK on a
# discrete part (:257) and the NVIDIA ICD (measured) all export it, so the layer's bus-info
# paths are the ones reproduced.
# Every input is STATIC for the life of the device or of the process: the cards and this
# tool's own manifests are read once, a llama process's environ (and the manifests its
# environment selects) once per pid, and a pid's result is forgotten with the pid.
_DRM_ROOT = "/sys/class/drm"     # where the cards are listed; the tests point it elsewhere
_VK_ROOT = ""                    # prefix of the loader's manifest directories; the tests root it
_VK_ICD_OF_DRIVER = {            # kernel driver -> the ICD libraries that claim its cards
    "amdgpu": ("libvulkan_radeon.so", "amdvlk64.so"),
    "nvidia": ("libGLX_nvidia.so.0",), "nvidia-drm": ("libGLX_nvidia.so.0",),
    "nouveau": ("libvulkan_nouveau.so",),
    "i915": ("libvulkan_intel.so", "libvulkan_intel_hasvk.so"),
    "xe": ("libvulkan_intel.so", "libvulkan_intel_hasvk.so"),
}
_VK_SPLIT_BY_GENERATION = ("i915", "xe")   # two ICDs installed, one exposes a given card
_VK_CPU_ICD = "libvulkan_lvp.so"
_VK_DEVSEL_LAYER = "VK_LAYER_MESA_device_select"
_vulkan_map = None               # the result for this tool's own environment, once
_vulkan_pid_maps = {}            # pid -> the result for that process's environment, once each


def _vulkan_cards():
    """The identity of every card under _DRM_ROOT, from sysfs alone: PCI address, vendor and
    device ids, whether boot_vga reads 1 (the file exists on VGA-class devices only) and the
    kernel driver bound to it, None without one."""
    out = []
    for card in sorted(glob.glob(f"{_DRM_ROOT}/card[0-9]*")):
        if "-" in os.path.basename(card):        # connector, not a card
            continue
        dev = f"{card}/device"
        drv = f"{dev}/driver"

        def hexint(name):
            try:
                return int(read(f"{dev}/{name}", ""), 16)
            except ValueError:
                return None
        out.append({"pci": os.path.basename(os.path.realpath(dev)),
                    "vendor": hexint("vendor"), "device": hexint("device"),
                    "boot_vga": read(f"{dev}/boot_vga") == "1",
                    "driver": os.path.basename(os.path.realpath(drv)) if os.path.exists(drv) else None})
    return out


def _vk_search_dirs(env, leaf):
    """The loader's manifest directories for `leaf` (icd.d or implicit_layer.d) under `env`,
    in its order, each under _VK_ROOT, duplicates dropped as the loader drops them."""
    home = env.get("HOME")
    dirs = [env.get("XDG_CONFIG_HOME") or (f"{home}/.config" if home else "")]
    dirs += (env.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":")
    dirs.append("/etc")
    dirs.append(env.get("XDG_DATA_HOME") or (f"{home}/.local/share" if home else ""))
    dirs += (env.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")
    out = []
    for d in dirs:
        p = f"{_VK_ROOT}{d.rstrip('/')}/vulkan/{leaf}"
        if d and p not in out:
            out.append(p)
    return out


def _vk_manifest_files(env, leaf, override=None, extra=None):
    """The manifest files the loader would open: those under `override` (files or
    directories, colon separated) instead of the search path when it is set, plus those
    under `extra` in either case."""
    files = []

    def add(paths):
        for p in paths.split(":"):
            if os.path.isdir(p):
                files.extend(sorted(glob.glob(f"{p}/*.json")))
            elif p:
                files.append(p)
    if override:
        add(override)
    else:
        for d in _vk_search_dirs(env, leaf):
            files.extend(sorted(glob.glob(f"{d}/*.json")))
    if extra:
        add(extra)
    return files


def _vk_json(path):
    """A manifest as the loader reads it: None when it cannot be parsed, which the loader
    logs and skips (loader.c:3600-3607)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _vk_filter(value):
    """A loader filter variable as (kind, text) terms: `~all~`, `*` and `**` match anything,
    `~implicit~` every implicit layer, `*x*` a substring, `*x` a suffix, `x*` a prefix and
    a bare name the whole name, all case-insensitively (loader_environment.c:182-244)."""
    terms = []
    for tok in (value or "").lower().split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in ("~all~", "*", "**", "~implicit~", "~explicit~"):
            terms.append(("special", tok))
        elif tok.startswith("*") and tok.endswith("*"):
            terms.append(("substring", tok[1:-1]))
        elif tok.startswith("*"):
            terms.append(("suffix", tok[1:]))
        elif tok.endswith("*"):
            terms.append(("prefix", tok[:-1]))
        else:
            terms.append(("fullname", tok))
    return terms


def _vk_filter_match(name, terms, implicit=False):
    """Whether `name` matches any term (loader_environment.c:393-440)."""
    n = name.lower()
    for kind, text in terms:
        if kind == "special":
            if text in ("~all~", "*", "**") or (implicit and text == "~implicit~"):
                return True
        elif kind == "substring" and text in n:
            return True
        elif kind == "suffix" and n.endswith(text):
            return True
        elif kind == "prefix" and n.startswith(text):
            return True
        elif kind == "fullname" and n == text:
            return True
    return False


def _vulkan_icds(env):
    """The library file name of every ICD the loader would load under `env`."""
    files = _vk_manifest_files(env, "icd.d", env.get("VK_DRIVER_FILES") or env.get("VK_ICD_FILENAMES"),
                               env.get("VK_ADD_DRIVER_FILES"))
    select = _vk_filter(env.get("VK_LOADER_DRIVERS_SELECT"))
    disable = _vk_filter(env.get("VK_LOADER_DRIVERS_DISABLE"))
    out = []
    for path in files:
        name = os.path.basename(path)
        selected = bool(select) and _vk_filter_match(name, select)
        if (disable and _vk_filter_match(name, disable) and not selected) or (select and not selected):
            continue
        m = _vk_json(path)
        lib = m.get("ICD", {}).get("library_path") if isinstance(m, dict) else None
        if isinstance(lib, str):
            out.append(os.path.basename(lib))
    return out


def _vulkan_layers(env):
    """Every implicit layer the loader would find under `env`: its name, the enable_environment
    pair (None when absent) and the disable_environment pair, which the loader requires
    (loader.c:2424-2436). A manifest carries one `layer` or a `layers` list."""
    out = []
    for path in _vk_manifest_files(env, "implicit_layer.d"):
        m = _vk_json(path)
        if not isinstance(m, dict):
            continue
        items = m.get("layers") if isinstance(m.get("layers"), list) else [m.get("layer")]
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("name"), str):
                continue

            def pair(key):
                d = it.get(key)
                if isinstance(d, dict) and d:
                    k, v = next(iter(d.items()))
                    return str(k), str(v)
                return None
            disable = pair("disable_environment")
            if disable is None:
                continue
            out.append({"name": it["name"], "enable": pair("enable_environment"), "disable": disable})
    return out


def _vk_layer_enabled(layer, env):
    """The loader's rule for an implicit layer (loader.c:937-994)."""
    name = layer["name"]
    forced_off = (_vk_filter_match(name, _vk_filter(env.get("VK_LOADER_LAYERS_DISABLE")), implicit=True)
                  and not _vk_filter_match(name, _vk_filter(env.get("VK_LOADER_LAYERS_ALLOW"))))
    forced_on = _vk_filter_match(name, _vk_filter(env.get("VK_LOADER_LAYERS_ENABLE")))
    en = layer["enable"]
    enable = en is None or env.get(en[0]) == en[1]
    if forced_on:
        enable = True
    elif enable and forced_off:
        return False
    if layer["disable"][0] in env:
        return False
    return enable


def _atoi(s):
    """C atoi: the leading integer of the string, 0 without one."""
    m = re.match(r"\s*([+-]?\d+)", s or "")
    return int(m.group(1)) if m else 0


def _vid_did(s):
    """(vendor, device) from a `vid:did` string as sscanf("%x:%x") reads it, else None."""
    m = re.match(r"\s*(?:0[xX])?([0-9a-fA-F]+)\s*:\s*(?:0[xX])?([0-9a-fA-F]+)", s or "")
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else None


def _bang(s):
    return len(s) > 1 and s.endswith("!")


def _vk_device_select(devices, env, notes):
    """The device-select layer's reordering of `devices` under `env`, or (None, why)."""
    sel = env.get("MESA_VK_DEVICE_SELECT")
    dri = env.get("DRI_PRIME")
    if sel == "list":
        return None, "MESA_VK_DEVICE_SELECT=list: the process prints the devices and exits"

    def explicit(s):                              # the LAST vid:did match
        vd = _vid_did(s)
        idx = None
        for i, d in enumerate(devices):
            if vd and (d["vendor"], d["device"]) == vd:
                idx = i
        return idx

    def tag(s):                                   # pci-dddd_bb_dd_f, the LAST match
        n = len("pci-xxxx_yy_zz_w")
        if len(s) < n:
            return None
        s = s[:-1] if (len(s) == n + 1 and s.endswith("!")) else s
        idx = None
        for i, d in enumerate(devices):
            if d["pci"] and s == "pci-" + d["pci"].replace(":", "_").replace(".", "_"):
                idx = i
        return idx
    dri_int = -1                                  # :453-461: atoi without a colon, then clamped
    if dri is not None:                           # to 0, so a vid:did or a tag reads as 0 too
        dri_int = max(_atoi(dri) if ":" not in dri else -1, 0)
    default, only = None, False
    if sel:
        default = explicit(sel)
        only = default is not None and _bang(sel)
    if default is None and dri is not None and dri_int == 0:
        default = explicit(dri)
        if default is None:
            default = tag(dri)
        only = default is not None and _bang(dri)
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY") or env.get("WAYLAND_SOCKET"):
        notes.append("X11/Wayland default not consulted: ggml enables no surface extension")
    if default is None:
        default = next((i for i, d in enumerate(devices) if d["boot_vga"]), None)
    if default is None:
        default = next((i for i, d in enumerate(devices) if d["pci"]), 0 if devices else None)
    cpus = sum(1 for d in devices if not d["pci"])
    if dri_int > 0 and len(devices) > cpus + 1 and default in (0, 1):
        skip, found = dri_int, None
        for i, d in enumerate(devices):
            if i == default or not d["pci"]:
                continue
            skip -= 1
            if skip <= 0:
                found = i
                break
        if found is not None:
            default, only = found, _bang(dri)
        else:
            default = 0                           # the layer's -1, returned as index 0 (:549)
    if env.get("MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE") == "1":
        only = True
    if default is None:
        return [], None
    first = devices[default]
    return [first] if only else [first] + [d for i, d in enumerate(devices) if i != default], None


def _vulkan_order(cards, icds, layers, env):
    """{pci address: VulkanN} for a ggml-vulkan process under `env` on a machine holding
    `cards`, with the ICD libraries `icds` and the implicit layers `layers` installed, as
    {"map", "why", "notes"}: `map` is None with a `why` where the chain cannot be reproduced
    from files, and `notes` name what the map assumes."""
    notes = []

    def fail(why):
        return {"map": None, "why": why, "notes": notes}
    if env.get("VK_INSTANCE_LAYERS"):
        return fail(f"VK_INSTANCE_LAYERS={env['VK_INSTANCE_LAYERS']}: an explicit layer in the chain")
    if _atoi(env.get("VK_LOADER_DISABLE_SELECT")) != 0:
        return fail("VK_LOADER_DISABLE_SELECT set: the drivers' own order, which no file gives")
    if not icds:
        return {"map": {}, "why": "no Vulkan ICD manifest found", "notes": notes}
    devices = []
    for c in cards:
        claimed = [lib for lib in _VK_ICD_OF_DRIVER.get(c["driver"], ()) if lib in icds]
        if not claimed:
            continue
        if len(claimed) > 1 and c["driver"] not in _VK_SPLIT_BY_GENERATION:
            return fail(f"{c['pci']} claimed by {' and '.join(claimed)}: ggml keeps one by deviceUUID")
        dom, bus, devfn = c["pci"].split(":")
        # the loader's type priority: an i915/xe card on bus 0 is the integrated GPU
        # (sysfs has no type flag: this is where Intel puts it), every other card discrete
        integrated = c["driver"] in _VK_SPLIT_BY_GENERATION and (int(dom, 16), int(bus, 16)) == (0, 0)
        devices.append({"pci": c["pci"], "vendor": c["vendor"], "device": c["device"],
                        "boot_vga": c["boot_vga"], "prio": 5 if integrated else 10,
                        "key": (int(dom, 16), int(bus, 16), int(devfn.split(".")[0], 16), int(devfn.split(".")[1], 16))})
    devices.sort(key=lambda d: (-d["prio"], d["key"]))
    if _VK_CPU_ICD in icds:
        devices.append({"pci": None, "vendor": 0x10005, "device": 0, "boot_vga": False, "prio": 1})
    ldr = env.get("VK_LOADER_DEVICE_SELECT")
    if ldr:
        hits = [i for i, d in enumerate(devices) if (d["vendor"], d["device"]) == _vid_did(ldr)]
        if len(hits) > 1:
            return fail(f"VK_LOADER_DEVICE_SELECT={ldr} matches {len(hits)} devices: the first in driver order, unknown")
        if hits:
            devices.insert(0, devices.pop(hits[0]))
    for layer in layers:
        if layer["name"] != _VK_DEVSEL_LAYER and _vk_layer_enabled(layer, env):
            return fail(f"{layer['name']} enabled: an implicit layer in the chain")
    devsel = next((l for l in layers if l["name"] == _VK_DEVSEL_LAYER), None)
    if devsel and _vk_layer_enabled(devsel, env):
        devices, why = _vk_device_select(devices, env, notes)
        if why:
            return fail(why)
    vis = env.get("GGML_VK_VISIBLE_DEVICES")
    if vis is not None:
        chosen = []
        for tok in vis.replace(",", " ").split():
            m = re.match(r"\d+", tok)
            if not m:                             # `ss >> size_t` stops at the first non-number
                break
            i = int(m.group(0))
            if i >= len(devices):
                return fail(f"GGML_VK_VISIBLE_DEVICES={vis}: index {i} of {len(devices)} devices, ggml aborts")
            chosen.append(devices[i])
    else:
        chosen = [d for d in devices if d["pci"]]
        notes.append("every card assumed to pass ggml_vk_device_is_supported (16-bit storage): not in any file")
    return {"map": {d["pci"]: i for i, d in enumerate(chosen) if d["pci"]}, "why": None, "notes": notes}


def _vulkan_resolve(env, cards=None):
    """The map for `env` on this machine: the cards (read now unless given), the manifests
    that environment selects, then the rule."""
    cards = _vulkan_cards() if cards is None else cards
    r = _vulkan_order(cards, _vulkan_icds(env), _vulkan_layers(env), env)
    r["cards"] = cards
    return r


def vulkan_device_map(env=None):
    """{pci address: N} of the VulkanN a llama.cpp process started from `env` (this tool's
    own environment when None) would name on its -dev/-ts line; {} when the map is unknown
    or empty, with the reason in _vulkan_map["why"], which --probe prints. Computed once and
    kept: every input is static for the life of the cards."""
    global _vulkan_map
    if _vulkan_map is None:
        _vulkan_map = _vulkan_resolve(dict(os.environ) if env is None else env)
    return _vulkan_map["map"] or {}


def _proc_environ(pid):
    """(environment, None) of process `pid` from /proc/<pid>/environ, or (None, why): the
    kernel serves it with PTRACE_MODE_READ on the process, as page-sized copies of the
    process's own memory, no device involved (fs/proc/base.c:1001-1066)."""
    try:
        with open(f"{_PROC_ROOT}/{pid}/environ", "rb") as f:
            raw = f.read()
    except PermissionError:
        return None, "environ permission denied (another user's process: run with sudo)"
    except OSError as e:
        return None, f"environ unreadable ({e.strerror or type(e).__name__})"
    env = {}
    for rec in raw.split(b"\0"):
        k, sep, v = rec.partition(b"=")
        if sep:
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return env, None


def vulkan_map_for_pid(pid):
    """The map for the environment of llama process `pid`, read once and kept until
    _vulkan_forget_pids drops it: the server's -dev names are interpreted in ITS environment
    (GGML_VK_VISIBLE_DEVICES, DRI_PRIME, MESA_VK_DEVICE_SELECT, the loader's variables), not
    in this tool's. Unknown, with the reason, when the environ cannot be read."""
    r = _vulkan_pid_maps.get(pid)
    if r is None:
        if not str(pid).isdigit():
            r = {"map": None, "why": "no local process for this port", "notes": []}
        else:
            vulkan_device_map()                       # the cards, read once for every map
            env, why = _proc_environ(pid)
            r = (_vulkan_resolve(env, _vulkan_map["cards"]) if env is not None
                 else {"map": None, "why": why, "notes": []})
        _vulkan_pid_maps[pid] = r
    return r


def _vulkan_forget_pids(live):
    """Drop the maps of every pid not in `live`, so a returning pid is read afresh."""
    for pid in [p for p in _vulkan_pid_maps if p not in live]:
        del _vulkan_pid_maps[pid]


def _vulkan_text(r):
    """One map as --probe prints it: the VulkanN of every card, or the reason there is none."""
    if r.get("map"):
        return " ".join(f"Vulkan{i}→{pci}" for pci, i in sorted(r["map"].items(), key=lambda kv: kv[1]))
    return r.get("why") or "no device"


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
    # v1.x (Navi2x/RDNA2) layout, the fields we use sit at stable offsets.
    if fmt_rev == 1 and len(d) >= 120:
        u16 = lambda o: struct.unpack_from("<H", d, o)[0]
        u32 = lambda o: struct.unpack_from("<I", d, o)[0]
        u64 = lambda o: struct.unpack_from("<Q", d, o)[0]
        good = lambda x: None if x in (0xFFFF, 0) else x
        # Clocks, validated against a Radeon RX 6750 XT (Navi 22, content_rev 3): read at
        # the same moment as pp_dpm_sclk and pp_dpm_mclk across three different loads, the
        # pairs matched exactly (2490/1124, 2245/675, 500/96). These are the current
        # clocks; the averages live at other offsets and rightly do not agree.
        out["mem_clock"] = good(u16(58))
        out["gfx_clock"] = good(u16(54))
        # Offsets 10 and 12 are temperatures, not voltages. In gpu_metrics_v1_3 they are
        # temperature_vrgfx and temperature_vrsoc, in degrees Celsius, and printing them
        # as millivolts is the kind of mislabel that survives because the numbers look
        # plausible. On a Navi 22 card they held steady at 47 and 45 while hwmon's
        # in0_input swung between 806 and 1043 mV with the clocks: a voltage that does not
        # follow the clocks is not a voltage, and 47 beside an edge temperature of 50 is
        # not a coincidence. The real core voltage comes from hwmon in0_input, which this
        # file already reads. The blob's own voltage fields at 102, 104 and 106 read
        # 0xffff on that card, meaning unsupported.
        out["vrm_temp_gfx"] = good(u16(10))
        out["vrm_temp_soc"] = good(u16(12))
        media = u16(20)                           # media engine activity, 0 is valid
        out["media_activity"] = None if media == 0xFFFF else media
        # Throttle. Zero is a valid reading here: it means nothing is throttling, which is
        # the normal state of a healthy card. Treating it as "field not present" sent
        # every idle card down the fallback path, where a 32-bit field at a different
        # offset was read as a throttle bitmask and produced an invented reason to worry.
        # Offset 112 is structurally sound on v1_3: indep_throttle_status is the last u64
        # and ends exactly at 120, which is the size the blob declares for itself. Only
        # the all-ones sentinel means "not supported" and earns the older field.
        bits = u64(112)
        if bits == 0xFFFFFFFFFFFFFFFF:
            bits = u32(68)                         # older 32-bit field, pre-v1_3 layouts
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


# ------------------------------------------- legacy powerplay (smu7) and the SMC message
# On the legacy powerplay backends some sysfs reads do not read a register: they post a
# message to the SMC firmware and busy-wait for its answer. The path, in the local 6.18.26
# tree (drivers/gpu/drm/amd/, every line opened and read, not guessed):
#   smumgr.c:130-153 smum_send_msg_to_smc takes hwmgr->msg_lock and calls the backend;
#   ci_smumgr.c:208-225 ci_send_msg_to_smc writes SMC_MESSAGE_0 and PHM_WAIT_FIELD_UNEQUAL
#   (216) on SMC_RESP_0; smu_helper.h:236-239 -> smu_helper.c:155-177
#   phm_wait_for_register_unequal: `for (i = 0; i < hwmgr->usec_timeout; i++) { read;
#   udelay(1); }` with usec_timeout = AMD_MAX_USEC_TIMEOUT = 1000000 (hwmgr.c:85,
#   include/amd_shared.h:30): up to ONE SECOND of udelay busy-wait per message, with no way
#   to abort, under adev->pm.mutex (taken by every amdgpu_dpm_* wrapper, amdgpu_dpm.c).
# MEASURED on this machine: a loop reading pp_dpm_mclk during a Vulkan workload froze the
# system (smu7_print_clock_levels -> ci_send_msg_to_smc -> phm_wait_for_register_unequal
# spinning while the GPU was in recovery: RCU stall, forced reboot). The classification
# of every file this reader touches, sysfs show function down to the hwmgr:
#
# SENDS AN SMC MESSAGE (never read in the periodic path on a legacy card, see below):
#   pp_dpm_sclk   amdgpu_pm.c:1076 -> 988-1011 amdgpu_get_pp_dpm_clock -> amdgpu_dpm.c:
#                 1244-1261 -> amd_powerplay.c:727-740 -> smu7_hwmgr.c:4978-4993, message
#                 PPSMC_MSG_API_GetSclkFrequency at 4979 (the `*` marker needs it)
#   pp_dpm_mclk   same path, smu7_hwmgr.c:4994-5009, PPSMC_MSG_API_GetMclkFrequency at 4995
#   freq1_input   amdgpu_pm.c:3409-3424 -> 1424-1440 -> amdgpu_dpm.c:517-536 read_sensor
#                 -> amd_powerplay.c:799-828 -> smu7_hwmgr.c:4011-4017, message at 4012
#   freq2_input   amdgpu_pm.c:3433-3448 -> ... -> smu7_hwmgr.c:4018-4024, message at 4019
#   power1_average amdgpu_pm.c:3278-3289 -> 3260-3276 -> sensor GPU_AVG_POWER ->
#                 smu7_hwmgr.c:4059-4066 -> smu7_get_gpu_power 3952-3995. On Hawaii,
#                 Bonaire, Fiji and Tonga PPSMC_MSG_GetCurrPkgPwr is NOT used (3961-3971,
#                 the comment says it is unsupported there); the fallback 3979-3991 sends
#                 PPSMC_MSG_PmStatusLogStart, then up to ten times msleep(500) and
#                 PPSMC_MSG_PmStatusLogSample until SMU_PM_STATUS_95 is non-zero: two to
#                 eleven messages and 0.5 to 5 s of sleep, holding pm.mutex, per read.
#                 The old per-tick read of this file therefore also serialised every
#                 other pm read of the tick behind a half-second kernel sleep.
# REGISTER READ ONLY (cgs_read_ind_register / PHM_READ_*, an MMIO read, no wait):
#   gpu_busy_percent, mem_busy_percent  amdgpu_pm.c:1450-1488 -> smu7_hwmgr.c:4025-4038
#                 (the SMC's own AverageGraphicsActivity / AverageMemoryActivity soft
#                 registers, 4027-4033); this is the idle signal used below
#   temp1_input   amdgpu_pm.c:2756-2792 (EDGE_TEMP is GPU_TEMP, kgd_pp_interface.h:146-147)
#                 -> smu7_hwmgr.c:4039-4042 -> smu7_thermal.c:293-309 (CG_MULT_THERMAL_STATUS)
#   in0_input     amdgpu_pm.c:3185-3200 -> smu7_hwmgr.c:4067-4077 (PWR_SVI2_STATUS)
#   pwm1          amdgpu_pm.c:2991-3011 -> amdgpu_dpm.c:1512-1527 -> amd_powerplay.c:549-563
#                 -> smu7_thermal.c:54-78 (CG_FDO_CTRL1, CG_THERMAL_STATUS)
#   fan1_input, fan1_target  amdgpu_pm.c:3013-3033, 3069-3089 -> amdgpu_dpm.c:1529-1544 ->
#                 amd_powerplay.c:565-579 -> smu7_thermal.c:80-100 (CG_TACH_STATUS; -ENODEV
#                 without tachometer pulses, -EINVAL when TACH_PERIOD is 0, which is why
#                 the read fails on a passive board and the file has no value)
#   pp_dpm_pcie   amdgpu_pm.c:1211-1216 -> ... -> smu7_hwmgr.c:5010-5025 -> 212-221
#                 (ixPCIE_LC_SPEED_CNTL). The "x8"/"x16" in its lines are string literals
#                 chosen by the speed code at 5021-5023, not a lane count.
# STORED FIELDS ONLY (kernel memory; pm.mutex is still taken, and every show function
# first passes amdgpu_pm_get_access_if_active 148-166, which refuses a suspended device):
#   power_dpm_state 214-234 -> amdgpu_dpm.c:1013-1032 -> amd_powerplay.c:466-497
#   power_dpm_force_performance_level 326-354 -> amdgpu_dpm.c:1050-1066 -> amd_powerplay.c:388-397
#   pp_num_states 409-437 -> amdgpu_dpm.c:1150-1165 -> amd_powerplay.c:597-631
#   pp_cur_state 439-471 (the two above); pp_force_state 473-484 (a flag, else "\n")
#   pp_table 551-577 -> amd_powerplay.c:633-642, a memcpy of the binary soft table: not read
#   pp_sclk_od, pp_mclk_od 1226-1290 -> amdgpu_dpm.c:1336-1386 -> smu7_hwmgr.c:5089-5145
#   pp_power_profile_mode 1342-1362 -> amdgpu_dpm.c:1410-1425 -> smu7_hwmgr.c:5558-5616
#   pp_od_clk_voltage: hidden unless od_enabled (1919-1944, amdgpu_dpm.c:1667-1687);
#                 when present its smu7 path 839-881 -> smu7_hwmgr.c:5026-5060 prints
#                 tables from memory. Absent on this card, not read.
#   thermal_throttling_logging 1591-1602 (an atomic and a ratelimit, no pm access at all)
#   hwmon temp1_crit, temp1_crit_hyst 2794-2808; temp1_label 2842-2852; fan1_min, fan1_max
#                 3035-3067 -> amd_powerplay.c:820-825; fan1_enable, pwm1_enable 3128-3148,
#                 2880-2900 -> amdgpu_dpm.c:1478-1493 -> smu7_hwmgr.c:5084-5087; pwm1_min,
#                 pwm1_max 2939-2951 (0 and 255); power1_cap and _default/_min/_max
#                 3304-3361 -> amdgpu_dpm.c:1580-1599 -> amd_powerplay.c:988-1024 (min is
#                 the literal 0 at 1015-1016); power1_label 3363-3376; freq1_label,
#                 freq2_label, in0_label 3426-3431, 3450-3455, 3219-3224
#   board_info amdgpu/amdgpu_device.c:363-390; vbios_version, vbios_build
#                 amdgpu_atombios.c:1825-1849; fw_version/* amdgpu_ucode.c:737-770
#   mem_info_vram_*, mem_info_vis_vram_* amdgpu_vram_mgr.c:101-163, 206-213;
#                 mem_info_gtt_* amdgpu_gtt_mgr.c:43-77; mem_info_preempt_used
#                 amdgpu_preempt_mgr.c:36-47 (TTM counters)
#   current_link_speed/width, max_link_speed/width drivers/pci/pci-sysfs.c:198-260 (PCIe
#                 capability config reads); power_state 155-162; power/control and
#                 power/runtime_status drivers/base/power/sysfs.c:100-105, 149-165
# ABSENT ON A LEGACY CARD, and this absence is how one is recognised: gpu_metrics is not
# created below GC 9.1.0 (amdgpu_pm.c:2620-2622 marks it unsupported, 2687-2688 then skips
# device_create_file). Every card without it runs a powerplay hwmgr or the legacy dpm code,
# never swsmu; smu7 (Bonaire, Hawaii, Tonga, Fiji, Polaris, VegaM) is the family measured
# here. Corroborated on the same card by pp_dpm_socclk (hidden below GC 9.0.0, 1993-1995)
# and pp_features (2615-2619) being absent too. A card WITH gpu_metrics keeps the per-tick
# reads it always had. Also absent here: power1_input (GPU_INPUT_POWER is -EOPNOTSUPP on
# Hawaii, smu7_hwmgr.c:4051-4058, hidden by 3744-3747), temp1_emergency, temp2/temp3, in1,
# in2, power2_* (hwmon_attributes_visible 3795-3836), unique_id, pcie_bw, vcn_busy_percent.
#
# THE DESIGN THAT FOLLOWS: on a legacy card the five message files are read only in the SMC
# tier, and the SMC tier runs only when three independent conditions hold at the same tick:
#   (i)  the register window: this tick's own register-only readings say the card is idle
#        (both busy percentages at or under _AMD_IDLE_BUSY) and, once any busy tick has been
#        seen, at least _AMD_IDLE_WINDOW seconds and _AMD_IDLE_SAMPLES consecutive idle ticks
#        have passed since the last busy one. MEASURED (17/09 12:07:52, strace under a
#        llama-bench load): the previous design honoured a single sample, read 0 % in the
#        one-to-two-second gap between two repetitions, opened pp_dpm_sclk 0.6 ms later and
#        the load was back at 100 % 0.2 s after that, inside the read. A window of seconds,
#        not one sample, is what the alternation cannot pass. Before the first busy tick the
#        single-sample rule still applies, so a card that is idle when the tool starts reads
#        at once; the residual is a tool started inside a gap whose client the scan below
#        cannot see.
#   (ii) at least _AMD_SLOW_INTERVAL has passed since the previous SMC read.
#   (iii) the card's DRM clients, scanned at that moment and never under load, include no
#        COMPUTE CLIENT (Fabrizio: «con i client DRM, ma deve essere in sola lettura; non si
#        può spammare di polling un driver»). See _drm_clients_of and _amd_compute_clients
#        for the definition and its evidence.
# The activity register is written by the SMC itself, so an SMC wedged under load holds it
# high and the gate stays shut in exactly the state that froze the machine; an SMC wedged
# at idle is the residual risk, and it is the one state in which nothing is running that a
# freeze would cost. When the gate is shut the three live fields are None with an "idle
# only" reason that names what shut it: the idle number is never replayed as the number
# under load, nor while a client that can start at any moment holds the card. Rejected: a
# read at start (the tool may start mid-benchmark, and power under load is what the idle
# figure misstates most); a refresh key (the one input that could send a message under
# load is what this removes, and --line has no keys); leaving the fields out (the card's
# only power meter).
# The interval: the Hawaii power path above costs at least half a second of kernel sleep
# per read, during which pm.mutex is held and the register-only reads of the same tick
# wait behind it; every ten seconds that is under five percent of the loop, and a single
# burst of four to thirteen messages at idle (two tables, two to eleven for the power).
# freq1_input and freq2_input are NEVER read on a legacy card, in any tier: they send the
# SAME message as the two tables (smu7_hwmgr.c:4012 = 4979, 4019 = 4995) and the table
# stars the first level at or above the clock that message returns (4981-4987, 4997-5003),
# so each of them costs one more message per burst for a number the table already shows
# (Fabrizio: «punto 2 ok», 17/09). They stay in the set below as a CLASSIFICATION: the
# set is what the tests forbid under load, and both files do message the SMC.
_AMD_SMC_MESSAGE_FILES = frozenset(("pp_dpm_sclk", "pp_dpm_mclk", "freq1_input",
                                    "freq2_input", "power1_average"))
_AMD_SLOW_INTERVAL = 10.0        # seconds between slow passes: the SMC tier and the slow tier
_AMD_IDLE_BUSY = 5               # busy percent at or under which the card counts as idle
_AMD_IDLE_WINDOW = 5.0           # seconds the register must have read idle since the last busy tick
_AMD_IDLE_SAMPLES = 3            # and consecutive idle ticks, whatever the tick period
_AMD_CLIENT_VRAM_MIB = 64        # resident VRAM at or above which a DRM client holds a working set
_PROC_ROOT = "/proc"             # where the DRM client scan walks; the tests point it elsewhere

# ------------------------------------------------- read tiers: what a value can do over time
# Fabrizio: «bisogna riuscire a discernere quali sono le informazioni che possono essere
# catturate ogni secondo e quali sono le informazioni che devono essere catturate una volta
# sola, non ha senso rimandare un polling o richiesta ogni volta, tanto quelle informazioni
# rimarranno tali». Every file a reader touches is one of three kinds, and the kind is read
# off the kernel's show function, not guessed:
#   STATIC  the value cannot change while the driver instance exists: a string parsed from
#           the VBIOS at init (vbios_version/vbios_build amdgpu_atombios.c:1825-1845, the
#           ctx->vbios_pn and build_num of the atom context; board_info amdgpu_device.c:
#           363-388), a firmware version loaded at init (amdgpu_ucode.c:737-770), a table
#           parsed at init (pp_num_states amd_powerplay.c:597-631 walks hwmgr->ps; fan1_min/
#           fan1_max amd_powerplay.c:820-825 return thermal_controller.fanInfo fields;
#           temp1_crit/temp1_crit_hyst amdgpu_pm.c:2794-2808 return pm.dpm.thermal.max_temp/
#           min_temp, set once in hardwaremanager.c:250-251), a literal (pwm1_min/pwm1_max
#           amdgpu_pm.c:2939-2951 print 0 and 255; power1_cap_min amd_powerplay.c:1015-1016
#           is 0; power1_label amdgpu_pm.c:3363-3376, temp1_label 2842-2852, freq/in labels
#           3426-3431, 3450-3455, 3219-3224), a size fixed at probe (mem_info_vram_total
#           amdgpu_vram_mgr.c:101-108 gmc.real_vram_size, mem_info_vis_vram_total 118-125,
#           mem_info_gtt_total amdgpu_gtt_mgr.c:43-53 man->size), the default and maximum
#           power limit (amd_powerplay.c:1005-1013, default_power_limit and the OD
#           percentage of the platform descriptor), or the existence of an attribute at all
#           (amdgpu_pm.c:2687-2688 never creates an unsupported one). Read ONCE at
#           construction, served from the cache for the life of the object, never opened by
#           the periodic sample.
#   SLOW    the value can change at runtime but only by root's hand or by a state machine
#           that moves in seconds: power1_cap (amd_powerplay.c:984 stores what root wrote),
#           power_dpm_state 214-234, power_dpm_force_performance_level 326-354, pp_cur_state
#           439-471, pp_force_state 473-484, pp_power_profile_mode 1342, pp_sclk_od/pp_mclk_od
#           1226, thermal_throttling_logging 1591-1602, pwm1_enable 2880-2900, fan1_enable
#           3128-3148, fan1_target 3069-3089 (the fan controller's rpm, a register read),
#           pp_dpm_pcie 1211-1216 -> smu7_hwmgr.c:5010-5025 (a static table with a register-
#           read marker), mem_info_vis_vram_used amdgpu_vram_mgr.c:154-163 and
#           mem_info_preempt_used amdgpu_preempt_mgr.c:36-45 (TTM usage of pools a model
#           does not live in), power_state drivers/pci/pci-sysfs.c:155-162 (the PCI D-state),
#           power/control and power/runtime_status drivers/base/power/sysfs.c:100-105,
#           149-165, and the PCIe link walk (current_link_* re-read a config word, 223-260;
#           max_link_* 198-221 are static but come through the shared pcie_chain(), which
#           this round leaves as it is). Read once per _AMD_SLOW_INTERVAL, ungated: none of
#           them sends an SMC message.
#   FAST    the value moves with the load and the panel exists to show it: gpu_busy_percent/
#           mem_busy_percent (smu7_hwmgr.c:4025-4038, a soft register), mem_info_vram_used/
#           mem_info_gtt_used (TTM usage, amdgpu_vram_mgr.c:135-144, amdgpu_gtt_mgr.c:63-72:
#           a model load moves them within a second), temp*_input (smu7_thermal.c:293-309),
#           in0_input (smu7_hwmgr.c:4067-4077), pwm1 and fan1_input (smu7_thermal.c:54-100).
#           Read every tick. On a card with gpu_metrics the blob and the clock tables are
#           fast too, as they always were; on a legacy card the clock tables and the power
#           are the SMC tier above, gated.
# The three tables below are the AMD reader's declaration, and the suite counts every open
# of a simulated minute against them. The fw_version/* files are static too (globbed once at
# construction, so they have no fixed basename to list here).
_AMD_STATIC_FILES = frozenset((
    "vbios_version", "vbios_build", "board_info", "pp_num_states",
    "mem_info_vram_total", "mem_info_gtt_total", "mem_info_vis_vram_total",
    "power1_cap_default", "power1_cap_min", "power1_cap_max", "power1_label",
    "fan1_min", "fan1_max", "pwm1_min", "pwm1_max",
    *(f"temp{n}_{s}" for n in (1, 2, 3) for s in ("label", "crit", "emergency", "crit_hyst")),
))
_AMD_SLOW_FILES = frozenset((
    "power_dpm_state", "power_dpm_force_performance_level", "pp_cur_state", "pp_force_state",
    "pp_power_profile_mode", "pp_sclk_od", "pp_mclk_od", "pp_dpm_pcie",
    "thermal_throttling_logging", "mem_info_vis_vram_used", "mem_info_preempt_used",
    "power_state", "control", "runtime_status",
    "power1_cap", "fan1_target", "pwm1_enable", "fan1_enable",
))
_AMD_TICK_FILES = frozenset((
    "gpu_busy_percent", "mem_busy_percent", "mem_info_vram_used", "mem_info_gtt_used",
    "temp1_input", "temp2_input", "temp3_input", "in0_input", "fan1_input", "pwm1",
    "gpu_metrics", "pp_dpm_fclk", "pp_dpm_socclk",       # only where the driver publishes them
))


class _Tiers:
    """The one place a reader keeps what it read once and when its slow pass last ran, so
    every device reader paces its reads the same way: STATIC values are read at construction
    into `static` and never again; a SLOW pass runs when `due()` says the interval has
    elapsed, the reader keeps its result in `values` and calls `stamp()`; FAST reads never
    come through here. The interval is passed at each call rather than stored, so the module
    constant that names it stays the single knob."""

    def __init__(self):
        self.static = {}
        self.values = None           # the last slow pass, None until the first
        self.at = None               # when it ran (monotonic)

    def due(self, now, interval):
        return self.at is None or now - self.at >= interval

    def stamp(self, now):
        self.at = now

    def age(self, now):
        return None if self.at is None else now - self.at


# -------------------------------------------------------- DRM clients of a card, from /proc
# The kernel names every open DRM file in the fdinfo of the descriptor that holds it
# (drm_file.c:974-1004 prints drm-driver, drm-client-id at 985 and drm-pdev at 990, then the
# driver's own keys). For amdgpu those are amdgpu_fdinfo.c:58-124: the per-VM memory stats,
# which are a memcpy of counters the driver maintains as buffers move (amdgpu_vm_get_memory,
# amdgpu_vm.c:1238-1244, under vm->status_lock: no walk of the client's buffers), the
# resident figures under the legacy drm-memory-<region> keys (100-105), and one
# drm-engine-<ip> line per hardware engine the client has run on, printed only when the
# accumulated time is non-zero (117-123). That time is the sum, per context and entity, of
# the scheduled-to-finished span of the last amdgpu_sched_jobs fences (amdgpu_ctx.c:170-200,
# 958-992, under the context manager's mutex and each context's ring_lock). So a fdinfo read
# costs a mutex, a walk of the client's contexts and a memcpy: kernel memory, no register, no
# SMC message, no pm.mutex; it is the generic drm_show_memory_stats (drm_file.c:915-961),
# which does walk every GEM handle under table_lock, that amdgpu does NOT use. The cost of a
# scan is therefore one fdinfo read per descriptor open on this card's nodes, plus the VFS
# work of listing /proc and resolving each fd link (fs/proc/fd.c:174-195, a path copy), and
# the scan runs only when the register window already passed: never under load.
# Both the fd links and the fdinfo files need PTRACE_MODE_READ on the process (fs/proc/base.c:
# 708-720 and 1799-1811 for the link, fs/proc/fd.c:85-101 for fdinfo): as a normal user the
# scan sees its own processes and counts the ones it may not read, so the panel can say so.
# A COMPUTE CLIENT is a client (one drm-client-id on this card's drm-pdev, however many
# descriptors share it, drm-usage-stats.rst:67-77) that has either run on a hardware engine
# of this card (any drm-engine-* time, so the counter need not be moving: an idle llama-server
# between two requests is exactly the client that can start work during a slow read) or holds
# at least _AMD_CLIENT_VRAM_MIB of RESIDENT VRAM (a loaded model that has not run yet).
# MEASURED (17/09 12:13, during a llama-bench Vulkan load): llama-bench, two descriptors on
# renderD128 with client id 584, 12235116 KiB resident and drm-engine-compute at
# 23017368693 ns then 23021380298 ns a moment later; Xorg, five descriptors on card1 with
# client id 38, 0 KiB resident (10152 KiB requested and all of it evicted) and no drm-engine
# line at all, because it has never rendered on this card. The resident key, not the
# requested one, is what separates them; the three fds llama-bench held on the NVIDIA render
# nodes carried no drm-* key and are never read (their nodes are not this card's). Declared
# not measured here: a display server that does drive this card renders on gfx and would
# count as a compute client, shutting the gate with its name in the reason.
_DRM_NODE = re.compile(r"^(card|renderD)\d+$")


def _drm_nodes_of(dev):
    """The names of this card's DRM nodes (card1, renderD128), as the kernel lists them
    under the PCI device's drm/ directory; empty when the driver publishes none."""
    try:
        return {n for n in os.listdir(f"{dev}/drm") if _DRM_NODE.match(n)}
    except OSError:
        return set()


def _drm_fdinfo_parse(text):
    """One fdinfo text as a record, None when it carries no drm-driver key (a plain file, a
    NVIDIA node). Sizes default to bytes with an optional KiB/MiB unit (drm-usage-stats.rst:
    160-161); the resident VRAM is the legacy drm-memory-vram key when present, else
    drm-resident-vram. engine_ns is the sum over the drm-engine-* lines, capacity lines
    excluded, and 0 when there is none: amdgpu prints no line for a zero counter."""
    keys = {}
    for line in (text or "").splitlines():
        k, sep, v = line.partition(":")
        if sep and k.startswith("drm-"):
            keys[k.strip()] = v.strip()
    if "drm-driver" not in keys:
        return None

    def kib(key):
        m = re.match(r"(\d+)\s*(KiB|MiB)?$", keys.get(key, ""))
        if not m:
            return 0
        n, unit = int(m.group(1)), m.group(2)
        return n * 1024 if unit == "MiB" else n if unit == "KiB" else n // 1024
    engine = 0
    for k, v in keys.items():
        if k.startswith("drm-engine-") and not k.startswith("drm-engine-capacity-"):
            m = re.match(r"(\d+)", v)
            engine += int(m.group(1)) if m else 0
    try:
        cid = int(keys.get("drm-client-id", ""))
    except ValueError:
        cid = None
    return {"driver": keys["drm-driver"], "client_id": cid, "pdev": keys.get("drm-pdev"),
            "vram_kib": kib("drm-memory-vram" if "drm-memory-vram" in keys else "drm-resident-vram"),
            "engine_ns": engine}


def _amd_is_compute(client):
    return client["engine_ns"] > 0 or client["vram_mib"] >= _AMD_CLIENT_VRAM_MIB


def _amd_compute_clients(clients):
    """How many of `clients` can put work on the card: see the block above for the rule."""
    return sum(1 for c in clients if _amd_is_compute(c))


def _drm_clients_of(pdev, nodes, proc_root=None):
    """Every DRM client of the card at `pdev` whose node names are `nodes`, read from /proc
    alone: the target of each fd link (a path copy resolved by the VFS) and, only for a link
    into /dev/dri at one of this card's nodes, that fd's fdinfo. Never a device node, never
    anything but a read. Clients are one record per drm-client-id, whatever the number of
    descriptors, holding the largest resident VRAM and engine time seen across them, sorted
    with the largest holder first. A process that exits between the listing and a read is
    skipped; one whose descriptors may not be read is counted in `unreadable`."""
    root = _PROC_ROOT if proc_root is None else proc_root
    clients, unreadable, pids, fds = {}, 0, 0, 0
    try:
        entries = os.listdir(root)
    except OSError:
        entries = []
    for pid in entries:
        if not pid.isdigit():
            continue
        pids += 1
        base = f"{root}/{pid}"
        try:
            names = os.listdir(f"{base}/fd")
        except PermissionError:
            unreadable += 1
            continue
        except OSError:                          # gone between the listing and this
            continue
        for fd in names:
            try:
                target = os.readlink(f"{base}/fd/{fd}")
            except OSError:
                continue
            if os.path.dirname(target) != "/dev/dri" or os.path.basename(target) not in nodes:
                continue
            fds += 1
            rec = _drm_fdinfo_parse(read(f"{base}/fdinfo/{fd}"))
            if not rec or rec["pdev"] != pdev or rec["client_id"] is None:
                continue
            c = clients.get(rec["client_id"])
            if c is None:
                c = clients[rec["client_id"]] = {"id": rec["client_id"], "pid": int(pid),
                                                 "comm": read(f"{base}/comm", "") or "",
                                                 "vram_mib": 0, "engine_ns": 0}
            c["vram_mib"] = max(c["vram_mib"], rec["vram_kib"] // 1024)
            c["engine_ns"] = max(c["engine_ns"], rec["engine_ns"])
    out = sorted(clients.values(), key=lambda c: (-c["vram_mib"], -c["engine_ns"], c["id"]))
    for c in out:
        c["compute"] = _amd_is_compute(c)
    return {"clients": out, "unreadable": unreadable, "pids": pids, "fds": fds}


def _amd_client_reason(clients):
    """The "idle only" reason when a compute client shuts the SMC gate, None when none does.
    Short on purpose, as the other reasons are: the "no data" row prefixes it with the field
    names and holds 66 cells at 80 columns, so with a 15-character comm this stays inside."""
    comp = [c for c in clients if c.get("compute")]
    if not comp:
        return None
    if len(comp) > 1:
        return f"idle only (SMC): {len(comp)} compute clients"
    c = comp[0]
    if c["vram_mib"] >= _AMD_CLIENT_VRAM_MIB:
        return f"idle only (SMC): {c['comm']} holds {c['vram_mib'] / 1024:.1f} GiB"
    return f"idle only (SMC): {c['comm']} on the GPU"


def _amd_clients_text(a):
    """The clocks-group cell: how many clients can put work on the card, the largest
    holder, what the scan could not read, and the scan's age."""
    if not a.get("clients_scans"):
        return a.get("clients_why", "")
    comp = [c for c in a.get("clients") or [] if c.get("compute")]
    parts = [f"{len(comp)} compute" if comp else "none compute"]
    if comp:
        c = comp[0]
        parts.append(f"{c['comm']} {c['vram_mib'] / 1024:.1f} GiB"
                     if c["vram_mib"] >= _AMD_CLIENT_VRAM_MIB else c["comm"])
    if a.get("clients_unreadable"):
        parts.append(f"{a['clients_unreadable']} unreadable")
    if a.get("clients_age") is not None:
        parts.append(f"scan {_fmt_dur(a['clients_age'])} ago")
    return " · ".join(parts)


def _amd_dpm_levels(text):
    """(levels, active) from a pp_dpm_* table: the label of each `N: label` line with the
    trailing marker removed, and the index of the line carrying `*`, None without one."""
    levels, active = [], None
    for line in (text or "").splitlines():
        m = re.match(r"\s*(\d+):\s*(.*?)\s*(\*)?\s*$", line)
        if not m:
            continue
        if m.group(3):
            active = len(levels)
        levels.append(m.group(2))
    return levels, active


def _amd_clock_levels(text):
    """As _amd_dpm_levels, for a clock table: MHz integers, or (None, None) when any level
    is outside 1..6000 MHz, the same bound sane() applies to the live clocks."""
    labels, active = _amd_dpm_levels(text)
    mhz = []
    for lab in labels:
        m = re.match(r"(\d+)\s*Mhz", lab, re.I)
        v = int(m.group(1)) if m else None
        if v is None or not 1 <= v <= 6000:
            return None, None
        mhz.append(v)
    return (mhz or None), (active if mhz else None)


class AmdGpu:
    vendor = "AMD"

    def __init__(self, card, name):
        self.card = card
        self.name = name
        self.dev = f"{card}/device"
        self.hw = _hwmon_of(card)
        # the PCI address the kernel prints as drm-pdev in every client's fdinfo: the
        # device link resolves to the PCI node, whose name is the address
        self.pdev = os.path.basename(os.path.realpath(self.dev))
        # legacy powerplay (smu7 and its relatives), recognised by the file the driver does
        # not create for it; see the classification above
        self.legacy = not os.path.exists(f"{self.dev}/gpu_metrics")
        self.tiers = _Tiers()        # the static values and the ungated slow pass
        self._smc = _Tiers()         # the gated SMC pass of a legacy card
        self._busy_t = None          # when the register last read busy (monotonic)
        self._idle_n = 0             # consecutive idle ticks since then
        self._clients = None         # the last DRM client scan, see _drm_clients_of
        self._clients_t = None       # when it ran
        self._clients_n = 0          # how many scans so far
        self._nodes = _drm_nodes_of(self.dev)
        self._read_static()
        self.temp_labels = self.tiers.static["temp_labels"]
        self.temp_crit = self.tiers.static["temp_crit_map"]
        self._guide = ("junction" if "junction" in self.temp_labels
                       else next(iter(self.temp_labels), None))

    def _read_static(self):
        """Everything the driver publishes once (see the read tiers above): the identity
        (VBIOS part number and build, board form factor, firmware versions), the labels and
        thresholds of the temperature channels, the power-state table, the memory totals,
        the power-cap range and label, the fan and pwm ranges, and the existence of the
        optional attributes the periodic sample would otherwise probe every tick. Read at
        construction into tiers.static, labelled "at start" in the panel where it shows,
        and never opened again by sample(). `missing` names what this driver does not
        publish, in the same style the slow pass uses."""
        st, missing = self.tiers.static, []
        st["missing"] = missing
        card = self.card

        def attr(name, root=None):
            path = f"{root or self.dev}/{name}"
            if not os.path.exists(path):
                missing.append(name)
                return None
            return read(path)

        def num(name, key, lo, hi, scale=1, root=None):
            v = attr(name, root)
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
            return sane(f"{card}.amd.{key}", v // scale, lo, hi)     # floor, as mib() does
        for key, name in (("vbios", "vbios_version"), ("vbios_build", "vbios_build"),
                          ("board", "board_info")):
            v = attr(name)
            st[key] = v.split(":", 1)[-1].strip() if (v and key == "board") else v
        fw_dir = f"{self.dev}/fw_version"
        if os.path.isdir(fw_dir):
            st["fw"] = {os.path.basename(f)[:-len("_fw_version")]: read(f)
                        for f in sorted(glob.glob(f"{fw_dir}/*_fw_version"))}
        else:
            missing.append("fw_version")
            st["fw"] = None
        txt = attr("pp_num_states")
        st["states"] = ([(int(m.group(1)), m.group(2))
                         for m in re.finditer(r"^(\d+)\s+(\w+)\s*$", txt, re.M)] or None) if txt else None
        # read_int() answers 0 for a file that is not there, and a total of 0 MiB would draw
        # a full empty bar: None, as sample() already does for the used figures
        for key, name in (("vram_total", "mem_info_vram_total"), ("gtt_total", "mem_info_gtt_total")):
            v = read_int(f"{self.dev}/{name}", None)
            st[key] = None if v is None else v // 1048576
        vram_total = st["vram_total"] or (1 << 22)
        st["vis_vram_total"] = num("mem_info_vis_vram_total", "vis_total", 1, vram_total, 1048576)
        # the attributes whose absence is as fixed as their presence (amdgpu_pm.c:2687-2688,
        # hwmon_attributes_visible): decided once, so sample() never opens a file the driver
        # did not create
        st["has"] = {n: os.path.exists(f"{self.dev}/{n}") for n in ("pp_dpm_fclk", "pp_dpm_socclk")}
        st["has"]["fan1_input"] = bool(self.hw) and os.path.exists(f"{self.hw}/fan1_input")
        st["temp_labels"] = _hwmon_temp_labels(self.hw)
        # per-sensor critical and emergency thresholds declared by the driver, better than
        # an invented number for colouring the temperatures; temp1_crit is also the "temp
        # crit" of the limits row, read the once for both
        st["temp_crit_map"], crit_raw = {}, {}
        for label, key in st["temp_labels"].items():
            crit_raw[key] = read_int(f"{self.hw}/{key}_crit", None)
            st["temp_crit_map"][label] = ((crit_raw[key] or 0) // 1000,
                                          read_int(f"{self.hw}/{key}_emergency") // 1000)
        if self.hw:
            hw = self.hw
            for key, name in (("power_cap_default", "power1_cap_default"),
                              ("power_cap_min", "power1_cap_min"),
                              ("power_cap_max", "power1_cap_max")):
                st[key] = num(name, key, 0, 2000, 1000000, hw)
            st["power_label"] = attr("power1_label", hw)
            st["fan_min"] = num("fan1_min", "fan_min", 0, 20000, root=hw)
            st["fan_max"] = num("fan1_max", "fan_max", 0, 20000, root=hw)
            st["pwm_min"] = num("pwm1_min", "pwm_min", 0, 255, root=hw)
            st["pwm_max"] = num("pwm1_max", "pwm_max", 0, 255, root=hw)
            if "temp1" in crit_raw:
                crit = crit_raw["temp1"]
                if crit is None and not os.path.exists(f"{hw}/temp1_crit"):
                    missing.append("temp1_crit")
                st["temp_crit"] = (None if crit is None
                                   else sane(f"{card}.amd.temp_crit", crit // 1000, 1, 130))
            else:
                st["temp_crit"] = num("temp1_crit", "temp_crit", 1, 130, 1000, hw)
            hyst = attr("temp1_crit_hyst", hw)
            try:
                hyst = int(hyst)
            except (TypeError, ValueError):
                hyst = None
            # -273150 is the driver's "unset" (min_temp never programmed), not a threshold
            st["temp_crit_hyst"] = (None if hyst is None or hyst <= -273000
                                    else sane(f"{card}.amd.hyst", hyst // 1000, -50, 130))
        else:
            missing.append("hwmon")
            for key in ("power_cap_default", "power_cap_min", "power_cap_max", "power_label",
                        "fan_min", "fan_max", "pwm_min", "pwm_max", "temp_crit", "temp_crit_hyst"):
                st[key] = None

    def _scan_clients(self, now):
        """The DRM client scan, at most once per _AMD_SLOW_INTERVAL and only from the SMC
        gate, so it never runs under load: the reason it exists is to be cheap on the
        driver. Returns the compute-client reason, None when no known client shuts the gate.
        Without a resolvable node there is nothing to match a link against and /proc is not
        walked at all: the register window alone gates the card."""
        if not self._nodes:
            return None
        if self._clients_t is None or now - self._clients_t >= _AMD_SLOW_INTERVAL:
            self._clients = _drm_clients_of(self.pdev, self._nodes)
            self._clients_t = now
            self._clients_n += 1
        return _amd_client_reason(self._clients["clients"])

    def _smc_tier(self, d, now):
        """The five SMC-message reads, gated as the block above the class describes: the
        register window, the interval, and the DRM clients scanned just before the read.
        Returns (fields for the sample, fields for the detail dict, needs reasons)."""
        util, mem = d.get("util"), d.get("mem_util")
        idle = (util is not None and util <= _AMD_IDLE_BUSY
                and (mem is None or mem <= _AMD_IDLE_BUSY))
        if idle:
            self._idle_n += 1
        else:
            self._busy_t, self._idle_n = now, 0
        settled = idle and (self._busy_t is None
                            or (now - self._busy_t >= _AMD_IDLE_WINDOW
                                and self._idle_n >= _AMD_IDLE_SAMPLES))
        why_client = None
        if settled and self._smc.due(now, _AMD_SLOW_INTERVAL):
            why_client = self._scan_clients(now)
            if why_client is None:
                r = {}
                for k, f in (("sclk", "pp_dpm_sclk"), ("mclk", "pp_dpm_mclk")):
                    levels, active = _amd_clock_levels(read(f"{self.dev}/{f}"))
                    r[f"{k}_levels"], r[f"{k}_active"] = levels, active
                    r[k] = sane(f"{self.card}.{k}",
                                levels[active] if levels and active is not None else None, 1, 6000)
                uw = read_int(f"{self.hw}/power1_average", None) if self.hw else None
                r["power"] = sane(f"{self.card}.w", uw / 1e6 if uw is not None else None, 0.1, 800)
                self._smc.values = r
                self._smc.stamp(now)
        elif settled and self._clients is not None:
            # inside the interval: the last scan's verdict still stands, as its values do
            why_client = _amd_client_reason(self._clients["clients"])
        vals, age = self._smc.values, self._smc.age(now)
        top = {"slow_age": age, "slow_fields": [], "sclk": None, "mclk": None, "power": None}
        detail = {k: (vals or {}).get(k)
                  for k in ("sclk_levels", "mclk_levels", "sclk_active", "mclk_active")}
        # the reasons are short on purpose: the "no data" row prefixes them with the field
        # names ("power, clocks — ") and has 66 cells at 80 columns, past which _put clips
        # the sentence; _fmt_dur keeps an age of hours to eight characters
        if util is None:
            why = "idle only (SMC): no gpu_busy_percent"
        elif why_client:
            why = why_client
        elif vals is None:
            why = "idle only (SMC): waiting for an idle tick"
        elif not settled:
            why = f"idle only (SMC): GPU busy, idle read {_fmt_dur(age)} ago"
        else:
            why = None
            for k in ("sclk", "mclk", "power"):
                top[k] = vals.get(k)
            top["slow_fields"] = [k for k in ("sclk", "mclk", "power") if vals.get(k) is not None]
        if why and vals is not None:
            # the level lists are static tables; the markers are not
            detail.update(sclk_active=None, mclk_active=None)
        needs = {}
        if why:
            needs["clocks"] = why
            if self.hw:
                needs["power"] = why
        return top, detail, needs

    def _read_slow(self):
        """The slow pass (see the read tiers above): the levers root can move and the
        states that move in seconds, read once per _AMD_SLOW_INTERVAL from the register-
        only and stored-field attributes, never gated because none of them messages the
        SMC. Every numeric passes sane() with a bound; an attribute the driver does not
        publish is None and named in `missing`."""
        a, missing = {}, []
        st, card = self.tiers.static, self.card

        def attr(name, root=None):
            path = f"{root or self.dev}/{name}"
            if not os.path.exists(path):
                missing.append(name)
                return None
            return read(path)

        def num(name, key, lo, hi, scale=1, root=None):
            v = attr(name, root)
            try:
                v = int(v)
            except (TypeError, ValueError):
                return None
            return sane(f"{card}.amd.{key}", v // scale, lo, hi)     # floor, as mib() does

        a["dpm_state"] = attr("power_dpm_state")
        a["perf_level"] = attr("power_dpm_force_performance_level")
        a["cur_state"] = num("pp_cur_state", "cur_state", 0,
                             len(st["states"]) - 1 if st["states"] else 15)
        txt = attr("pp_force_state")
        a["forced_state"] = int(txt) if txt and txt.isdigit() else None
        txt = attr("pp_power_profile_mode")
        m = re.search(r"^\s*\d+\s+(\S+)\s*\*:", txt or "", re.M)
        a["profile"] = m.group(1) if m else None
        a["sclk_od"] = num("pp_sclk_od", "sclk_od", 0, 100)
        a["mclk_od"] = num("pp_mclk_od", "mclk_od", 0, 100)
        levels, active = _amd_dpm_levels(attr("pp_dpm_pcie"))
        a["pcie_levels"], a["pcie_active"] = (levels or None), active
        m = re.search(r"throttling logging (enabled|disabled), with interval (\d+) seconds",
                      attr("thermal_throttling_logging") or "")
        a["throttle_log"] = ((m.group(1), int(m.group(2)))
                             if m and 1 <= int(m.group(2)) <= 86400 else None)
        vram_total = st["vram_total"] or (1 << 22)
        a["vis_vram_used"] = num("mem_info_vis_vram_used", "vis_used", 0,
                                 st["vis_vram_total"] or vram_total, 1048576)
        a["preempt_used"] = num("mem_info_preempt_used", "preempt", 0, 1 << 20, 1048576)
        a["pci_power_state"] = attr("power_state")
        a["runtime_control"] = attr("power/control")
        a["runtime_status"] = attr("power/runtime_status")
        if self.hw:
            hw = self.hw
            a["power_cap"] = read_int(f"{hw}/power1_cap") // 1000000 or None
            a["fan_target"] = num("fan1_target", "fan_target", 0, 20000, root=hw)
            mode = num("pwm1_enable", "pwm_enable", 0, 2, root=hw)
            a["fan_mode"] = {0: "none", 1: "manual", 2: "auto"}.get(mode)
            a["fan_enable"] = num("fan1_enable", "fan_enable", 0, 1, root=hw)
        else:
            a["power_cap"] = a["fan_target"] = a["fan_mode"] = a["fan_enable"] = None
        a["pcie"] = pcie_chain(self.dev)
        a["missing"] = missing
        return a

    def _amd_details(self, d, smc, slow, now):
        """Everything the driver publishes for the card beyond the main row: the static
        values read at construction, the slow pass, the SMC tier's tables and the DRM
        client scan, plus the one per-tick verdict that needs no read (why the fan has no
        figure: a file that exists but yields nothing, the tachometer of a passive board,
        is None with its own reason, never 0). `missing` is the union of what the two
        passes found absent."""
        st = self.tiers.static
        a = {k: v for k, v in st.items()
             if k not in ("missing", "has", "temp_labels", "temp_crit_map", "vram_total", "gtt_total")}
        a.update((k, v) for k, v in slow.items() if k not in ("missing", "pcie", "power_cap"))
        a.update(smc)
        if self.hw:
            if not st["has"]["fan1_input"]:
                a["fan_why"] = "no fan1_input on this hwmon node"
            elif d.get("fan_rpm") is None:
                a["fan_why"] = "no tachometer reading (fan1_input has no value)"
            else:
                a["fan_why"] = None
        scan = self._clients or {}
        a["clients"] = scan.get("clients", [])
        a["clients_compute"] = _amd_compute_clients(a["clients"])
        a["clients_scans"] = self._clients_n
        a["clients_age"] = None if self._clients_t is None else now - self._clients_t
        a["clients_unreadable"] = scan.get("unreadable", 0)
        a["clients_why"] = None if self._nodes else f"no drm node under {self.dev}/drm"
        a["missing"] = list(st["missing"]) + [m for m in slow.get("missing", ())
                                              if m not in st["missing"]]
        return a

    def sample(self):
        now = time.monotonic()
        st = self.tiers.static
        d = {"vendor": "AMD", "name": self.name, "temp_crit": self.temp_crit}
        # read_int() answers 0 for a file that is not there, because that is its default,
        # and the six readings below used to take it bare. A card whose driver does not
        # publish gpu_busy_percent therefore reported 0 % utilisation, and one without
        # mem_info_vram_* reported 0 MiB: not a dash, but a full empty bar drawn at zero,
        # which is the one thing sane()'s own docstring forbids. Against a sysfs tree with
        # those files absent the sample came back as util 0, mem_util 0, vram_total 0 and
        # gtt_total 0, with no reason recorded anywhere. Passing None as the default is
        # the whole fix, since read_int already accepts one. The two totals are static
        # (see the read tiers) and come from the construction-time read.
        def mib(f):
            v = read_int(f"{self.dev}/{f}", None)
            return None if v is None else v // 1048576
        d["vram_used"] = mib("mem_info_vram_used")
        d["vram_total"] = st["vram_total"]
        d["gtt_used"] = mib("mem_info_gtt_used")
        d["gtt_total"] = st["gtt_total"]
        d["vram_free"] = ((d["vram_total"] - d["vram_used"])
                          if d["vram_total"] and d["vram_used"] is not None else None)
        d["util"] = read_int(f"{self.dev}/gpu_busy_percent", None)
        d["mem_util"] = read_int(f"{self.dev}/mem_busy_percent", None)

        def dpm(f):
            for line in (read(f"{self.dev}/{f}") or "").splitlines():
                if "*" in line:
                    m = re.search(r"(\d+)\s*Mhz", line, re.I)
                    if m:
                        return int(m.group(1))
            return None
        # On a legacy card the sclk/mclk tables and the power are SMC messages (see the
        # classification above the class) and come from the SMC tier;
        # every other card reads them as it always did.
        slow_needs, amd_smc = {}, {}
        if self.legacy:
            top, amd_smc, slow_needs = self._smc_tier(d, now)
            d.update(top)
        else:
            d["sclk"] = sane(f"{self.card}.sclk", dpm("pp_dpm_sclk"), 1, 6000)
            d["mclk"] = sane(f"{self.card}.mclk", dpm("pp_dpm_mclk"), 1, 6000)
        # the two tables a legacy card does not publish: their absence was decided once
        d["fclk"] = sane(f"{self.card}.fclk",
                         dpm("pp_dpm_fclk") if st["has"]["pp_dpm_fclk"] else None, 1, 6000)
        d["socclk"] = sane(f"{self.card}.socclk",
                           dpm("pp_dpm_socclk") if st["has"]["pp_dpm_socclk"] else None, 1, 6000)
        d["temp"] = {}
        for label, key in self.temp_labels.items():
            d["temp"][label] = sane(f"{self.card}.t.{label}",
                                    read_int(f"{self.hw}/{key}_input") // 1000, 1, 130)
        d["temp_main"] = d["temp"].get(self._guide) if self._guide else None
        if self.hw:
            if not self.legacy:              # on a legacy card the SMC tier owns power
                d["power"] = sane(f"{self.card}.w",
                                  read_int(f"{self.hw}/power1_average") / 1e6, 0.1, 800)
            # vddgfx has been seen at 6 mV while the card drew 8 W: below 400 it is
            # not a reading
            d["voltage"] = sane(f"{self.card}.mv", read_int(f"{self.hw}/in0_input"), 400, 1400)
            # A card in zero-fan mode reads 0 RPM, and 0 is a reading, not an absence:
            # recent cards park their fans when cool, so this is the commonest idle state
            # and not an edge case. `or None` reported it as "no sensor", and the panel's
            # guard compounded it by testing truthiness, so the row disappeared entirely,
            # a stopped fan and a card with no tachometer looked exactly alike. The same
            # line stood in the Intel reader; both are fixed, both are covered by a test.
            d["fan_rpm"] = read_int(f"{self.hw}/fan1_input", None) if st["has"]["fan1_input"] else None
            _pwm = read_int(f"{self.hw}/pwm1", -1)      # 0–255 duty cycle → percent
            d["fan_pct"] = round(_pwm * 100 / 255) if _pwm >= 0 else None
        if not self.legacy:                  # a legacy card has no gpu_metrics: decided once
            d.update(_amd_gpu_metrics(self.card))
        if d.get("mem_clock"):
            d["mclk"] = d["mem_clock"]
        if d.get("gfx_clock"):
            d["sclk"] = d["gfx_clock"]
        # the VRM temperatures join the others rather than getting a line of their own: the
        # panel already iterates `temp` and colours each against its driver-declared limits.
        # hwmon declares no limit for these two, so they fall back to the fixed 80/90 °C,
        # which is the same treatment any unlabelled sensor gets, not a special case.
        for _k, _lbl in (("vrm_temp_gfx", "vrm gfx"), ("vrm_temp_soc", "vrm soc")):
            if d.get(_k) is not None:
                d.setdefault("temp", {})[_lbl] = d[_k]
        # the slow pass: the levers, the states and the PCIe link, once per interval
        if self.tiers.due(now, _AMD_SLOW_INTERVAL):
            self.tiers.values = self._read_slow()
            self.tiers.stamp(now)
        slow = self.tiers.values
        d["power_cap"] = slow["power_cap"]
        d["pcie"] = slow["pcie"]
        d["amd"] = self._amd_details(d, amd_smc, slow, now)
        # Why a field is empty. On AMD the answers are of a different kind than on
        # the other two vendors, which is why the wording is not shared with them. Intel
        # and NVIDIA are usually missing a program, so their reasons name a binary
        # (intel_gpu_top, nvtop, nvidia-smi) and installing it clears the dash. AMD reads
        # almost everything straight from sysfs, so the honest answer is about the driver
        # and the card: an attribute this kernel module does not publish, a card with no
        # hwmon node, a hwmon node missing one of its optional entries. Telling an AMD
        # user to install something would send them after a package that changes nothing.
        # The five keys are the closed set _gpu_rows iterates; a sixth would be written
        # here and never rendered.
        needs = {}
        if d.get("util") is None:
            needs["util"] = "gpu_busy_percent not exposed by this driver"
        if d.get("vram_total") is None:
            needs["vram"] = "mem_info_vram_* not exposed by this driver"
        if all(d.get(k) is None for k in ("sclk", "mclk", "fclk", "socclk")):
            needs["clocks"] = "no pp_dpm_* table and no clocks in gpu_metrics"
        if not self.hw:
            needs["temp"] = needs["power"] = "no hwmon node on this card"
        else:
            # Having the node is not having its files. hwmon exposes an optional set,
            # and partial hardware is the shape that slips through a check written for
            # missing hardware: an Arc A770 reports temperature but no energy counter.
            if d.get("power") is None:
                needs["power"] = "no power1_average on this hwmon node"
            if not d.get("temp"):
                needs["temp"] = "no temperature input on this hwmon node"
        # the slow tier's reason for a legacy card replaces the generic one for the fields
        # it owns: "no pp_dpm_* table" would be false, the table is there and unread
        needs.update(slow_needs)
        if needs:
            d["needs"] = needs
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
        # recorded at construction so a panel can say which program is missing instead of
        # printing a bare dash. `which` is cheap and the answer cannot change mid-run.
        self.tool_ok = which("intel_gpu_top")

    def run(self):
        if not self.tool_ok:
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
                    # ("RCS BCS VCS VECS CCS"); older builds used an "RCS/0" suffix; accept both.
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
        # recorded at construction so a panel can say which program is missing instead of
        # printing a bare dash. `which` is cheap and the answer cannot change mid-run.
        self.tool_ok = which("nvtop")
        # `nvtop -s` takes seconds to answer the first time, and during that window an empty
        # by_key is not the same fact as "nvtop does not report this card": saying the
        # second while the first is true prints something false for the first few ticks.
        self.answered = False

    def run(self):
        import json
        if not self.tool_ok:
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
                self.answered = True
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
            # A counter that stops answering is not a steady load. This returned the last
            # computed wattage with no age check, so a card whose energy attribute became
            # unreadable displayed a frozen number for the rest of the session, styled
            # exactly like a live one. `t` is already the timestamp of the last good
            # counter read, so the age was available all along and simply not consulted,
            # the same defect sane() carried, in the same file, on a different cache.
            if time.monotonic() - self._energy["t"] > _SANE_TTL:
                return None
            return self._energy["w"]
        t = time.monotonic()
        if self._energy["j"] and t > self._energy["t"]:
            dt = t - self._energy["t"]
            if dt > 0.4:
                dj = j - self._energy["j"]
                # A counter that went backwards has wrapped, and clamping the negative
                # delta to zero printed that as "0 W": a fabricated reading, and on screen
                # no different from a card drawing nothing. The CPU path can correct the
                # same event by adding max_energy_range_uj back, because intel-rapl
                # publishes that range. hwmon declares no equivalent for energy1_input, so
                # the wrap cannot be undone here. Keeping the last good figure and
                # re-baselining costs one stale tick instead of one false one, and the
                # tick after it reads correctly again.
                if dj >= 0:
                    self._energy["w"] = dj / dt / 1e6
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
            # Not setdefault. The line above inserts "sclk" whenever the engine feed is
            # alive, whatever its frequency column held, None and 0 included, and
            # setdefault only fills a key that is absent, never one that is merely falsy.
            # From that point it was a guaranteed no-op and the sysfs reading below was
            # computed and discarded: with a card holding a good 1350 in sysfs and a feed
            # reporting 0, the sample came out with no clock at all. A key being present
            # is not the same fact as a key having a value.
            if d.get("sclk") is None:
                # rps_act_freq_mhz is the actual frequency, and an idle i915 is power
                # gated at 0: ten reads in three seconds return zero while the panel is
                # polling, and the setpoint files beside it read 350, 600 and 1450 because
                # they describe other things. Four layers used to agree in hiding that
                # reading. A floor of 1 refused it, a trailing `or None` would have killed
                # it even if the floor had let it through, read_int with no default hands
                # back 0 for a file that is not there, and the panel guarded the row on
                # truthiness. The default matters most: without it, absent and gated
                # collapse into the same value, so the floor could not simply be lowered.
                d["sclk"] = sane(f"{self.card}.f",
                                 read_int(f"{self.gt}/rps_act_freq_mhz", None), 0, 4000)
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
            # A card in zero-fan mode reads 0 RPM, and 0 is a reading, not an absence:
            # recent cards park their fans when cool, so this is the commonest idle state
            # and not an edge case. `or None` reported it as "no sensor", and the panel's
            # guard compounded it by testing truthiness, so the row disappeared entirely,
            # a stopped fan and a card with no tachometer looked exactly alike. The same
            # line stood in the Intel reader; both are fixed, both are covered by a test.
            d["fan_rpm"] = read_int(f"{self.hw}/fan1_input", None)
            _pwm = read_int(f"{self.hw}/pwm1", -1)      # 0–255 duty cycle → percent
            d["fan_pct"] = round(_pwm * 100 / 255) if _pwm >= 0 else None
            d["power"] = self._power()
            d["power_cap"] = read_int(f"{self.hw}/power1_max") // 1000000 or None
        d["pcie"] = pcie_chain(self.dev)
        # VRAM from nvtop if available
        if self.vram_feed:
            mine = norm_name(self.name)
            rows = list(self.vram_feed.by_key.items())
            # identity first; the old vendor-substring test is kept as a fallback for a
            # card whose two sources genuinely disagree on the name (a discrete Arc reads
            # 'DG2 [Arc A770]' from pci.ids and 'Intel Arc A770 Graphics' from nvtop)
            hit = next((v for nm, v in rows if norm_name(nm) == mine), None)
            if hit is None:
                hit = next((v for nm, v in rows
                            if "Arc" in nm or "DG2" in nm or "Intel" in nm), None)
            if hit is not None:
                used, total, ts, nvutil = hit
                if time.monotonic() - ts < 8:
                    # An integrated GPU has no memory of its own, and the test for that is
                    # the total, not the used figure. An earlier version of this guard
                    # rested on the idea that nvtop reports no used figure for integrated
                    # graphics, which turns out to be false at least some of the time: on a
                    # TigerLake-H iGPU it answers with a real used figure alongside a total
                    # of 33412202496 bytes, which is 31864 MiB and exactly what
                    # /proc/meminfo reports as MemTotal. The guard therefore never fired,
                    # the pair was taken, and the panel drew a half-full VRAM bar whose
                    # denominator was the machine's RAM. The numerator matched nothing in
                    # /proc/meminfo either, so it was a number under a label that did not
                    # describe it.
                    # Comparing the totals is a measurement rather than a guess at the
                    # card's nature: a discrete Arc reports its own 16 GiB against the
                    # machine's RAM total and keeps its bar.
                    if total and total == RAM_TOTAL_MIB:
                        d["vram_shared"] = True
                    elif used is not None and total:
                        d["vram_used"], d["vram_total"] = used, total
                    elif total:
                        # the older signature: a total with no used figure at all
                        d["vram_shared"] = True
                    if d.get("util") is None and nvutil is not None:
                        d["util"] = nvutil       # fallback when intel_gpu_top can't read the PMU
        # Why a field is empty. On Intel the answer differs per field because the sources
        # do: engine load comes from intel_gpu_top with nvtop as a fallback, VRAM only from
        # nvtop, and temperature and power from the card's own hwmon node, which an
        # integrated GPU may simply not have. The reason names the binary and never the
        # package, since the same tools ship as intel-gpu-tools on some distributions and
        # igt-gpu-tools on others while the binary name stays put.
        needs = {}
        _eng = self.engine_feed
        if d.get("util") is None:
            if _eng is not None and not _eng.tool_ok:
                needs["util"] = ("needs intel_gpu_top or nvtop"
                                 if (self.vram_feed and not self.vram_feed.tool_ok)
                                 else "needs intel_gpu_top")
            elif self.vram_feed is not None and self.vram_feed.tool_ok and not self.vram_feed.answered:
                needs["util"] = "waiting for nvtop"
            else:
                needs["util"] = "no engine data from intel_gpu_top"
        if not d.get("vram_total"):
            if d.get("vram_shared"):
                needs["vram"] = "memory is shared with system RAM"
            elif self.vram_feed is not None and not self.vram_feed.tool_ok:
                needs["vram"] = "needs nvtop"
            elif self.vram_feed is not None and not self.vram_feed.answered:
                needs["vram"] = "waiting for nvtop"
            else:
                needs["vram"] = "not reported by nvtop for this card"
        if not self.hw:
            needs["temp"] = needs["power"] = "no hwmon node on this card"
        else:
            # Having the node is not having its files. hwmon is a directory of optional
            # entries, and the check above only covered the case where the whole node is
            # absent. On an Arc A770 the temperature reads fine, so the node is plainly
            # there, while power stays empty because that card exposes no energy1_input
            # for the derivative to work on. A field left empty for an unexplained reason
            # is exactly what this mechanism exists to prevent, and it survived here in
            # the shape the check did not cover: partial hardware rather than missing
            # hardware.
            if d.get("power") is None:
                needs["power"] = "no energy counter (energy1_input) on this card"
            if not d.get("temp"):
                needs["temp"] = "no temperature input on this hwmon node"
        if needs:
            d["needs"] = needs
        return d


# ----------------------------------------------------------------- NVIDIA GPUs
# NVIDIA does not expose the useful counters in sysfs; nvidia-smi is the source.
# Three persistent nvidia-smi processes feed every NVIDIA card at once (see _NvidiaFeed).
_NVSMI_FIELDS = ("index", "name", "utilization.gpu", "utilization.memory",
                 "memory.used", "memory.total", "temperature.gpu", "power.draw",
                 "power.limit", "clocks.sm", "clocks.mem")

# Per-process memory, which fdinfo does not carry on NVIDIA. The DRM readers below look
# for drm-memory-vram / drm-memory-gtt, which amdgpu and i915 export and the NVIDIA driver
# does not, so every CUDA process reads as a dash. Across three CUDA servers there were 21
# open file descriptors to the nvidia device and no fdinfo file carrying the counter, while
# nvidia-smi reported real figures for the same pids. It is the compute-apps stream of
# _NvidiaFeed (`--query-compute-apps … -l 1`, one line per process per second, nothing at
# all while no process exists). Keyed by pid string, which is what walking /proc gives, and
# stamped, because a value from a process that has since exited is not a current value.
_NV_PROC_MEM = {}


def _nv_bus(s):
    """nvidia-smi writes a bus id with a 32-bit domain and upper-case hex
    ("00000000:1B:00.0"); sysfs names the same device "0000:1b:00.0". One spelling, the
    sysfs one, so the driver's tables can be joined to the card discovered under
    /sys/class/drm. Anything that is not a bus id comes back lower-cased and stripped."""
    m = re.match(r"([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])$",
                 (s or "").strip().lower())
    if not m:
        return (s or "").strip().lower()
    return f"{int(m.group(1), 16):04x}:{m.group(2)}:{m.group(3)}.{m.group(4)}"


# What nvidia-smi prints for "no such thing on this card". A field holding one of these
# is left ABSENT rather than set to a placeholder string, so it renders as nothing and not
# as a number: the P100 answers N/A for fan_speed (it is passively cooled) and the 2060
# for every ECC counter (it has no ECC), and both are absences, not zeros. The bracketed
# spellings are the ones nvidia-smi uses in csv output; the XML on this driver uses the
# bare ones. Both are accepted because they state the same fact.
_NV_NA = ("", "N/A", "[N/A]", "Not Supported", "[Not Supported]")


def _nv_num(text):
    """The number at the front of "38 C", "25.53 W", "16384 MiB", "683 KB/s", "0 %",
    "4991356120 us" or "16x": int when written as one, float otherwise, None when the
    text does not start with a number."""
    m = re.match(r"\s*(-?\d+)(\.\d+)?", text or "")
    if not m:
        return None
    return float(m.group(0)) if m.group(2) else int(m.group(1))


# Field maps for `nvidia-smi -q -x`: key in the sample <- element path under <gpu>. The
# names follow the XML of driver 580.173.02 as captured on this machine; the DTD the
# output names (nvsmi_device_v13.dtd) is not shipped with the driver here, so the two
# older spellings handled below (clocks_throttle_reasons, power_readings) come from
# nvidia-smi output seen on earlier drivers and are NOT verified against a DTD.
#
# Deliberately not read, and why:
#   supported_clocks: hundreds of (mem, graphics) pairs, a table and not a reading;
#   processes / accounted_processes: per-process memory already arrives through
#     --query-compute-apps (see _NV_PROC_MEM) and is keyed by pid there;
#   mig_mode, mig_devices, fabric, platformInfo, power_smoothing, power_profiles,
#     module_power_readings, gpu_memory_power_readings, cc_protected_memory_usage,
#     dram_encryption_mode, c2c_mode, ibmnpu, capabilities, inforom_bbx_flush,
#     gpu_operation_mode, driver_model, addressing_mode, sparse_operation_mode,
#     supported_gpu_target_temp, deferred_clocks, clock_policy, max_customer_boost_clocks:
#     N/A on these Pascal and Turing cards, or a capability and not a monitoring value;
#   pdi, board_id, minor_number, gpu_recovery_action, display_mode (deprecated by the
#     driver itself, it prints "Requested functionality has been deprecated"): identifiers
#     or constants with nothing to watch.
# serial and uuid ARE read, for --probe, and never reach the panel: see _nv_detail_rows.
_NV_XML_TEXT = {
    "product": "product_name", "brand": "product_brand", "arch": "product_architecture",
    "serial": "serial", "uuid": "uuid", "vbios": "vbios_version",
    "board_part": "board_part_number", "gpu_part": "gpu_part_number",
    "inforom_img": "inforom_version/img_version", "gsp_fw": "gsp_firmware_version",
    "persistence": "persistence_mode", "display_active": "display_active",
    "display_attached": "display_attached", "compute_mode": "compute_mode",
    "ecc_mode": "ecc_mode/current_ecc", "ecc_pending": "ecc_mode/pending_ecc",
    "accounting": "accounting_mode", "virt": "gpu_virtualization_mode/virtualization_mode",
    "pstate": "performance_state",
    "retired_pending": "retired_pages/pending_retirement",
    # text: on the cards here it is N/A. Newer drivers on cards with row remapping print
    # it as a block of child elements, which findtext reads as empty and skips; not
    # verified on such a card, so only the plain form is claimed.
    "remapped_rows": "remapped_rows",
}
_NV_XML_NUM = {
    "nv_gen_max": "pci/pci_gpu_link_info/pcie_gen/max_link_gen",
    "nv_gen_cur": "pci/pci_gpu_link_info/pcie_gen/current_link_gen",
    "nv_gen_dev_cur": "pci/pci_gpu_link_info/pcie_gen/device_current_link_gen",
    "nv_gen_dev_max": "pci/pci_gpu_link_info/pcie_gen/max_device_link_gen",
    "nv_gen_host_max": "pci/pci_gpu_link_info/pcie_gen/max_host_link_gen",
    "nv_width_max": "pci/pci_gpu_link_info/link_widths/max_link_width",
    "nv_width_cur": "pci/pci_gpu_link_info/link_widths/current_link_width",
    "pcie_replay": "pci/replay_counter", "pcie_replay_rollover": "pci/replay_rollover_counter",
    "pcie_tx_kbs": "pci/tx_util", "pcie_rx_kbs": "pci/rx_util",
    "fan_pct": "fan_speed",
    "vram_reserved": "fb_memory_usage/reserved", "vram_free": "fb_memory_usage/free",
    "bar1_used": "bar1_memory_usage/used", "bar1_total": "bar1_memory_usage/total",
    "enc_util": "utilization/encoder_util", "dec_util": "utilization/decoder_util",
    "jpeg_util": "utilization/jpeg_util", "ofa_util": "utilization/ofa_util",
    "enc_sessions": "encoder_stats/session_count", "fbc_sessions": "fbc_stats/session_count",
    "ecc_vol_sbe": "ecc_errors/volatile/single_bit/total",
    "ecc_vol_dbe": "ecc_errors/volatile/double_bit/total",
    "ecc_agg_sbe": "ecc_errors/aggregate/single_bit/total",
    "ecc_agg_dbe": "ecc_errors/aggregate/double_bit/total",
    "retired_sbe": "retired_pages/multiple_single_bit_retirement/retired_count",
    "retired_dbe": "retired_pages/double_bit_retirement/retired_count",
    "temp_mem": "temperature/memory_temp", "temp_tlimit": "temperature/gpu_temp_tlimit",
    "temp_shutdown": "temperature/gpu_temp_max_threshold",
    "temp_slowdown": "temperature/gpu_temp_slow_threshold",
    "temp_max_op": "temperature/gpu_temp_max_gpu_threshold",
    "temp_target": "temperature/gpu_target_temperature",
    "temp_mem_max": "temperature/gpu_temp_max_mem_threshold",
    "clk_video": "clocks/video_clock",
    "clk_app_gfx": "applications_clocks/graphics_clock",
    "clk_app_mem": "applications_clocks/mem_clock",
    "clk_max_gfx": "max_clocks/graphics_clock", "clk_max_mem": "max_clocks/mem_clock",
    "clk_max_video": "max_clocks/video_clock",
}
# under <gpu_power_readings> on this driver, <power_readings> on older ones (unverified)
_NV_XML_POWER = {
    "power_instant": "instant_power_draw", "power_average": "average_power_draw",
    "power_enforced": "current_power_limit", "power_requested": "requested_power_limit",
    "power_default": "default_power_limit", "power_min": "min_power_limit",
    "power_max": "max_power_limit",
}


def parse_nvidia_smi_xml(text):
    """`nvidia-smi -q -x` to {"driver_version", "cuda_version", "gpus": {bus: fields}},
    bus in the sysfs spelling (see _nv_bus) and fields named by the three maps above.
    Pure: no subprocess, no clock, so it can be run on a captured file. Unparseable
    input gives the empty shape rather than an exception, for the same reason the csv
    reader swallows a failed query: one bad answer must not take the feed down.
    Clock event reasons: the names under <clocks_event_reasons> whose value is "Active",
    with the clocks_event_reason_ prefix stripped (clocks_throttle_reason_ on older
    drivers), and the cumulative microseconds under <clocks_event_reasons_counters>
    keyed the same way. Note the counters block spells three reasons differently from the
    reasons block (sw_therm_slowdown, hw_therm_slowdown, hw_power_brake): both spellings
    are in _NV_CLOCK_REASON_NAMES."""
    import xml.etree.ElementTree as ET
    out = {"driver_version": None, "cuda_version": None, "gpus": {}}
    try:
        root = ET.fromstring(text or "")
    except ET.ParseError:
        return out

    def val(node, path):
        v = node.findtext(path)
        v = v.strip() if v is not None else ""
        return None if v in _NV_NA else v

    out["driver_version"] = val(root, "driver_version")
    out["cuda_version"] = val(root, "cuda_version")
    for gpu in root.findall("gpu"):
        f = {}
        for key, path in _NV_XML_TEXT.items():
            v = val(gpu, path)
            if v is not None:
                f[key] = v
        power = gpu.find("gpu_power_readings")
        if power is None:
            power = gpu.find("power_readings")
        numeric = [(gpu, _NV_XML_NUM)] + ([(power, _NV_XML_POWER)] if power is not None else [])
        for node, table in numeric:
            for key, path in table.items():
                v = val(node, path)
                n = _nv_num(v) if v is not None else None
                if n is not None:
                    f[key] = n
        reasons = gpu.find("clocks_event_reasons")
        if reasons is None:
            reasons = gpu.find("clocks_throttle_reasons")
        if reasons is not None:
            f["clock_reasons_active"] = [
                re.sub(r"^clocks_(?:event|throttle)_reason_", "", c.tag)
                for c in reasons if (c.text or "").strip() == "Active"]
        counters = gpu.find("clocks_event_reasons_counters")
        if counters is not None:
            us = {}
            for c in counters:
                v = (c.text or "").strip()
                n = _nv_num(v) if v not in _NV_NA else None
                if n is not None:
                    us[re.sub(r"^clocks_event_reasons_counters_", "", c.tag)] = n
            f["clock_reason_us"] = us
        bus = _nv_bus(gpu.get("id") or val(gpu, "pci/pci_bus_id") or "")
        if bus:
            out["gpus"][bus] = f
    return out


# The clock event reasons nvidia-smi reports, as nvml.h names them
# (nvmlClocksEventReasonSwPowerCap and friends in the CUDA 12.8 header; the driver's
# older /usr/include/nvml.h spells them nvmlClocksThrottleReason*), with the wording the
# panel uses. The three extra spellings at the end are how the counters block names the
# same reasons.
_NV_CLOCK_REASON_NAMES = {
    "sw_power_cap": "sw power cap", "hw_slowdown": "hw slowdown",
    "hw_thermal_slowdown": "hw thermal", "sw_thermal_slowdown": "sw thermal",
    "hw_power_brake_slowdown": "hw power brake", "sync_boost": "sync boost",
    "gpu_idle": "idle", "applications_clocks_setting": "app clock setting",
    "display_clocks_setting": "display clock setting",
    "sw_therm_slowdown": "sw thermal", "hw_therm_slowdown": "hw thermal",
    "hw_power_brake": "hw power brake",
}
# Reasons that mean the card is being held back by a limit it is hitting, whatever the
# wattage: a hardware slowdown is the board pulling the clocks down on its own
# (nvidia-smi --help-query-gpu, hw_slowdown: "HW Slowdown (reducing the core clocks by a
# factor of 2 or more) is engaged. This is an indicator of: HW Thermal Slowdown:
# temperature being too high; HW Power Brake Slowdown: External Power Brake Assertion is
# triggered (e.g. by the system power supply); Power draw is too high and Fast Trigger
# protection is reducing the clocks", and it notes the flag "May be also reported during
# PState or clock change").
_NV_ALWAYS_THROTTLE = ("hw_slowdown", "hw_thermal_slowdown", "sw_thermal_slowdown",
                       "hw_power_brake_slowdown")
# sw_power_cap is a throttle only when the card is actually near its limit. The reason
# reads Active whenever the power-scaling algorithm has a hand on the clocks (nvidia-smi
# --help-query-gpu: "SW Power Scaling algorithm is reducing the clocks below requested
# clocks because the GPU is consuming too much power"), and MEASURED on this machine that
# includes the RTX 2060 sitting in P8 at 2 W of a 190 W limit with a display attached:
# that is the P-state policy parking the card, not a power limit being hit, and drawing
# "⚠ throttling" over it would teach the reader to ignore the row. 0.80 is the same
# fraction the power panel already colours WARN, so the two agree on what "near" means.
_NV_POWER_CAP_NEAR = 0.80


def nv_classify_clock_reasons(active, power, limit):
    """(throttle, neutral): the active reasons that mean the card is limited, and the ones
    that merely explain why the clocks are where they are (idle, an application clock
    setting, sync boost). A name this file does not know goes to neutral verbatim, so a
    reason added by a newer driver is shown and never alarmed on."""
    throttle, neutral = [], []
    for name in active or []:
        human = _NV_CLOCK_REASON_NAMES.get(name, name)
        if name in _NV_ALWAYS_THROTTLE:
            throttle.append(human)
        elif (name == "sw_power_cap" and power is not None and limit
              and power >= _NV_POWER_CAP_NEAR * limit):
            throttle.append(human)
        else:
            neutral.append(human)
    return throttle, neutral


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


# The NVIDIA feed runs nvidia-smi as PERSISTENT processes instead of spawning it for every
# reading (round 8a, variant V3; Fabrizio: «vai con V3»). MEASURED by the auditor on this
# machine, GPUs idle, 300 s per variant, N = 150 two-second windows: spawning per loop (core
# csv and compute apps every second, index and `-q -x` every third loop) cost 136 spawns and
# 4.79 s of nvidia-smi CPU per minute, almost all of it system time in the driver; the same
# four queries as three `-l`/`-lms` loops plus the index table once cost 0.8 spawns and
# 2.09 s per minute (−56 %, Mann-Whitney z −12.4) and refreshed the core rows every 1.01 s
# and the detail every 3.19 s instead of 1.09 / 3.53 s, with the same fields and the same
# parser. What a persistent process changes is the LIFECYCLE, which is what the class below
# is about:
#   * a child that dies is restarted after _NV_BACKOFF_S[failures]: 1, 2, 4, 8, 16, then 30 s
#     for ever, the count starting over once a child has lived _NV_STABLE_S. In a total crash
#     loop that is 6 launches per stream in the first minute and 2 per minute after, so a
#     driver that refuses can never turn the feed into a spawn storm;
#   * output is split into whole records before parsing, a csv line or one `-q -x` document
#     cut at </nvidia_smi_log>, because a pipe hands over chunks, not answers, and a document
#     that has not closed is never parsed;
#   * silence is not death. `--query-compute-apps -l 1` prints one line per compute process
#     per iteration and may print NOTHING while no process exists (declared, not verified:
#     RustDesk always held a context during the measurement). A stream is alive while its
#     process is, and the data it produced is aged by its readers exactly as before (`ts`
#     under 6 s in NvidiaGpu.sample, detail_ts within _SANE_TTL, _NV_PROC_MEM 6 s in
#     _nv_proc_fill and dropped after 60 s on every pass), so a number from a stream that
#     stalled never reads as live;
#   * every exit path ends in _shutdown: terminate, kill after _NV_GRACE_S, reap. `stop = True`
#     (what probe, the TUI and --line set) does it synchronously in the caller's thread;
#     run()'s finally does it after an exception in a reader; an atexit handler does it on the
#     paths that never set stop (an uncaught exception or a Ctrl-C in the TUI). The children
#     stay in this process group, so a terminal Ctrl-C reaches them as well and the loops end
#     on it ("Probe until Ctrl+C", nvidia-smi --help). What the standard library cannot
#     guarantee: SIGKILL (or SIGTERM under Python's default action) on llamagputop runs no
#     Python at all. Then the pipes' read ends close with the process and each child's next
#     write fails with EPIPE and SIGPIPE, which subprocess restored to its default action in
#     the child (restore_signals), so a loop should end within one of its own intervals, if
#     nvidia-smi keeps that disposition. prctl PDEATHSIG was rejected: it needs preexec_fn,
#     which the subprocess documentation marks unsafe with threads in the parent, and this
#     feed is one thread beside _LlamaFeed.
_NV_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)   # restart delays after consecutive deaths
_NV_STABLE_S = 60.0       # a child that lived this long before dying starts the delays over
_NV_GRACE_S = 1.0         # after terminate, before kill
_NV_BUFFER_MAX = 8 << 20  # bytes a stream may hold without a record boundary before it is dropped
_NV_XML_END = b"</nvidia_smi_log>"


def _nv_spawn(args):
    """One nvidia-smi child on a pipe. subprocess.Popen is looked up at call time, not bound,
    so the tests' tripwire on that name sees every launch; _NvidiaFeed.spawn is the factory
    the tests replace with fakes."""
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)


class _NvSmiStream:
    """One of the feed's nvidia-smi invocations: its argv, its child while one runs, the bytes
    the child sent that do not yet make a whole record, and the restart bookkeeping."""
    def __init__(self, args, persistent, xml=False):
        self.args = list(args)
        self.persistent = persistent
        self.xml = xml
        self.handler = None
        self.proc = None
        self.buf = b""
        self.failures = 0
        self.next_at = 0.0
        self.started_at = 0.0
        self.done = False         # a one-shot that has answered; only a core launch re-arms it

    def records(self, data):
        """The complete records once `data` is appended: csv lines split on newline, or `-q -x`
        documents cut after </nvidia_smi_log> and left-trimmed (the parser refuses a leading
        newline before the XML declaration, and `-l` prints one between documents). A partial
        tail waits for the next chunk; a tail that grows past _NV_BUFFER_MAX without a boundary
        is a broken stream and is dropped."""
        self.buf += data
        out = []
        if self.xml:
            while True:
                i = self.buf.find(_NV_XML_END)
                if i < 0:
                    break
                end = i + len(_NV_XML_END)
                out.append(self.buf[:end].lstrip())
                self.buf = self.buf[end:]
        else:
            *out, self.buf = self.buf.split(b"\n")
        if len(self.buf) > _NV_BUFFER_MAX:
            self.buf = b""
        return [x.decode("utf-8", "replace") for x in out]


class _NvidiaFeed(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.by_index = {}
        # recorded at construction so a panel can say which program is missing instead of
        # printing a bare dash. `which` is cheap and the answer cannot change mid-run.
        self.tool_ok = which("nvidia-smi")
        # everything `nvidia-smi -q -x` reports, keyed by the sysfs bus id, replaced by every
        # document of the detail stream (one every 3 s); the two version strings ride beside it
        self.detail_by_bus = {}
        self.driver = {}
        self.detail_ts = 0.0
        # the driver's own index for each bus id, so a card is matched by address and not
        # by the order two enumerations happen to share (see NvidiaGpu.sample)
        self.index_by_bus = {}
        self.spawn = _nv_spawn
        self._lock = threading.Lock()     # a launch and a shutdown never overlap
        self._closed = False              # shutdown has begun: nothing is launched from here on
        self._stopping = False
        self._registered = False          # the atexit handler, once, at the first launch
        self._sel = None                  # the feed thread's selector over the children's pipes
        self._index_rows = {}
        query = "--query-gpu=" + ",".join(_NVSMI_FIELDS)
        self._core = _NvSmiStream(["nvidia-smi", query, "--format=csv,noheader,nounits",
                                   "-lms", "1000"], persistent=True)
        self._index = _NvSmiStream(["nvidia-smi", "--query-gpu=index,pci.bus_id",
                                    "--format=csv,noheader"], persistent=False)
        self._apps = _NvSmiStream(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                   "--format=csv,noheader,nounits", "-l", "1"], persistent=True)
        self._detail = _NvSmiStream(["nvidia-smi", "-q", "-x", "-l", "3"], persistent=True, xml=True)
        self._core.handler, self._index.handler = self._core_line, self._index_line
        self._apps.handler, self._detail.handler = self._apps_line, self._detail_doc
        # the index right after the core: a core launch re-arms it and the same pass starts it
        self._streams = (self._core, self._index, self._apps, self._detail)

    @property
    def stop(self):
        return self._stopping

    @stop.setter
    def stop(self, value):
        """The flag every caller sets on exit. True also ends the children NOW, in the caller's
        thread: the feed thread is a daemon and may never run again once main() returns, so a
        flag it would notice later is no guarantee against orphans."""
        self._stopping = bool(value)
        if value:
            self._shutdown()

    def run(self):
        if not self.tool_ok:
            return
        try:
            while not self._stopping:
                self._service(0.5)
        finally:
            self._shutdown()
            self._release()

    def _service(self, timeout):
        """One pass: launch what is due, read what the children sent (whole records only),
        notice deaths, sweep the per-process table. `timeout` is how long to wait for data."""
        if self._sel is None:
            self._sel = selectors.DefaultSelector()
        now = time.monotonic()
        with self._lock:
            if not self._closed:
                for s in self._streams:
                    if s.proc is None and not s.done and now >= s.next_at:
                        self._launch(s, now)
        ready = self._sel.select(timeout)
        now = time.monotonic()
        for key, _ in ready:
            self._read(key.data, now)
        for s in self._streams:
            if s.proc is not None and s.proc.poll() is not None:
                self._ended(s, now)
        # drop entries for processes that have exited, or the dict grows for the whole run.
        # The panel already ignores them by age; this only bounds memory. On every pass and
        # not after an answer, because the apps stream may stay silent for as long as it likes
        for pid in [k for k, (_m, t) in _NV_PROC_MEM.items() if now - t > 60]:
            _NV_PROC_MEM.pop(pid, None)

    def _launch(self, s, now):
        """Start one child, under the lock. A launch that fails is a death: same backoff."""
        s.started_at = now
        s.buf = b""
        if s is self._index:
            self._index_rows = {}
        try:
            s.proc = self.spawn(list(s.args))
        except OSError:
            s.proc = None
            self._schedule(s, now)
            return
        os.set_blocking(s.proc.stdout.fileno(), False)
        self._sel.register(s.proc.stdout, selectors.EVENT_READ, s)
        if not self._registered:
            atexit.register(self._shutdown)
            self._registered = True
        if s is self._core:
            # the table the core rows are joined with: static, read again only with them
            self._index.done = False
            self._index.next_at = now

    def _read(self, s, now):
        try:
            data = os.read(s.proc.stdout.fileno(), 65536)
        except BlockingIOError:
            return
        if data:
            self._consume(s, data, now)
        else:
            self._ended(s, now)

    def _consume(self, s, data, now):
        for rec in s.records(data):
            try:
                s.handler(rec, now)
            except ValueError:        # one malformed record; the stream goes on, as before
                continue

    def _ended(self, s, now):
        """The child exited or closed its pipe: drain what it left (whole records only, a
        partial tail is never parsed), reap it, then schedule the restart or retire a one-shot."""
        proc = s.proc
        fd = proc.stdout.fileno()
        for _ in range(64):
            try:
                data = os.read(fd, 65536)
            except OSError:           # nothing more, or nothing more for now: it is dead
                break
            if not data:
                break
            self._consume(s, data, now)
        try:
            self._sel.unregister(proc.stdout)
        except (KeyError, ValueError):
            pass
        proc.stdout.close()
        self._reap(proc)
        s.proc = None
        s.buf = b""
        if s is self._index and self._index_rows:
            self.index_by_bus = self._index_rows
        if not s.persistent:
            s.done = True
        elif not self._closed:
            self._schedule(s, now)

    def _schedule(self, s, now):
        if now - s.started_at >= _NV_STABLE_S:
            s.failures = 0
        s.next_at = now + _NV_BACKOFF_S[min(s.failures, len(_NV_BACKOFF_S) - 1)]
        s.failures += 1

    def _reap(self, proc):
        """One child: terminate if still alive, then wait, kill after the grace, wait."""
        if proc.poll() is None:
            proc.terminate()
        self._wait_or_kill(proc, time.monotonic() + _NV_GRACE_S)

    def _wait_or_kill(self, proc, deadline):
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=_NV_GRACE_S)
            except subprocess.TimeoutExpired:
                # SIGKILL did not end it within the grace: an uninterruptible wait in the
                # driver. It dies when that returns and is reaped by init once we are gone.
                pass

    def _shutdown(self):
        """Every exit path ends here: no child is launched from now on, and every child that
        exists is terminated, killed after the grace if it did not go, and reaped. Idempotent,
        callable from any thread: the stop setter, atexit, run()'s finally."""
        with self._lock:
            self._closed = True
            procs = [s.proc for s in self._streams if s.proc is not None]
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + _NV_GRACE_S
        for p in procs:
            self._wait_or_kill(p, deadline)

    def _release(self):
        """The feed thread's own resources: the pipes it read and the selector over them."""
        for s in self._streams:
            if s.proc is not None:
                try:
                    self._sel.unregister(s.proc.stdout)
                except (KeyError, ValueError):
                    pass
                s.proc.stdout.close()
                s.proc = None
        if self._sel is not None:
            self._sel.close()
            self._sel = None

    # ---- the four readers, one record at a time: the pre-V3 parsing, the timestamp handed in
    def _core_line(self, line, now):
        c = [x.strip() for x in line.split(",")]
        if len(c) < len(_NVSMI_FIELDS):
            return

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        self.by_index[int(c[0])] = {
            "name": c[1], "util": num(c[2]), "mem_util": num(c[3]),
            "vram_used": num(c[4]), "vram_total": num(c[5]),
            "temp_main": num(c[6]), "power": num(c[7]), "power_cap": num(c[8]),
            "sclk": num(c[9]), "mclk": num(c[10]), "ts": now}

    def _apps_line(self, line, now):
        # memory per compute process: a different table with a different key (pid, not card
        # index), its own stream so one failing cannot take the other down with it
        c = [x.strip() for x in line.split(",")]
        if len(c) < 2:
            return
        try:
            _NV_PROC_MEM[c[0]] = (int(float(c[1])), now)
        except ValueError:
            if "N/A" in c[1]:
                _NV_PROC_MEM[c[0]] = (None, now)

    def _index_line(self, line, now):
        c = [x.strip() for x in line.split(",")]
        if len(c) >= 2 and c[0].isdigit():
            self._index_rows[_nv_bus(c[1])] = int(c[0])

    def _detail_doc(self, doc, now):
        # everything else the driver knows, its own stream for the reason the apps stream
        # gives: one invalid field name fails a whole --query-gpu call, so the core query
        # stays byte-identical and the extra data comes from a process whose failure costs
        # only itself
        q = parse_nvidia_smi_xml(doc)
        if q["gpus"]:
            self.detail_by_bus = q["gpus"]
            self.driver = {k: q[k] for k in ("driver_version", "cuda_version") if q.get(k)}
            self.detail_ts = now


# Session baseline of the PCIe replay and replay-rollover counters per bus id, taken on
# the first detail sample of a run: the counters are cumulative and the alarm is what
# arrived since the session began (see _pcie_rows).
_NV_REPLAY_BASE = {}


class NvidiaGpu:
    vendor = "NVIDIA"

    def __init__(self, name, index, feed, dev=None):
        self.name = name
        self.index = index
        self.feed = feed
        # the sysfs device node, for the PCIe link the driver does not report the way
        # the other two vendors' readers do; None keeps the old construction working
        self.dev = dev

    def sample(self):
        d = {"vendor": "NVIDIA", "name": self.name}
        # Which row of the nvidia-smi table is this card. `index` is the position among
        # the NVIDIA nodes under /sys/class/drm, and nvidia-smi numbers its cards by its
        # own enumeration; the two agree here (card1/2/3 are index 0/1/2) and there is
        # no rule that they must. The driver publishes index and bus id side by side
        # (--query-gpu=index,pci.bus_id), so the address, which both sides know, is the
        # join, and the position is only the fallback while that table has not arrived.
        idx = getattr(self.feed, "index_by_bus", {}).get(getattr(self, "pci_addr", None))
        if idx is None:
            idx = self.index
        f = self.feed.by_index.get(idx)
        if f and time.monotonic() - f.get("ts", 0) < 6:
            d.update({k: f.get(k) for k in
                      ("util", "mem_util", "vram_used", "vram_total", "temp_main",
                       "power", "power_cap", "sclk", "mclk")})
            d["temp"] = {"gpu": d.get("temp_main")} if d.get("temp_main") is not None else {}
        else:
            # NVIDIA exposes none of this in sysfs, so nvidia-smi is the only source there
            # is and its absence takes out every field at once. Saying so beats five dashes
            # that each leave the reader to guess whether the card is idle or unread.
            why = ("needs nvidia-smi" if not self.feed.tool_ok
                   else "nvidia-smi returned nothing for this card")
            d["needs"] = {f: why for f in ("util", "vram", "temp", "power", "clocks")}
        # The link comes from sysfs, as it does for AMD and Intel: the nvidia module does
        # not publish clocks or temperatures there, but the PCI core publishes the link
        # attributes for every device whatever its driver, so this needs no nvidia-smi.
        d["pcie"] = pcie_chain(self.dev) if self.dev else None
        # The detail table, when the feed has a fresh one for this bus id. Same age rule
        # as sane(): the feed refreshes it every ~3 s, so a table older than _SANE_TTL
        # means the query has been failing and its numbers are not this tick's.
        # setdefault, never update: the nine core fields came from the csv query a moment
        # ago and are the live ones; a value in the detail table for the same key would be
        # up to three seconds older, and two sources for one key is how a number flickers.
        bus = getattr(self, "pci_addr", None)
        detail = self.feed.detail_by_bus.get(bus) if bus else None
        if detail and time.monotonic() - self.feed.detail_ts <= _SANE_TTL:
            for k, v in detail.items():
                d.setdefault(k, v)
            drv = self.feed.driver
            if drv.get("driver_version"):
                d["nv_driver"] = drv["driver_version"]
            if drv.get("cuda_version"):
                d["cuda"] = drv["cuda_version"]
            # the driver's own thresholds colour the temperature (round 7c, option B,
            # Fabrizio: «4 va bene B come consigli»): CRIT from the slowdown point, where
            # the card is already cutting its clocks, else from the shutdown point as
            # before; WARN from the "max operating" point when it lies below the CRIT
            # point, else _temp_attr_th's fixed 80 °C. Effective here, every integer
            # 60–95 °C through the real colour function (tested): RTX 2060 (max op 88,
            # slowdown 90, shutdown 93) OK < 80, WARN 80–89, CRIT ≥ 90; Tesla P100
            # (slowdown 82, shutdown 85) OK < 80, WARN 80–81, CRIT ≥ 82. Until round 7c
            # CRIT sat at the shutdown point (93 / 85). A card reporting neither point
            # keeps the earlier shape (WARN from max op); the AMD hwmon crit/emergency
            # mapping is not part of the decision and is untouched.
            crit_at = detail.get("temp_slowdown")
            if crit_at is None:
                crit_at = detail.get("temp_shutdown")
            max_op = detail.get("temp_max_op")
            warn_at = max_op if max_op is not None and (crit_at is None or max_op < crit_at) else 0
            if crit_at is not None or warn_at:
                d["temp_crit"] = {"gpu": (warn_at, crit_at or 0)}
            if detail.get("temp_mem") is not None:
                d.setdefault("temp", {})["mem"] = detail["temp_mem"]
            d["throttle"], d["clock_reasons"] = nv_classify_clock_reasons(
                detail.get("clock_reasons_active"), d.get("power"),
                d.get("power_enforced") or d.get("power_cap"))
            # PCIe throughput has a physical bound, and the driver does not respect it.
            # MEASURED on both P100 here, recurring every few seconds: tx or rx of
            # 209715151 KB/s, about 200 GB/s on a link that carries 8, and `nvidia-smi
            # dmon -s t` prints the same figure (205909 / 209715 MB/s), so it is NVML and
            # not this parser; plausible values seen in between were 2103906, 25341, 48
            # and 0 KB/s. Whatever the driver's reason, the link's best possible
            # bandwidth is a fact of the hardware: the maxima both ends support (not the
            # current negotiation, which rises mid-window under load), times the encoding
            # efficiency. sane() drops what exceeds it and holds the last good value for
            # up to _SANE_TTL, as every other reading in this file is held. Without a
            # known negotiation the figures pass through as reported.
            link = d.get("pcie") or {}
            if link.get("card_max_gts") is not None:
                gts = min(g for g in (link["card_max_gts"], link.get("slot_max_gts")) if g)
                width = min(w for w in (link["card_max_width"], link.get("slot_max_width")) if w)
                bound = gts * width * _PCIE_EFF.get(gts, 128 / 130) / 8 * 1e6
                for k in ("pcie_tx_kbs", "pcie_rx_kbs"):
                    if d.get(k) is not None:
                        d[k] = sane(f"{bus}.{k}", d[k], 0, bound)
            # replay counters as a session delta, the same rule as pcie_aer: a total that
            # went down was reset (driver reload, a GPU reset), so the baseline is taken
            # again rather than reporting a negative delta
            if d.get("pcie_replay") is not None:
                cur = (d["pcie_replay"], d.get("pcie_replay_rollover") or 0)
                base = _NV_REPLAY_BASE.get(bus)
                if base is None or base[0] > cur[0] or base[1] > cur[1]:
                    _NV_REPLAY_BASE[bus] = base = cur
                d["pcie_replay_session"] = cur[0] - base[0]
                d["pcie_replay_rollover_session"] = cur[1] - base[1]
        return d


# ------------------------------------------------------------- GPU discovery
def discover_gpus():
    """Enumerate every render node under /sys/class/drm and build the right kind of
    monitor for each driver. NVIDIA cards are correlated to nvidia-smi by index."""
    gpus, feeds = [], {}
    intel_vram = None
    nvidia_feed = None
    nvidia_seen = 0
    for card in sorted(glob.glob(f"{_DRM_ROOT}/card[0-9]*")):
        if "-" in os.path.basename(card):        # connector, not a card
            continue
        dev = f"{card}/device"
        drv_link = f"{dev}/driver"
        if not os.path.exists(drv_link):
            continue
        driver = os.path.basename(os.path.realpath(drv_link))
        pci_addr = os.path.basename(os.path.realpath(dev))
        vendor_id = read(f"{dev}/vendor", "")
        name = gpu_product_name(dev, vendor_id)
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
            g = NvidiaGpu(name, nvidia_seen, nvidia_feed, dev=dev)
            nvidia_seen += 1
        if g is not None:
            g.pci_addr = pci_addr
            gpus.append(g)
    # Fall back to nvidia-smi if DRM reported no NVIDIA cards (e.g. WSL2, Hyper-V GPU-PV)
    if nvidia_seen == 0 and which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,pci.bus_id", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3).stdout
            for s_line in out.strip().splitlines():
                parts = [p.strip() for p in s_line.split(",")]
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                    except ValueError:
                        continue
                    name = parts[1]
                    pci_addr = parts[2] if len(parts) > 2 else ""
                    if nvidia_feed is None:
                        nvidia_feed = _NvidiaFeed()
                        feeds["nvidia"] = nvidia_feed
                    g = NvidiaGpu(name, idx, nvidia_feed)
                    g.pci_addr = pci_addr
                    gpus.append(g)
                    nvidia_seen += 1
        except Exception:
            pass
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


# k10temp publishes temperature labels only on Zen (k10temp.c, in the is_visible callback
# for hwmon_temp_label: "Show temperature labels only on Zen CPUs"), so on a Family 10h-16h
# part the channel is a bare temp1_input. The driver's own label table names channel 0
# "Tctl" on every family (k10temp.c, the k10temp_temp_label[] array), so that is the name
# used here rather than inventing one. Channel 1 would be Tdie, which pre-Zen parts do
# not have; a k10temp channel this table does not name keeps its channel name.
_K10TEMP_UNLABELED = {"temp1": "Tctl"}


def hwmon_unlabeled_temps(hw):
    """{label: channel} for every temp*_input under `hw` that has no temp*_label beside
    it. The label is k10temp's own name for the channel when the chip is k10temp, else
    the channel name (e.g. "temp2"), so a chip this file has never seen still shows its
    sensors under an honest, if terse, name. {} without a hwmon node."""
    out = {}
    if not hw:
        return out
    chip = read(f"{hw}/name")
    for f in sorted(glob.glob(f"{hw}/temp*_input")):
        ch = os.path.basename(f).replace("_input", "")
        if os.path.exists(f"{hw}/{ch}_label"):
            continue
        out[_K10TEMP_UNLABELED.get(ch, ch) if chip == "k10temp" else ch] = ch
    return out


def hwmon_temp_limits(hw, channels):
    """{label: (max_c, crit_c)} in whole degrees from the <ch>_max and <ch>_crit files
    the driver declares, 0 standing for a file that is not there, and only for labels
    that declare at least one. On this machine's FX-8350 k10temp declares temp1_max
    70000 and temp1_crit 80000 (k10temp.rst: "The maximum value for Tctl is available in
    the file temp1_max. If the BIOS has enabled hardware temperature control, the
    threshold at which the processor will throttle itself to avoid damage is available in
    temp1_crit and temp1_crit_hyst"), which beats the fixed 80/90 °C the panel falls
    back to, and matters twice over on this part because its scale is not physical (see
    CPU_TEMPS_RELATIVE)."""
    out = {}
    if not hw:
        return out
    for label, ch in channels.items():
        mx = read_int(f"{hw}/{ch}_max") // 1000
        crit = read_int(f"{hw}/{ch}_crit") // 1000
        if mx or crit:
            out[label] = (mx, crit)
    return out


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

# The loop above finds nothing on a chip that publishes no labels, so on this machine's
# FX-8350 (k10temp, Family 15h) CPU_TEMPS stayed empty and the CPU panel drew no
# temperature at all, while the T_CPU fallback a few lines up had already found temp1 for
# the summary figure: the same sensor was read for one field and dropped for the other.
# The label-less channels join the table under the driver's own name (setdefault: a
# labelled channel of the same name wins). The per-channel limits the driver declares
# are read once here, the same way AmdGpu reads its hwmon _crit/_emergency at
# construction; they colour the panel instead of the fixed 80/90 °C.
# CPU_TEMPS_RELATIVE names the channels whose scale is not a physical temperature: on
# k10temp before Family 17h "Tctl is a non-physical temperature on an arbitrary scale
# measured in degrees. It does _not_ represent an actual physical temperature like die or
# case temperature" (k10temp.rst, quoting the AMD manual); the panel marks them "rel" so
# a 72 beside a GPU's 72 is not read as the same thing. Zen parts label their channels,
# so nothing lands here for them, which is right: from Family 17h the driver "aims to
# compensate and report the real temperature" (same document).
CPU_TEMPS_RELATIVE = set()
if HW_CPU:
    _unlabeled = hwmon_unlabeled_temps(HW_CPU)
    for _lbl, _ch in _unlabeled.items():
        CPU_TEMPS.setdefault(_lbl, _ch)
    if read(f"{HW_CPU}/name") == "k10temp":
        CPU_TEMPS_RELATIVE = set(_unlabeled)
CPU_TEMP_LIMITS = hwmon_temp_limits(HW_CPU, CPU_TEMPS)

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
    # constants of the sensor, carried on every sample so the panel needs no second
    # source: which limits the driver declared, and which scales are not physical
    d["temp_limits"] = CPU_TEMP_LIMITS
    d["temp_relative"] = sorted(CPU_TEMPS_RELATIVE)
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
        # unconditional until now: the line ran whether or not the counter had been read
        # this tick, so an unreadable RAPL node froze the CPU wattage forever. Same age
        # check as the GPU path above and as sane(), from the timestamp already kept here.
        d["power"] = (_CPU_E["w"]
                      if time.monotonic() - _CPU_E["t"] <= _SANE_TTL else None)
    return d


# ---------------------------------------------------------- other hwmon chips
# Every hwmon chip the CPU and GPU readers do not already own: the Super I/O on the board
# (fans, voltages, case temperatures), a USB-C power source, a drive. The kernel's hwmon
# sysfs interface (Documentation/hwmon/sysfs-interface.rst, with the units per attribute
# in Documentation/ABI/testing/sysfs-class-hwmon) is uniform across drivers,
# <type><n>_input in fixed units per type, optional <ch>_label, _max, _crit and _alarm,
# so one reader covers chips this file has never heard of, and the panel shows whatever
# the box has instead of a list written for one machine.
#
# Chips skipped by name, with the reason:
#   fam15h_power: power1_input is CurrPwrWatts (fam15h_power.rst: "This driver provides
#     ProcessorPwrWatts and CurrPwrWatts: power1_crit (ProcessorPwrWatts), power1_input
#     (CurrPwrWatts)", and "CurrPwrWatts: Specifies in watts the current amount of power
#     being consumed by the processor"), which the driver computes in power1_input_show
#     from the TDP running-average accumulator and the TDP limit (fam15h_power.c,
#     REG_TDP_RUNNING_AVERAGE and REG_TDP_LIMIT3): a figure the processor derives, not a
#     meter. MEASURED at true idle on this FX-8350: idle n=15 (utilisation median 4.4 %)
#     median 109.5 W, 8-thread load n=20 median 123.5 W, after n=15 median 110.1 W,
#     Mann-Whitney U=0, p about 6e-7. So it does track load, and returns. What is NOT
#     verified is the absolute scale, and two explanations stand: this machine runs no
#     cpufreq and no cpuidle driver, so the CPU sits near 4.1 GHz with HLT-only idle and
#     a floor of 110 W could be real; or the estimate is compressed just under the
#     125.2 W TDP, with 14 W between idle and full load. N < 150 per condition, so the
#     numbers are indicative. It stays skipped by the owner's decision until the reading
#     has been checked against a wall or EPS meter, and the CPU panel does not add it
#     either. Showing it here under a "W" would be the same claim through a side door.
_HWMON_SKIP = {"fam15h_power"}

# type -> (divisor to the display unit, unit suffix, decimals). Units per
# Documentation/ABI/testing/sysfs-class-hwmon: tempY_input "millidegree Celsius",
# inY_input "millivolt", currY_input "milliampere", powerY_input "microWatt",
# energyY_input "microJoule", fanY_input "revolution/min (RPM)", humidityY_input
# "milli-percent (per cent mille, pcm)"; pwmY is an "Integer value in the range 0 to
# 255", "255 is max or 100%"
_HWMON_TYPES = {
    "temp": (1000.0, "°C", 0), "in": (1000.0, " V", 3), "curr": (1000.0, " A", 3),
    "power": (1e6, " W", 1), "energy": (1e6, " J", 0), "fan": (1.0, " rpm", 0),
    "humidity": (1000.0, "%", 0),
}
_HWMON_ORDER = ("temp", "fan", "pwm", "in", "curr", "power", "energy", "humidity", "intrusion")


def hwmon_chip(path):
    """One chip: {"name", "path", "device", "channels"}, or None when it has no readable
    channel. `device` is the basename of what the chip's `device` link points at (the PCI
    or platform device it belongs to, e.g. "0000:00:18.3"), or "virtual" for a chip with
    none. Channels carry the raw channel name, its type, a label (the driver's <ch>_label
    when it gives one, else the channel name), the value in the display unit, the unit,
    the declared (max, crit) limits for a temperature, and whether <ch>_alarm is set.
    Temperatures pass through sane() with a -40..125 window: a Super I/O reports -128 or
    127 for a diode that is not connected, and those are not temperatures. Every other
    type keeps 0 as a value, because 0 V on an idle USB-C source and 0 rpm on a parked fan
    are readings (the ucsi chip on this machine reads 0 on all five of its channels)."""
    name = read(f"{path}/name") or os.path.basename(path)
    dev_link = f"{path}/device"
    device = os.path.basename(os.path.realpath(dev_link)) if os.path.exists(dev_link) else "virtual"
    channels = []
    for f in glob.glob(f"{path}/*"):
        base = os.path.basename(f)
        m = re.match(r"^(temp|in|curr|power|energy|fan|humidity)(\d+)_input$", base)
        if m:
            typ, n = m.group(1), int(m.group(2))
            ch = f"{typ}{n}"
            raw = read_int(f, None)
            if raw is None:                       # EIO/ENODATA: the driver refused
                continue
            div, unit, _ = _HWMON_TYPES[typ]
            if typ == "temp":
                value = sane(f"hwmon.{path}.{ch}", raw // 1000, -40, 125)
                mx, crit = read_int(f"{path}/{ch}_max") // 1000, read_int(f"{path}/{ch}_crit") // 1000
                limits = (mx, crit) if (mx or crit) else None
            else:
                value, limits = raw / div, None
        elif re.match(r"^pwm\d+$", base):
            typ, n, ch = "pwm", int(base[3:]), base
            raw = read_int(f, None)
            if raw is None:
                continue
            value, unit, limits = round(raw * 100 / 255), "%", None   # 0-255 duty cycle
        elif re.match(r"^intrusion\d+_alarm$", base):
            typ, n, ch = "intrusion", int(base[9:].split("_")[0]), base.replace("_alarm", "")
            raw = read_int(f, None)
            if raw is None:
                continue
            value, unit, limits = raw, "", None
        else:
            continue
        channels.append({"ch": ch, "type": typ, "n": n,
                         "label": read(f"{path}/{ch}_label") or ch,
                         "value": value, "unit": unit, "limits": limits,
                         "alarm": read_int(f"{path}/{ch}_alarm", 0) == 1})
    if not channels:
        return None
    channels.sort(key=lambda c: (_HWMON_ORDER.index(c["type"]), c["n"]))
    return {"name": name, "path": path, "device": device, "channels": channels}


def discover_hwmon_extra(exclude=(), root="/sys/class/hwmon"):
    """Every chip under `root` except the ones whose real path is in `exclude` (the CPU
    sensor and each GPU's own node, which their panels already show) and the names in
    _HWMON_SKIP. `root` is a parameter so the reader can be pointed at a fake tree."""
    ex = {os.path.realpath(p) for p in exclude}
    out = []
    for h in sorted(glob.glob(f"{root}/hwmon*")):
        if os.path.realpath(h) in ex:
            continue
        chip = hwmon_chip(h)
        if chip and chip["name"] not in _HWMON_SKIP:
            out.append(chip)
    return out


def _hwmon_exclude(gpus):
    """The hwmon nodes other panels own: the CPU sensor and every GPU monitor's `hw`."""
    ex = set()
    if HW_CPU:
        ex.add(os.path.realpath(HW_CPU))
    for g in gpus:
        hw = getattr(g, "hw", None)
        if hw:
            ex.add(os.path.realpath(hw))
    return ex


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
    # swap split by destination: zram stays compressed in RAM, a swapfile leaves it
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


# MemTotal does not change for the life of a boot, so it is read once, and it is read
# through mem_sample() rather than through a second parser of /proc/meminfo, a second
# parser is a second place the same number can drift. It exists because it is the only
# honest way to tell an integrated GPU from a discrete one: not the vendor (Arc is Intel
# and discrete), not the name, but the measurable fact that the "VRAM total" a helper
# reports for the card is the machine's RAM. A 0 here means unreadable, and 0 never
# equals a real total, so the comparison simply never fires.
RAM_TOTAL_MIB = mem_sample().get("total") or 0


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
    """Every running llama-server, keyed by the port it listens on, read from the
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
    """Which speculative head a server uses, read from its command line, no endpoint
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
        # Same aliases as the config panel, and they used to differ. This list carried
        # `-n-max`, which is not a llama.cpp flag at all: dead code holding the place of
        # the real ones, while the panel's `speculative` group knew `--draft-max` and
        # `--draft`. Two readers of one flag with two lists is a divergence that ages
        # badly: the panel would show the value and this line would not. Checked against
        # llama.cpp's own arg.cpp: `--spec-draft-n-max` is the live flag, and
        # `--draft` / `--draft-n` / `--draft-max` were removed upstream (they now abort
        # with "use --spec-draft-n-max"). The removed ones are kept on purpose, because a
        # server running an older build still carries them on its command line.
        for flag in ("--spec-draft-n-max", "--draft-max", "--draft-n", "--draft"):
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
    2026-08-25 across the four servers here, :7795 (-c 12288 -np 2) reports 6144 a slot,
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
    ['Vulkan0', 'Vulkan1']. An empty list means the flag was absent, which in llama.cpp
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


# Every flag llama-server accepts, grouped for reading. This list is the order and the
# labels; it is not the filter. Whatever it does not name still reaches the panel through
# the "other" group below, computed as the complement of what these entries consumed. That
# inversion is the whole point: the previous version showed a hardcoded subset, so a run
# started with --dry-multiplier, --pooling, --jinja or --load-mode simply had no line
# anywhere, and nothing said so: across four servers only six to eight settings appeared
# out of the fifteen to eighteen flags actually passed. A curated list also ages against
# llama.cpp, since every flag added upstream stays invisible until someone edits this
# file, which is the same defect one release later.
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
        # the label of a negating switch is the concept, never the flag: "no-warmup off"
        # says warmup is on, which is the opposite of what --no-warmup did
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
        # DRY (Do not Repeat Yourself), the repetition sampler that works on n-grams
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
        # --draft/--draft-n/--draft-max were removed upstream in favour of the first;
        # kept for older builds, and deliberately identical to llama_spec_from_cmdline
        ("n-max", ("--spec-draft-n-max", "--draft-max", "--draft-n", "--draft")),
        ("n-min", ("--spec-draft-n-min", "--draft-min")),
        # p-min is the adaptive cut-off: the draft stops proposing once its own
        # confidence drops below it (default 0.00 = never cut). One of the biggest,
        # most-tuned knobs: it reshapes the acceptance curve.
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
    cannot see that from /proc, the live values are in the slot, not the command line."""
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
    command line that no process is running, including flags this build has never seen."""
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
                # a negating switch says what it turns off, so print the flag itself
                # rather than a bare "on" that reads as the opposite
                if v is True:
                    v = "off" if flag.startswith("--no-") else "on"
                shown.append(str(v))
            rows.append((label, " ".join(shown)))
        if rows:
            out[group] = rows
    # The complement, and it is what makes this list an ordering instead of a filter
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


def _nv_proc_fill(v):
    """Last resort for a process whose DRM counters were not readable: NVIDIA's own view.
    Only tried when fdinfo gave nothing AND the process holds a descriptor on the nvidia
    device, so it never invents a card for a CPU-only server.

    THE SOURCE IS RECORDED, not blended, because the two do not answer the same question.
    fdinfo splits device memory from the host-side GTT spill, which is the whole point of
    this panel; nvidia-smi reports one figure for memory ON THE CARD and does not split it.
    So `gtt` is set to None, UNKNOWN, instead of staying 0, because a 0 here reads as
    "nothing spilled" about a thing nobody measured. That is the same defect this file
    already fixed on prefill speed, one field along."""
    if v["read"] or not v["gpu_accel"]:
        return v
    mem = _NV_PROC_MEM.get(v["pid"])
    if mem and time.monotonic() - mem[1] < 6:
        if mem[0] is not None:
            v["vram"], v["gtt"], v["read"], v["vram_src"] = mem[0], None, True, "nvidia-smi"
        else:
            v["vram_src"] = "nvidia-smi-na"
    return v


def llama_processes():
    """Every llama.cpp process, with its resident memory and, read straight from
    fdinfo, how much of it is on the card (VRAM) versus spilled into host memory
    (GTT), which is the read-via-PCIe overflow that quietly tanks tokens/s. NVIDIA does
    not export those DRM counters at all, so there is a documented fallback to nvidia-smi
    (see _nv_proc_fill) that carries its own, narrower, meaning."""
    out = []
    for p in glob.glob(f"{_PROC_ROOT}/[0-9]*"):
        name = read(f"{p}/comm", "") or ""
        if not name.startswith("llama-"):
            continue
        v = {"pid": os.path.basename(p), "name": name, "rss": 0, "model": "",
             "vram": 0, "gtt": 0, "read": False, "caps": [], "error": None, "gpu_accel": False,
             "vram_src": None}
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
        # One pass over the fd links serves both questions: whether the process holds a GPU
        # device at all (gpu_accel, the same test as before) and WHICH descriptors are DRM
        # files, since only a link into /dev/dri can have a fdinfo with drm-* keys (the same
        # predicate the SMC gate's client scan uses). MEASURED by the auditor (strace, round
        # 7b, an idle llama-server on the W9170): the previous loop read the fdinfo of every
        # descriptor in order until the first with drm-memory-vram, fds 7, 8, 9 … 24, 633
        # reads in 45 s; on the captured llama-bench table (fds 0–32, the key on fd 24) that
        # is 25 reads per tick, 21 of them sockets, files and NVIDIA device nodes that can
        # never carry the key.
        # Now: the /dev/dri descriptors only, ascending, stopping at the first with the key
        # (the same descriptor as before: /proc lists fds ascending, fs/proc/fd.c:248-262,
        # and the sort makes the rule explicit rather than a property of the listing). The
        # driver-side cost is unchanged at one amdgpu_show_fdinfo per tick per process; the
        # NVIDIA render-node fdinfo carries no drm-driver line (not served by drm_show_fdinfo)
        # and is a plain proc read. Still per tick: the GTT figure is this panel's alarm and
        # the VRAM moves on a model load, and slowing it would save no driver read.
        # A process whose descriptors this user may not list (another user's, or a
        # non-dumpable one: /proc/<pid>/fd is 0500 owned per task_dump_owner, base.c:3321,
        # and fdinfo needs ptrace_may_access, fd.c:85-101, so the same processes deny both)
        # is reported as before, with its capabilities as the reason.
        drm_fds = []
        try:
            fds = os.listdir(f"{p}/fd")
        except OSError:
            v["caps"] = _capabilities(status)
            v["error"] = "oserror"
            out.append(_nv_proc_fill(v))
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{p}/fd/{fd}")
            except OSError:
                continue
            if any(k in target for k in ("nvidia", "renderD", "dxg", "wsl/drivers")):
                v["gpu_accel"] = True
                if "dxg" in target or "wsl/drivers" in target:
                    v["dxg"] = True
            if os.path.dirname(target) == "/dev/dri":
                drm_fds.append(fd)
        if v["pid"] in _NV_PROC_MEM:
            v["gpu_accel"] = True
        for f in sorted(drm_fds, key=int):
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
        out.append(_nv_proc_fill(v))
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
        # None means not measurable and is drawn as a dash with the reason. 0.0 would be
        # a claim ("this server prefills at zero tokens a second") that no source ever
        # made, and with /metrics off, which is the default, that claim was the only
        # thing the prefill field ever showed.
        self.tg = self.pp = None
        self.tg_last = self.pp_last = None   # rate of the last COMPLETED request
        # Time to first token of the last completed request, in seconds, as the server
        # timed it. It was already being computed and discarded: pp_last is prompt tokens
        # over prompt seconds, and that denominator is the wait before the first token.
        # Timing it here instead, by watching /slots, would carry the poll interval as its
        # error bar (0.4 s on a 1.94 s prefill is 20%), manufacturing precision when the
        # exact figure is already in hand, which is the defect this file fixed on prefill
        # speed. `_ttft_slots` records how many slots were working over that interval,
        # because with two the counters advanced for two requests and the sum is not a TTFT.
        self.ttft_last = None
        self._ttft_slots = 0
        # 6 s of trail at the fastest refresh (0.5 s) needs 13 slots, not 12: the old cap
        # silently shortened the window at the very rate a user picks to watch it closely.
        self._dhist = deque(maxlen=32)       # (time, n_decoded), generation
        self._phist = deque(maxlen=32)       # (time, n_prompt_processed), prefill
        self._dlast = self._plast = 0
        self._busy = False
        self.ctx = self.ctx_total = self.slots = None   # server FACTS, survive a bad tick
        self.model = ""
        self.model_at = 0.0
        self._rf = None                    # reasoning format (from /slots, survives ticks)
        self._sl = {}                      # per-slot trails keyed by slot id
        self.task_last = None
        self._cur_task = None
        self.tokens_session = 0
        self.tokens_total = None
        self._tokens_base = None
        self._tokens_pred_last = None
        self._session_tok_fallback = 0

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
        # A fixed sixty seconds was wrong in both directions at once: far too long a
        # penalty for one slow answer, and useless against a server that will stay silent
        # for longer than that anyway. During a 78-second prefill on CPU, /slots did not
        # answer at all: three consecutive 20-second timeouts, then a 13.8-second reply
        # the moment the prefill ended, and nothing under 1.2 seconds in between. No
        # timeout value would have helped, and the blackout hid the whole run. Hence:
        # retry immediately once, then back away geometrically, and never for long.
        # capped so the ladder below never computes 2**700 on a long outage just to throw
        # it away at the min(): the wait is bounded at 5 s from the fourth failure on
        self._fails = min(getattr(self, "_fails", 0) + 1, 8)
        # capped low on purpose: with the fan-out above, a probe that keeps failing no
        # longer slows anything else down, so there is nothing to buy by waiting longer,
        # and every second of wait is a second of missing the recovery
        self._timeout_until = time.monotonic() + (
            0 if self._fails < 2 else min(2 ** (self._fails - 1), 5))
        s = self._blank()
        s.update(self.state)               # keep whatever the last good tick knew
        s.update(phase="not answering", stale=True, pp=None, tg=None,
                 pp_last=self.pp_last, tg_last=self.tg_last)
        self.state = s
        return s

    # A prefill that lasts about as long as the poll interval cannot be measured by
    # polling: a 4001-token prefill running in 1.94 s leaves two usable samples, and the
    # slope through them reads around 790 t/s against the 2065 the server timed for the
    # same work. So the live prefill rate is published only once it has been running
    # longer than this. A large prompt on CPU takes a minute and is genuinely samplable;
    # a prefill on a GPU is over before the second tick. Below the threshold the
    # completed-request figure from /metrics is used instead, and it is exact. Publishing
    # the slope anyway would be manufacturing precision.
    _PREFILL_MIN_SPAN = 2.0

    @staticmethod
    def _slope(hist, cur, prev, minspan=0.6, window=6.0, anchor_zero=False):
        """Tokens per second from a (time, counter) trail, or None while the trail is
        too short to say. Called only while the slot is BUSY, that restriction is the
        fix, not a detail. Feeding it a finished request's counter, which llama.cpp
        leaves standing in the slot, makes the window slide off the end of the work: the
        numerator shrinks while the denominator stays ~6 s, so the reading decays, and
        once the window is entirely past the request the delta is 0 and the guard stops
        updating, leaving the last decayed value frozen on screen. MEASURED 2026-08-25
        on :8181, whose own timings reported 29.93 t/s: 29.7, 27.8, 24.7, 21.1, 17.7,
        14.2, 12.7, 9.4, 5.4, 1.606, then 1.606 for as long as anyone watched."""
        now = time.monotonic()
        if cur < prev:                     # counter restarted: a new request began
            hist.clear()
        # Where the trail starts is not the same question for the two rates, and using one
        # answer for both is wrong in opposite directions. Prefill starts at zero tokens
        # processed, so the zero is the anchor and dropping it loses the start of the work.
        # Generation also starts at zero decoded, but that zero sits at the end of prefill:
        # anchoring there would divide the generated tokens by generation time plus prefill
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
        MEASUREMENTS start empty. Every exit from sample() returns this shape, handing
        back a partial dict pushes the job of inventing defaults onto nine render sites,
        which is how a missing key becomes a zero on screen."""
        return {"alive": False, "phase": "off", "pp": None, "tg": None,
             "pp_last": self.pp_last, "tg_last": self.tg_last, "kv": None,
             "ttft_last": self.ttft_last, "ttft_slots": self._ttft_slots,
             "ctx": self.ctx, "ctx_total": self.ctx_total, "slots": self.slots,
             "model": self.model, "spec": None, "tok_step": None,
             "spec_acc": 0, "spec_draft": 0, "spec_pos": [], "active": None, "queued": None,
             "pp_life": None, "tg_life": None, "spec_type": None, "spec_head": None,
             "spec_nmax": None, "spec_stale": False, "cache_hit": 0, "prompt_new": 0,
             "max_tok": None, "reuse": None, "budget": 0, "decoded": 0, "kv_used": 0,
             "metrics_off": False, "slots_off": False, "stale": False, "kv_cap": None,
             "active_metric": None, "slot_stats": [],
             "task": None, "task_last": self.task_last,
             "tokens_session": self.tokens_session, "tokens_total": self.tokens_total}

    @staticmethod
    def _slot_close(st):
        """The request on a slot ended, or a new one began: whatever rate its trail had
        reached becomes that slot's LAST rate, and the trail closes so the next request
        starts from nothing instead of inheriting a slope it never ran at."""
        if st["tg"] is not None:
            st["tg_last"] = st["tg"]
        if st["pp"] is not None:
            st["pp_last"] = st["pp"]
        st["dh"].clear(); st["ph"].clear()
        st["dl"] = st["pl"] = 0
        st["tg"] = st["pp"] = None

    def _slot_stats(self, slots, busy_fn, next_fn):
        """One entry per slot of /slots, for a server started with -np N: its phase, the
        cache it holds against its OWN n_ctx, the generation budget, and a live rate for
        the request running on it. The figures above this one sum the slots — right for a
        single-slot server, and what it keeps showing — but with several slots working
        independently the sum hides which one is where. MEASURED 2026-09-17 on :8080
        (-np 5 -c 150000): slot 0 generating at 1294/30208, slots 1-4 idle and still
        holding 985, 1186, 237 and 5074 tokens of their last conversations.

        OCCUPANCY is max(n_prompt_tokens, cache + processed + decoded), because two builds
        disagree on what n_prompt_tokens holds. On b10930 it is prompt.tokens.size(), which
        grows by one per sampled token (server-context.cpp:540): 484 processed + 810
        decoded = 1294 reported, measured, so adding n_decoded to it counts the generation
        twice. On older builds it was the prompt alone and decoded had to be added. The max
        is right on both and cannot invent a token on either.

        RATES are the slope of the slot's own counters between polls — the same _slope()
        and the same busy-only rule as the aggregate — and they carry the poll interval as
        their error bar: the server times each request exactly but exports that figure
        only as fleet-wide /metrics deltas, never per slot. A trail closes on a CHANGE OF
        id_task rather than on the counter dropping, because at 0.4 s a poll can miss the
        restart of a request that counts straight past its predecessor. The two helpers
        are the ones sample() already defines for the aggregate, handed in so old builds
        (`state` int, top-level n_decoded, dict next_token) are read one way only."""
        out, seen = [], set()
        for i, sl in enumerate(slots):
            sid = sl.get("id", i)
            seen.add(sid)
            st = self._sl.get(sid)
            if st is None:
                st = self._sl[sid] = {"task": None, "dh": deque(maxlen=32),
                                      "ph": deque(maxlen=32), "dl": 0, "pl": 0,
                                      "tg": None, "pp": None, "tg_last": None,
                                      "pp_last": None, "busy": False}
            nx = next_fn(sl)
            busy = busy_fn(sl)
            dec = nx.get("n_decoded") or sl.get("n_decoded") or 0
            proc = sl.get("n_prompt_tokens_processed") or 0
            npt = sl.get("n_prompt_tokens") or 0
            npc = (sl.get("n_prompt_tokens_cache") or 0) + proc
            task = sl.get("id_task")
            if task != st["task"]:
                self._slot_close(st)
                st["task"] = task
            if busy:
                r = self._slope(st["dh"], dec, st["dl"])
                if r is not None:
                    st["tg"] = r
                r = self._slope(st["ph"], proc, st["pl"],
                                minspan=self._PREFILL_MIN_SPAN, anchor_zero=True)
                if r is not None:
                    st["pp"] = r
            elif st["busy"]:               # the tick the request ended: latch and close
                self._slot_close(st)
            st["dl"], st["pl"], st["busy"] = dec, proc, busy
            nctx = sl.get("n_ctx") or 0
            if not isinstance(nctx, int) or nctx < 0:
                nctx = 0
            used = max(npt, npc + dec)
            remain = nx.get("n_remain")
            out.append({"id": sid,
                        "phase": ("generating" if (busy and dec > 0)
                                  else "prefill" if busy else "idle"),
                        "n_ctx": nctx, "used": used,
                        "fill": min(1.0, used / nctx) if nctx else None,
                        "decoded": dec,
                        "budget": ((dec + remain) if isinstance(remain, int) and remain >= 0
                                   else None),
                        "processed": proc, "tg": st["tg"], "pp": st["pp"],
                        "tg_last": st["tg_last"], "pp_last": st["pp_last"]})
        for k in [k for k in self._sl if k not in seen]:
            del self._sl[k]
        return out

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
            # is not "every counter reads zero", it is "no counter exists", and reading it
            # as zeros is what pinned prefill, active, queued and prompt reuse to 0 on
            # every server at once.
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
                # a build that does not export the cache counter is not a build with zero
                # reuse. Forks differ here: ik_llama exports ten counters where mainline
                # exports fifteen, and reading the gap as a value would print a confident
                # "prompt reuse 0.0%" about something nothing measured
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
                if "llamacpp:tokens_predicted_total" in m:
                    cur_pred = int(c[C[2]])
                    if self._tokens_base is None:
                        self._tokens_base = cur_pred
                        self._tokens_pred_last = cur_pred
                    else:
                        delta = cur_pred - (self._tokens_pred_last or 0)
                        if delta > 0:
                            self.tokens_session += delta
                            self._tokens_pred_last = cur_pred
                        elif delta < 0:
                            self.tokens_session += cur_pred
                            self._tokens_base = 0
                            self._tokens_pred_last = cur_pred
                    self.tokens_total = cur_pred
                # a finished request moves these counters; their delta is the rate of the
                # request that just ended, which is the last rate, never the current one
                if self.cnt:
                    if c[C[0]] - self.cnt[C[0]] > 0 and c[C[1]] - self.cnt[C[1]] > 0:
                        _pdt = c[C[1]] - self.cnt[C[1]]
                        self.pp_last = (c[C[0]] - self.cnt[C[0]]) / _pdt
                        # keep the denominator too. `_bids` still holds the previous tick's
                        # busy slots at this point (the /slots block that refreshes it runs
                        # below), which is exactly the window these counters cover.
                        self.ttft_last = _pdt
                        self._ttft_slots = len(getattr(self, "_bids", ()) or ())
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
                # read. Say so; do not let ctx, kv and the phase fall through as zeros.
                d["slots_off"] = True
                raise ValueError("slots endpoint unavailable")
            x = s[0] if s else {}
            # Every slot, not slot 0. A server started with -np N hands a request to
            # whichever slot is free, so reading s[0] alone makes the panel say "idle"
            # while the server generates on another slot, and show that slot's leftover
            # n_decoded, which then sits frozen on screen. With two slots a request can
            # run to completion on slot 1 while slot 0 reports state 0 and holds
            # n_decoded from an older run, which makes roughly half the work invisible.
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
            # count towards the KV fill (those cells are occupied) and never towards a rate
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
            # the number of slots actually working is a live reading of the queue depth,
            # and unlike llamacpp:requests_processing it needs no --metrics
            # Direct observation beats a counter here: the number of slots actually
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
            # every slot on its own, beside the sums above (drawn only when there are
            # several); the parsed list is the one already in hand, no second request
            d["slot_stats"] = self._slot_stats(s, _slot_busy, _slot_next)

            _bs0 = busy_slots[0] if busy_slots else {}
            _tid = _bs0.get("id_task")
            _nptot = _bs0.get("n_prompt_tokens", 0) or 0
            _npcache = _bs0.get("n_prompt_tokens_cache", 0) or 0
            _npred = (_bs0.get("params", {}) or {}).get("n_predict") or (_bs0.get("params", {}) or {}).get("max_tokens") or 0
            _tbudget = d["budget"] or (_npred if isinstance(_npred, int) and _npred > 0 else 0)

            # ---- live rates, and only while the slot is busy ----------------------
            # The /metrics counters move only when a request finishes, so during a stream
            # the slot's own progress is the sole live source. When the slot goes idle
            # llama.cpp leaves the counters standing, so the trail must be closed here,
            # not left to age out: what stands after a request is the last rate, and it
            # is labelled as such rather than dressed up as the current one.
            if busy:
                if self._cur_task and _tid is not None and self._cur_task.get("id") != _tid:
                    self.task_last = {
                        "id": self._cur_task.get("id"),
                        "prompt_total": self._cur_task.get("prompt_total", 0),
                        "prompt_cache": self._cur_task.get("prompt_cache", 0),
                        "decoded": self._cur_task.get("decoded", 0),
                        "elapsed": max(0.0, now - self._cur_task.get("t_start", now)),
                        "pp": self.pp_last,
                        "tg": self.tg_last,
                    }
                    self._session_tok_fallback += self._cur_task.get("decoded", 0)
                    self._cur_task = None
                if not self._cur_task:
                    self._cur_task = {
                        "id": _tid,
                        "t_start": now,
                        "prompt_total": _nptot,
                        "prompt_cache": _npcache,
                        "decoded": dec,
                    }
                else:
                    self._cur_task["decoded"] = dec
                r = self._slope(self._dhist, dec, self._dlast)
                if r is not None:
                    self.tg = r
                r = self._slope(self._phist, _nproc, self._plast,
                                minspan=self._PREFILL_MIN_SPAN, anchor_zero=True)
                if r is not None:
                    self.pp = r
                d["phase"] = "generating" if dec > 0 else "prefill"
                if d["phase"] == "prefill":
                    cur_p = min(_nptot, _npcache + _nproc)
                    pct = cur_p / _nptot if _nptot > 0 else 0.0
                    rem = max(0, _nptot - _npcache - _nproc)
                    r_pp = self.pp or self.pp_last or d.get("pp_life")
                    eta = (rem / r_pp) if (r_pp and r_pp > 0 and rem > 0) else (0.0 if rem == 0 else None)
                else:
                    pct = min(1.0, dec / _tbudget) if _tbudget > 0 else 0.0
                    rem = max(0, _tbudget - dec) if _tbudget > 0 else None
                    r_tg = self.tg or self.tg_last or d.get("tg_life")
                    eta = (rem / r_tg) if (r_tg and r_tg > 0 and rem is not None) else None
                d["task"] = {
                    "id": _tid,
                    "phase": d["phase"],
                    "prompt_total": _nptot,
                    "prompt_cache": _npcache,
                    "prompt_proc": _nproc,
                    "decoded": dec,
                    "budget": _tbudget,
                    "pct": pct,
                    "eta": eta,
                }
            elif self._busy:               # the tick the request ended: latch and close
                if self.tg is not None:
                    self.tg_last = self.tg
                if self.pp is not None:
                    self.pp_last = self.pp
                if self._cur_task:
                    self.task_last = {
                        "id": self._cur_task.get("id"),
                        "prompt_total": self._cur_task.get("prompt_total", 0),
                        "prompt_cache": self._cur_task.get("prompt_cache", 0),
                        "decoded": self._cur_task.get("decoded", 0),
                        "elapsed": max(0.0, now - self._cur_task.get("t_start", now)),
                        "pp": self.pp_last,
                        "tg": self.tg_last,
                    }
                    self._session_tok_fallback += self._cur_task.get("decoded", 0)
                    self._cur_task = None
                self._dhist.clear(); self._phist.clear()
                self.tg = self.pp = None
                d["phase"] = "idle"
                d["task"] = None
            else:
                if self._cur_task:
                    self.task_last = {
                        "id": self._cur_task.get("id"),
                        "prompt_total": self._cur_task.get("prompt_total", 0),
                        "prompt_cache": self._cur_task.get("prompt_cache", 0),
                        "decoded": self._cur_task.get("decoded", 0),
                        "elapsed": max(0.0, now - self._cur_task.get("t_start", now)),
                        "pp": self.pp_last,
                        "tg": self.tg_last,
                    }
                    self._session_tok_fallback += self._cur_task.get("decoded", 0)
                    self._cur_task = None
                self.tg = self.pp = None
                d["phase"] = "idle"
                d["task"] = None
            d["task_last"] = self.task_last
            if self._tokens_base is None:
                self.tokens_session = self._session_tok_fallback
            else:
                self.tokens_session = max(self.tokens_session, self._session_tok_fallback)
            live_dec = dec if (busy and dec > 0) else 0
            d["tokens_session"] = self.tokens_session + live_dec
            d["tokens_total"] = (self.tokens_total + live_dec) if self.tokens_total is not None else None
            self._dlast, self._plast, self._busy = dec, _nproc, busy
            nctx = x.get("n_ctx", 0) or 0
            if not isinstance(nctx, int) or nctx < 0:
                nctx = 0                       # a negative context is not a context
            if nctx:
                self.ctx = nctx            # per-SLOT context; the total comes from /props
                d["ctx"] = nctx
            # New builds have n_prompt_tokens*; old builds do not, so fall back to
            # the decoded count alone (incomplete but honest). Summed over every slot,
            # busy or not, because an idle slot still holds its conversation in the cache.
            # AND THE GENERATED TOKENS ARE COUNTED ONCE. On b10930 n_prompt_tokens is
            # prompt.tokens.size(), which grows with every sampled token, so it already
            # holds the generation: MEASURED 2026-09-17 on :8080, 484 processed + 810
            # decoded = 1294 reported, and this line used to add the 810 again — 2104
            # for a slot holding 1294, +63%. On older builds n_prompt_tokens was the
            # prompt alone and the decoded count had to be added. max() of the two
            # readings is right on both and cannot invent a token on either; _slot_stats
            # uses the same formula per slot, so the bar and the slot lines agree.
            occupied = 0
            for sl in s:
                _npt = sl.get("n_prompt_tokens", 0) or 0
                _npc = (sl.get("n_prompt_tokens_cache", 0) or 0) \
                     + (sl.get("n_prompt_tokens_processed", 0) or 0)
                _sd = _slot_next(sl).get("n_decoded") or sl.get("n_decoded") or 0
                occupied += max(_npt, _npc + _sd)
            d["kv_used"] = occupied
            # denominator: the whole allocation when several slots share it, so the sum
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
        d["ttft_last"], d["ttft_slots"] = self.ttft_last, self._ttft_slots
        # reasoning format is in the slot data itself (both old and new builds).
        # Guarded, and the guard was removed once and put back: `x` is seeded to {} above,
        # which covers "/slots failed", but not "/slots answered with a list whose items are
        # not objects", there `.get` raises and takes the whole sample down with it, so a
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
                # /slots reports the per-slot context (-c divided by -np); the config panel
                # reads -c off the command line. Both are true and they are different
                # numbers under the same word, :7795 shows 6144 here and 12288 there.
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
    dropped when it exits. `explicit_port` FOCUSES on one port: only that server is
    probed and returned, and it is probed even if discovery did not find it. Returns one
    state dict per server, tagged with pid, port, and whether more than one server is
    present (which the UI uses to reveal per-server labels)."""
    servers = discover_llama_servers()
    if explicit_port:
        # Focus means focus. This argument used to only ever append: nothing here, in
        # _collect or in the draw loop filtered the list, so asking for one port on a
        # machine running four still drew four panels, and an unrecognised port drew five.
        # The help text has always claimed otherwise, and both halves of what it claims
        # hold together, so nothing has to be traded: filter to the requested port, and if
        # discovery missed it, probe it anyway. This function is the single point all
        # three surfaces pass through, the panel through the feed thread, --once and
        # --line through the collector, and --probe directly, so the filter cannot be
        # bypassed by forgetting a caller.
        # str().strip() because discovery yields ports as strings while a caller may hand
        # in an integer: comparing the wrong types would match nothing and draw an empty
        # panel, which is worse than the defect being fixed.
        want = str(explicit_port).strip()
        servers = [s for s in servers if s["port"] == want] or [
            {"pid": "?", "port": want, "host": "127.0.0.1", "model_hint": ""}]
    active, todo = set(), []
    for srv in servers:
        port = srv["port"]
        active.add(port)
        pr = _probes.get(port)
        if pr is None:
            pr = _probes[port] = LlamaProbe(port, srv["host"])
        todo.append((srv, pr))
    # Probe every server at once. Read one after another, a single server that has
    # stopped answering costs every other server its own timeout: one silent server
    # holding each pass for 2.5 s made three healthy ones refresh several times slower
    # than they could. That coupling is also what forced the retry ladder to back away
    # for many seconds at a time, which risks missing the moment the slow one recovers.
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
        # the VulkanN of this card for a process started from this tool's environment, and
        # why there is none when there is none (see the Vulkan device order above)
        why = (_vulkan_map or {}).get("why")
        print(f"   vulkan: {getattr(g, 'backend_dev', None) or '—'}" + (f" ({why})" if why else ""))
        if s.get("throttle"):
            print(f"   throttling: {', '.join(s['throttle'])}")
        if s.get("metrics_version"):
            print(f"   gpu_metrics v{s['metrics_version']}")
        link = s.get("pcie") or {}
        if link.get("card_gts") is not None:
            print("   pcie: " + "  ".join(f"{k}={v}" for k, v in link.items()
                                          if k not in ("aer", "aer_port", "aer_live", "aer_why")))
            print(f"   aer: live={link.get('aer_live')} why={link.get('aer_why')}")
            print(f"   aer card: {link.get('aer')}")
            print(f"   aer port: {link.get('aer_port')}")
        if s["vendor"] == "NVIDIA":
            # every field of the detail table, serial and uuid included: --probe is where
            # a card is identified, the panel is not (see _nv_detail_rows)
            keys = sorted(k for k in s if k in _NV_XML_TEXT or k in _NV_XML_NUM
                          or k in _NV_XML_POWER
                          or k in ("clock_reasons_active", "clock_reason_us", "clock_reasons",
                                   "temp_crit", "nv_driver", "cuda",
                                   "pcie_replay_session", "pcie_replay_rollover_session"))
            if keys:
                import textwrap
                print(textwrap.fill("  ".join(f"{k}={s[k]}" for k in keys), width=110,
                                    initial_indent="   nvidia-smi -q -x: ",
                                    subsequent_indent="      "))
            else:
                print("   nvidia-smi -q -x: no detail table for this card")
        if s["vendor"] == "AMD" and s.get("amd"):
            # the sysfs detail and the slow-tier state, see AmdGpu
            for line in _amd_probe_lines(s):
                print(line)
    c = cpu_sample()
    print(f"\n[CPU] {c['name']}")
    print(f"   util={c['util']:.0f}%  temp={c['temp']}C  freq={c['freq']}MHz  "
          f"power={c['power']}W (rapl:{'yes' if c['rapl_present'] else 'no'})  load={c['loadavg']}")
    print(f"   temps={c['temps']}  limits={c['temp_limits']}  relative={c['temp_relative']}")
    m = mem_sample()
    print(f"\n[MEM] free {m['free']} / {m['total']} MiB  swap {m['swap_used']}/{m['swap_total']}")
    for chip in discover_hwmon_extra(_hwmon_exclude(gpus)):
        print(f"\n[hwmon] {chip['name']}  device={chip['device']}  {chip['path']}")
        for ch in chip["channels"]:
            print(f"   {ch['ch']:<12} {ch['label']:<18} {ch['value']}{ch['unit']}"
                  f"  limits={ch['limits']}  alarm={ch['alarm']}")
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
        # its -dev list and the map its OWN environment gives those names (or why none)
        print(f"   dev={','.join(llama_devices_from_cmdline(ll['port'])) or '—'}  "
              f"vulkan={_vulkan_text(vulkan_map_for_pid(ll['pid']))}")
        for sl in ll.get("slot_stats") or []:
            print(f"   slot {sl['id']} {sl['phase']} {sl['used']}/{sl['n_ctx']} tok "
                  f"tg={_n(sl['tg'], 1)} pp={_n(sl['pp'])}")
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
    raising, and three readers parse external tools that way, intel_gpu_top, nvtop and
    nvidia-smi, so a sensor printing either would put it straight into the history; the
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
        # Nothing to hold is not a zero. The fallback used to end in 0 when the series
        # was still empty, so the first bad sample became a fabricated zero, and on a
        # source that never answers at all every sample took that branch and the whole
        # row charted as a measured flat line. An empty series is simply not drawn, so
        # the row appears once there is something true to put in it. Holding behaves
        # exactly as before the moment a real sample exists.
        if not s:
            return
        value = s[-1]
    s.append(value)


def median_dev(key, window=60):
    """Median and standard deviation of the RECENT samples (last `window`), not the
    whole history, over thousands of samples the median is so stable it looks
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
    """min, mean, max over the whole recorded history of a series, the spread the
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
, box drawing, blocks, sparklines, ·°±→… all take one cell, but the warning sign
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
    """Display width of a cell, a list of (text, attr) segments drawn as a unit."""
    return sum(_wwidth(t) for t, _ in cell)


def _flow(cells, width, sep=" │ "):
    """Pack a list of cells (each a segment list) into as few rows as possible without
    any row exceeding `width` cells, greedily, keeping order. This is what lets a
    section lay its metrics out HORIZONTALLY on a wide terminal, util, vram, temp and
    clock share one line, and reflow onto more lines as the terminal narrows, instead
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
    # A note that does not fit beside the title is left out whole. Drawn over the end of
    # the box, it was cut mid-number ("· 8.0 ─╮", the GB/s gone) because the corner is
    # painted last and covers the ellipsis, and a figure without its unit is worse than
    # no note. Same len() arithmetic as the fill line below, on purpose.
    if note and len(top) + len(note) + 2 > width: note = ""
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
    would fit the GTT, the kernel never had a reason to evict, it is host memory by
    choice (pinned transfer buffers, the embedding table) and -ts won't move it."""
    if gtt < 1000:
        return None
    if vram_free is not None and vram_free >= gtt:
        return (f"{gtt} MiB host memory, not eviction", WARN if gtt >= 3000 else OK)
    return (f"⚠ {gtt} MiB read via PCIe (spilled)", CRIT if gtt >= 3000 else WARN)


def _util_attr(u):
    return CRIT if (u or 0) >= 95 else WARN if (u or 0) >= 80 else OK


def _fmt_kbs(kbs):
    """A PCIe throughput figure as nvidia-smi reports it (`pci/tx_util`, an integer in
    "KB/s") in the unit its size calls for: KB/s below 1000, MB/s below 1 000 000, GB/s
    from there. Decimal, 1 KB = 1000 B: the same base as the link's `GB/s` cell on this
    row and as the bound in NvidiaGpu.sample (8 000 000 KB/s is the 8 GB/s link). nvml.h
    documents the counter only as "KB/s" and never says whether its KB is 1000 or 1024
    bytes, so no other conversion is invented here. Integer in KB/s, one decimal above:
    25 341 reads 25.3 MB/s and 2 103 906 reads 2.1 GB/s; the unit is chosen on the raw
    value, so the 50 values just under the GB/s switch read 1000.0 MB/s rather than
    being rounded into the next unit, and even that widest text is shorter than the
    "8000000 KB/s" it replaces. None, a figure the bound dropped, is the dash sane()
    promises."""
    if kbs is None:
        return "—"
    if kbs < 1000:
        return f"{kbs} KB/s"
    if kbs < 1_000_000:
        return f"{kbs / 1000:.1f} MB/s"
    return f"{kbs / 1e6:.1f} GB/s"


def _pcie_rows(s, inner):
    """The card's own PCIe negotiation and its AER counters, under a `pcie` label laid
    out the way the "no data" block is: what was negotiated, what the card can do, what
    the slot can do, and a flag only where the comparison means something (see
    _pcie_negotiation for which). A healthy link carries no WARN or CRIT anywhere in
    these rows; that is what keeps the one warning that can appear worth reading.
    On NVIDIA the driver's own view of the link is added, and the two sources are
    compared ONLY on the static maxima. The current generation and width are sampled at
    different moments by the two (sysfs every tick, nvidia-smi every third), and a card
    that drops to gen1 at idle would make them disagree on every other refresh; the
    maxima cannot change, so a disagreement there is a real finding. Empty for a sample
    without the negotiation keys, so the older pcie dict renders nothing new."""
    link = s.get("pcie") or {}
    if link.get("card_gts") is None:
        return []
    lab = 8

    def gen(g):
        return f"gen{g}" if g is not None else "gen?"

    # the card's own link bandwidth; the box note carries the root's (pcie_text), which
    # is the same number unless a switch sits between the two
    gbs = (link["card_gts"] * link["card_width"]
           * _PCIE_EFF.get(link["card_gts"], 128 / 130) / 8)
    cells = [[(f"{gen(link['gen'])} x{link['card_width']} now", 0)],
             [("card ", DIM), (f"{gen(link['card_max_gen'])} x{link['card_max_width']}", 0)]]
    if link.get("slot_max_width") is not None:
        cells.append([("slot ", DIM), (f"{gen(link['slot_max_gen'])} x{link['slot_max_width']}", 0)])
    cells.append([(f"{gbs:.1f} GB/s", 0)])
    if link.get("width_degraded"):
        floor = min(w for w in (link["card_max_width"], link.get("slot_max_width")) if w)
        cells.append([(f"⚠ width x{link['card_width']} below x{floor} both ends support", WARN)])
    if link.get("width_limited_by_slot"):
        cells.append([("width limited by slot", DIM)])
    if link.get("speed_limited_by_host"):
        cells.append([("speed limited by host", DIM)])
    if link.get("speed_below_both"):
        # "normal at idle", not "reduced when idle": the cell states the reading and
        # leaves the cause open, because the same reading under load is a real problem
        cells.append([("speed below both ends (normal at idle)", DIM)])
    if link.get("via_switch"):
        cells.append([("via on-card switch (matched by vendor)", DIM)])
    if s.get("pcie_tx_kbs") is not None:
        cells.append([("tx ", DIM), (_fmt_kbs(s["pcie_tx_kbs"]), 0),
                      (" rx ", DIM), (_fmt_kbs(s.get("pcie_rx_kbs")), 0)])
    if s.get("pcie_replay") is not None:
        # The GPU counts its own link retries (nvml.h, nvmlDeviceGetPcieReplayCounter:
        # "Retrieve the PCIe replay counter"), independently of the OS's AER machinery,
        # so on this machine it is the one link-error counter that can move at all. As
        # with AER, the alarm is the per-session delta and not the boot-time total: a
        # link that replayed during last week's re-seating and not since is healthy,
        # and replays arriving NOW are what a marginal riser looks like.
        rs = s.get("pcie_replay_session") or 0
        rrs = s.get("pcie_replay_rollover_session") or 0
        attr = WARN if (rs or rrs) else 0
        sess = lambda n: f" (+{n} this session)" if n else ""
        cells.append([("replays ", DIM), (f"{s['pcie_replay']}{sess(rs)}", attr),
                      (" rollovers ", DIM), (f"{s.get('pcie_replay_rollover')}{sess(rrs)}", attr)])
    dev_max, host_max = s.get("nv_gen_dev_max"), s.get("nv_gen_host_max")
    if ((dev_max is not None and link.get("card_max_gen") is not None
         and dev_max != link["card_max_gen"])
            or (host_max is not None and link.get("slot_max_gen") is not None
                and host_max != link["slot_max_gen"])):
        cells.append([(f"driver reports card gen{dev_max} host gen{host_max}", WARN)])
    rows = []
    if _show("pcie"):
        for i, r in enumerate(_flow(cells, inner - lab)):
            rows.append(([(f"{'pcie':<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)]) + r)
    aer = link.get("aer")
    if aer is not None and _show("aer"):
        if not link.get("aer_live"):
            # The kernel exposes the counters but nothing on this machine can increment
            # them (see _aer_live), so their zeros would be a structural zero: not "no
            # errors" but "no counting". The reason, dim, and no figures. Wrapped like
            # every other long row: appended as one row it was clipped by _put at 80
            # columns ("(0000:00:02.…"), which lost the second half of the sentence.
            import textwrap
            why = link.get("aer_why") or "not counted"
            for i, part in enumerate(textwrap.wrap(why, max(8, inner - lab))):
                rows.append(([(f"{'aer':<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)])
                            + [(part, DIM)])
            return rows
        rows += _aer_rows("aer", aer, inner, lab)
        # the link partner's view of the same link, when the kernel exposes it: it may
        # count errors the endpoint never saw (see pcie_aer)
        if link.get("aer_port") is not None:
            rows += _aer_rows("port", link["aer_port"], inner, lab)
    return rows


def _aer_rows(label, aer, inner, lab):
    """The AER counters of one device as rows under `label`: corrected (+session),
    nonfatal, fatal, the non-zero names, and on a root port its message totals."""
    # an uncorrectable error is a fault whenever it happened; corrected errors matter
    # when they are arriving now, hence the session delta and not the boot total
    attr = (CRIT if (aer["nonfatal"] or aer["fatal"])
            else WARN if aer["correctable_session"] else DIM)
    sess = f" (+{aer['correctable_session']} this session)" if aer["correctable_session"] else ""
    cells = [[("corrected ", DIM), (f"{aer['correctable']}{sess}", attr)],
             [("nonfatal ", DIM), (str(aer["nonfatal"]), attr)],
             [("fatal ", DIM), (str(aer["fatal"]), attr)]]
    if aer["nonzero"]:
        cells.append([(", ".join(aer["nonzero"]), attr)])
    if "rootport_cor" in aer:
        # message totals for the whole subtree below the port, attributed to no single
        # device (the ABI: "Total number of ERR_COR messages reported to rootport"), so
        # they inform and never colour
        cells.append([(f"received from subtree: cor {aer['rootport_cor']} · "
                       f"nonfatal {aer.get('rootport_nonfatal')} · "
                       f"fatal {aer.get('rootport_fatal')}", DIM)])
    rows = []
    for i, r in enumerate(_flow(cells, inner - lab)):
        rows.append(([(f"{label:<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)]) + r)
    return rows


def _nv_detail_rows(s, inner):
    """What nvidia-smi -q -x adds for one card, grouped under short labels, each group
    only when it has data and the whole thing empty for a sample without the detail
    table, so a card whose feed has not answered yet draws exactly what it drew before.
    Gated on the vendor rather than on the keys because AMD publishes vram_free too.
    NEVER serial or uuid here: the panel is meant to be screenshotted and the README
    carries a dump of it, the same reasoning as _SECRET_FLAGS. --probe prints both."""
    if s.get("vendor") != "NVIDIA":
        return []
    lab = 8
    groups = []
    # clocks: the P-state, what holds the clocks where they are (the neutral reasons; the
    # limiting ones are the ⚠ row _gpu_rows already draws) and, from the driver's own
    # counters, for how long each limit has been in force since it loaded
    cells = []
    if s.get("pstate"):
        cells.append([("pstate ", DIM), (s["pstate"], 0)])
    if s.get("clock_reasons"):
        cells.append([("held by: ", DIM), (", ".join(s["clock_reasons"]), DIM)])
    # a non-zero counter under a second (the P100s report sw_power_cap 427599 us) must
    # not read as "0s", which says the opposite of what the driver counted
    limited = [f"{_NV_CLOCK_REASON_NAMES.get(k, k.replace('_', ' '))} "
               f"{_fmt_dur(v / 1e6) if v >= 1e6 else '<1s'}"
               for k, v in (s.get("clock_reason_us") or {}).items() if v]
    if limited:
        cells.append([("time limited since driver load: ", DIM), (" · ".join(limited), DIM)])
    groups.append(("clocks", cells))
    # limits: the enforced power limit with the range it can be set in, every temperature
    # threshold the driver publishes, the clock ceilings and the application clocks
    cells = []
    if s.get("power_enforced") is not None:
        rng = " · ".join(f"{n} {s[k]:.0f}" for n, k in
                         (("default", "power_default"), ("min", "power_min"), ("max", "power_max"))
                         if s.get(k) is not None)
        cells.append([("power ", DIM), (f"{s['power_enforced']:.0f} W", 0)]
                     + ([(f" ({rng})", DIM)] if rng else []))
    temps = [f"{n} {s[k]}" for n, k in
             (("slowdown", "temp_slowdown"), ("shutdown", "temp_shutdown"),
              ("max op", "temp_max_op"), ("target", "temp_target"),
              ("tlimit", "temp_tlimit"), ("mem max", "temp_mem_max")) if s.get(k) is not None]
    if temps:
        cells.append([("temp ", DIM), (" · ".join(temps) + " °C", 0)])
    clks = [f"{n} {s[k]}" for n, k in
            (("core max", "clk_max_gfx"), ("vmem max", "clk_max_mem"),
             ("video max", "clk_max_video")) if s.get(k) is not None]
    if clks:
        cells.append([(" · ".join(clks) + " MHz", 0)])
    if s.get("clk_app_gfx") is not None or s.get("clk_app_mem") is not None:
        cells.append([("app clocks ", DIM),
                      (f"{s.get('clk_app_gfx', '—')}/{s.get('clk_app_mem', '—')} MHz", 0)])
    groups.append(("limits", cells))
    # memory: what the driver keeps for itself, what is free, and the BAR1 aperture
    cells = []
    if s.get("vram_reserved") is not None:
        cells.append([("reserved ", DIM), (f"{s['vram_reserved']} MiB", 0)])
    if s.get("vram_free") is not None:
        cells.append([("free ", DIM), (f"{s['vram_free']} MiB", 0)])
    if s.get("bar1_total") is not None:
        cells.append([("bar1 ", DIM), (f"{s.get('bar1_used', '—')}/{s['bar1_total']} MiB", 0)])
    groups.append(("memory", cells))
    # ecc, only on a card that has it: a double-bit error, or a page waiting to be
    # retired, is CRIT wherever it is counted; single-bit errors are WARN only while
    # they are volatile (since the last reset), the aggregate is the card's history
    cells = []
    if s.get("ecc_mode") is not None:
        cells.append([("mode ", DIM), (s["ecc_mode"], 0)])
        for n, sbe, dbe in (("volatile", "ecc_vol_sbe", "ecc_vol_dbe"),
                            ("aggregate", "ecc_agg_sbe", "ecc_agg_dbe"),
                            ("retired", "retired_sbe", "retired_dbe")):
            if s.get(sbe) is None and s.get(dbe) is None:
                continue
            attr = (CRIT if (s.get(dbe) or 0) > 0
                    else WARN if (sbe == "ecc_vol_sbe" and (s.get(sbe) or 0) > 0) else 0)
            cells.append([(f"{n} ", DIM), (f"sbe {s.get(sbe, '—')} dbe {s.get(dbe, '—')}", attr)])
        if s.get("retired_pending") is not None:
            cells.append([("pending ", DIM),
                          (s["retired_pending"], CRIT if s["retired_pending"] == "Yes" else 0)])
    groups.append(("ecc", cells))
    # video engines
    cells = []
    util = [f"{n} {s[k]}%" for n, k in
            (("enc", "enc_util"), ("dec", "dec_util"), ("jpeg", "jpeg_util"), ("ofa", "ofa_util"))
            if s.get(k) is not None]
    if util:
        cells.append([(" · ".join(util), 0)])
    if s.get("enc_sessions") is not None or s.get("fbc_sessions") is not None:
        cells.append([("sessions ", DIM),
                      (f"enc {s.get('enc_sessions', '—')} / fbc {s.get('fbc_sessions', '—')}", 0)])
    groups.append(("video", cells))
    # identity and modes: nothing here moves, so it sits last
    items = [("arch", s.get("arch")), ("brand", s.get("brand")), ("vbios", s.get("vbios")),
             ("part", " / ".join(str(s[k]) for k in ("board_part", "gpu_part") if s.get(k)) or None),
             ("inforom", s.get("inforom_img")), ("gsp fw", s.get("gsp_fw")),
             ("persistence", s.get("persistence")), ("compute mode", s.get("compute_mode")),
             ("display active", s.get("display_active")), ("ecc pending", s.get("ecc_pending")),
             ("accounting", s.get("accounting")), ("virt", s.get("virt")),
             ("driver", s.get("nv_driver")), ("CUDA", s.get("cuda"))]
    groups.append(("id", [[(f"{n} ", DIM), (str(v), 0)] for n, v in items if v is not None]))
    rows = []
    for name, cells in groups:
        if not _show(name):                    # switched off in the details menu
            continue
        for i, r in enumerate(_flow(cells, inner - lab)):
            rows.append(([(f"{name:<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)]) + r)
    return rows


_AMD_SLOW_NAMES = {"sclk": "core", "mclk": "vmem", "power": "power"}


def _amd_slow_cell(s):
    """The main-row label for the values a legacy AMD card read at idle: which ones, and
    how old the reading is, so a figure from the slow tier is never read as this tick's."""
    names = [_AMD_SLOW_NAMES[k] for k in ("sclk", "mclk", "power") if k in s["slow_fields"]]
    if not names:                      # nothing from the tier: nothing on this row
        return []
    return [("slow ", DIM), ("·".join(names) + f" idle only, {_fmt_dur(s['slow_age'])} ago", DIM)]


def _amd_detail_rows(s, inner):
    """What the amdgpu driver adds for one card beyond the main row, under the same group
    labels the NVIDIA detail uses (clocks, limits, memory, id), each group only when it has
    data and the whole thing empty for a sample without the `amd` dict, so a sample from an
    older reader draws exactly what it drew before. The DPM tables of a legacy card carry
    the age of the idle read they came from; the identity carries "at
    start". The pcie DPM table is the driver's power-state target and its "x16" is a string
    literal (smu7_hwmgr.c:5021-5023), so it is compared with the negotiated width from
    sysfs and flagged when they differ, never presented as the link."""
    a = s.get("amd")
    if s.get("vendor") != "AMD" or not a:
        return []
    lab = 8
    age = s.get("slow_age")
    ago = f"{_fmt_dur(age)} ago" if age is not None else None
    groups = []
    # clocks: the DPM tables with the active level starred, the pcie
    # DPM target, the profile and the levers that steer them
    cells = []
    for name in ("sclk", "mclk"):
        lv, act = a.get(f"{name}_levels"), a.get(f"{name}_active")
        if lv:
            txt = " ".join(f"{v}{'*' if i == act else ''}" for i, v in enumerate(lv))
            cells.append([(f"{name} ", DIM), (f"{txt} MHz", 0)])
    if s.get("slow_fields"):
        cells.append([(f"(idle only, {ago})", DIM)])
    elif ago and (a.get("sclk_levels") or a.get("mclk_levels")):
        cells.append([(f"levels from the idle read {ago}, marker withheld while the gate is shut", DIM)])
    # the DRM clients the SMC gate weighed, on a legacy card: who can put work on the card
    if a.get("clients_scans") or a.get("clients_why"):
        cells.append([("clients ", DIM), (_amd_clients_text(a), 0)])
    lv, act = a.get("pcie_levels"), a.get("pcie_active")
    if lv:
        same = len(set(lv)) == 1
        txt = (f"{lv[0].replace(', ', ' ')} ×{len(lv)}" if same
               else " ".join(f"{v}{'*' if i == act else ''}" for i, v in enumerate(lv)))
        if act is not None:
            txt += f", level {act}"
        cells.append([("pcie dpm ", DIM), (txt, 0)])
        link = s.get("pcie") or {}
        m = re.search(r"x(\d+)", lv[act if act is not None else 0])
        if m and link.get("card_width") and int(m.group(1)) != link["card_width"]:
            cells.append([(f"⚠ pcie dpm table x{m.group(1)}, link x{link['card_width']} "
                           f"(driver target, not the link)", WARN)])
    items = [("profile", a.get("profile")), ("perf level", a.get("perf_level")),
             ("dpm state", a.get("dpm_state"))]
    if a.get("cur_state") is not None and a.get("states"):
        names = dict(a["states"])
        items.append(("pstate", f"{a['cur_state']} {names.get(a['cur_state'], '?')} "
                                f"of {len(a['states'])}"))
    if a.get("forced_state") is not None:
        items.append(("forced", str(a["forced_state"])))
    if a.get("sclk_od") is not None or a.get("mclk_od") is not None:
        items.append(("od", f"sclk +{_n(a.get('sclk_od'))}% mclk +{_n(a.get('mclk_od'))}%"))
    cells += [[(f"{n} ", DIM), (v, 0)] for n, v in items if v is not None]
    groups.append(("clocks", cells))
    # limits: the power cap with its range, the thermal threshold, the fan and its reason,
    # the throttling log setting
    cells = []
    if s.get("power_cap") is not None:
        rng = " · ".join(f"{n} {a[k]:.0f}" for n, k in
                         (("default", "power_cap_default"), ("min", "power_cap_min"),
                          ("max", "power_cap_max")) if a.get(k) is not None)
        cells.append([("power cap ", DIM), (f"{s['power_cap']:.0f} W", 0)]
                     + ([(f" ({rng})", DIM)] if rng else []))
    if a.get("temp_crit") is not None:
        hyst = f" hyst {a['temp_crit_hyst']}" if a.get("temp_crit_hyst") is not None else ""
        cells.append([("temp crit ", DIM), (f"{a['temp_crit']}{hyst} °C", 0)])
    fan = []
    if a.get("fan_mode"):
        fan.append(a["fan_mode"])
    if s.get("fan_pct") is not None:
        rng = (f" ({a['pwm_min']}–{a['pwm_max']})"
               if a.get("pwm_min") is not None and a.get("pwm_max") is not None else "")
        fan.append(f"pwm {s['fan_pct']}%{rng}")
    if a.get("fan_min") is not None and a.get("fan_max") is not None:
        fan.append(f"tach range {a['fan_min']}–{a['fan_max']} rpm")
    if a.get("fan_target") is not None:
        fan.append(f"target {a['fan_target']} rpm")
    if fan:
        cells.append([("fan ", DIM), (" · ".join(fan), 0)])
    if a.get("fan_why"):
        cells.append([(a["fan_why"], DIM)])
    if a.get("throttle_log"):
        cells.append([(f"throttling logging {a['throttle_log'][0]}, {a['throttle_log'][1]} s", DIM)])
    groups.append(("limits", cells))
    # memory: the CPU-visible window, what is free, the GTT and the preemptible pool
    cells = []
    if a.get("vis_vram_total") is not None:
        cells.append([("visible ", DIM), (f"{_n(a.get('vis_vram_used'))}/{a['vis_vram_total']} MiB", 0)])
    if s.get("vram_free") is not None:
        cells.append([("free ", DIM), (f"{s['vram_free']} MiB", 0)])
    if s.get("gtt_total"):
        cells.append([("gtt ", DIM), (f"{_n(s.get('gtt_used'))}/{s['gtt_total']} MiB", 0)])
    if a.get("preempt_used") is not None:
        cells.append([("preempt ", DIM), (f"{a['preempt_used']} MiB", 0)])
    groups.append(("memory", cells))
    # id: read once at start, then the PCI power state and runtime PM, which are live
    cells = []
    if a.get("vbios"):
        build = f" build {a['vbios_build']}" if a.get("vbios_build") else ""
        cells.append([("vbios ", DIM), (f"{a['vbios']}{build}", 0)])
    if a.get("board"):
        cells.append([("board ", DIM), (a["board"], 0)])
    for name, ver in (a.get("fw") or {}).items():
        cells.append([(f"{name} ", DIM), (str(ver), 0)])
    if cells:
        cells.append([("(at start)", DIM)])
    if a.get("pci_power_state"):
        cells.append([("pci ", DIM), (a["pci_power_state"], 0)])
    if a.get("runtime_control") or a.get("runtime_status"):
        cells.append([("runtime pm ", DIM),
                      (f"{a.get('runtime_control') or '—'}, {a.get('runtime_status') or '—'}", 0)])
    for i, name in enumerate(a.get("missing") or []):
        cells.append([(("not exposed: " if i == 0 else "") + name, DIM)])
    groups.append(("id", cells))
    rows = []
    for name, cells in groups:
        if not _show(name):
            continue
        for i, r in enumerate(_flow(cells, inner - lab)):
            rows.append(([(f"{name:<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)]) + r)
    return rows


def _amd_probe_lines(s):
    """The `amd` dict and the slow-tier state of one card as --probe prints them: every
    key, sorted, one wrapped paragraph. break_on_hyphens off, a VBIOS part number holds
    hyphens and must not be split across lines."""
    import textwrap
    a = s.get("amd") or {}
    items = [f"{k}={a[k]}" for k in sorted(a)]
    items += [f"slow_age={s.get('slow_age')}", f"slow_fields={s.get('slow_fields')}"]
    return textwrap.wrap("  ".join(items), width=110, initial_indent="   amdgpu sysfs: ",
                         subsequent_indent="      ", break_on_hyphens=False)


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
    # `is not None`: a gated GPU really does sit at 0 MHz, and a truthy test hid the row
    # entirely, the same shape as the stopped fan a few boxes further down.
    if s.get("sclk") is not None:
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
    if s.get("fan_rpm") is not None:      # 0 rpm is a stopped fan, not a missing sensor
        pct = f" ({s['fan_pct']}%)" if s.get("fan_pct") is not None else ""
        cells.append([("fan ", DIM), (f"{s['fan_rpm']} rpm{pct}", 0)])
    # nvidia-smi reports the fan as a duty cycle only (--help-query-gpu, fan.speed: "The
    # fan speed value is the percent of the product's maximum noise tolerance fan speed
    # that the device's fan is currently intended to run at.") and no tachometer, so
    # fan_rpm stays None and the line above draws nothing. A separate statement, so that
    # line and every sample that reaches it are unchanged; 0 % is a parked fan and is
    # drawn, for the reason the rpm line gives.
    if s.get("fan_rpm") is None and s.get("fan_pct") is not None:
        cells.append([("fan ", DIM), (f"{s['fan_pct']}%", 0)])
    # a legacy AMD card's clocks and power come from an idle-only read (see AmdGpu): the
    # label sits in the same flow as the numbers, so they never read as this tick's
    if s.get("slow_fields"):
        cells.append(_amd_slow_cell(s))
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
    # Why the empty fields are empty. The README promises "a dash with a reason", and the
    # llama.cpp side keeps that promise (see _why); the card panel did not, and worse, a
    # field with no source did not even leave a dash, the cells are added inside `if vt:`
    # and friends, so they vanished. A silent absence reads worse than a dash: nothing
    # tells you whether the card is idle, a program is missing, or the tool is broken.
    # One line, grouped by reason, not a note per cell: _flow wraps, so every extra cell
    # costs a row on a narrow terminal, and the common case (one missing program taking out
    # every field) collapses to a single short line. Fixed field order, not dict order, so
    # the line does not reshuffle between ticks.
    needs = s.get("needs") or {}
    if needs:
        by_reason = {}
        for f in ("util", "vram", "temp", "power", "clocks"):
            if f in needs:
                by_reason.setdefault(needs[f], []).append(f)
        cells = [[(", ".join(fs), 0), (" — ", DIM), (why, DIM)]
                 for why, fs in by_reason.items()]
        lab = 8
        for i, r in enumerate(_flow(cells, inner - lab)):
            rows.append(([(f"{'no data':<{lab}}", DIM)] if i == 0 else [(" " * lab, 0)]) + r)
    # both return [] for a sample without their keys, so everything above is the whole
    # panel for a card whose sources have not changed
    rows += _pcie_rows(s, inner)
    rows += _nv_detail_rows(s, inner)
    rows += _amd_detail_rows(s, inner)
    return rows


def _cpu_rows(c, width):
    """The CPU packed onto as few lines as the width allows, utilisation (with its
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
    limits = c.get("temp_limits") or {}
    relative = c.get("temp_relative") or ()
    for label, v in (c.get("temps") or {}).items():
        if v is not None:
            # the driver's limits when it declared them (max → WARN, crit → CRIT, the
            # helper the GPU panel already uses), else the fixed 80/90; a channel whose
            # scale is not a physical temperature says so (see CPU_TEMPS_RELATIVE)
            cell = [(f"{label} ", DIM),
                    (f"{v}°C", _temp_attr_th(v, limits[label]) if label in limits else _temp_attr(v))]
            if label in relative:
                cell.append((" rel", DIM))
            cells.append(cell)
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


def _hwmon_text(ch):
    """A channel's value in its display unit, formatted per type."""
    if ch["type"] == "intrusion":
        return "alarm" if ch["value"] else "ok"
    dec = 0 if ch["type"] == "pwm" else _HWMON_TYPES[ch["type"]][2]
    return f"{ch['value']:.{dec}f}{ch['unit']}"


def _sensor_rows(chips, width):
    """Every other hwmon chip, one group per chip laid out like the config panel: the
    chip name in the 12-cell label column, then the device it belongs to and each
    channel as "label value". Temperatures are coloured by the limits the driver
    declares; a channel whose <ch>_alarm is set is WARN whatever its type; a temperature
    sane() refused (an unconnected diode) is not drawn at all. The name column trims a
    long chip name (ucsi_source_psy_0_00081 → ucsi_source); --probe prints it whole."""
    inner = width - 4
    lab = 12
    rows = []
    for chip in chips:
        cells = [[("device ", DIM), (chip["device"], DIM)]]
        for ch in chip["channels"]:
            if ch["value"] is None:
                continue
            attr = _temp_attr_th(ch["value"], ch["limits"]) if ch["type"] == "temp" else 0
            if ch["alarm"]:
                attr = WARN
            cells.append([(f"{ch['label']} ", DIM), (_hwmon_text(ch), attr)])
        name = _wtrim(chip["name"], lab - 1)
        for i, r in enumerate(_flow(cells, inner - lab)):
            rows.append(([(f"{name:<{lab}}", SER)] if i == 0 else [(" " * lab, 0)]) + r)
    return rows


def _chips_note(n):
    """The sensors box note, pluralised the way the title does "GPU" / "GPUs". Its own
    function because draw() cannot be exercised without a terminal and "1 chips" was
    on screen for exactly that reason."""
    return f"{n} chip" + ("s" if n != 1 else "")


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
    """Total watts the machine draws right now, summed from every meter it actually exposes,
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
    its session average and peak, and, when a server is generating, the fleet's energy
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
        tot_tok = sum((d.get("tokens_session") or 0) for d in (llamas or []) if d.get("alive"))
        if tot_tok > 0:
            line += [("  tokens ", DIM), (f"{_fmt_tokens(tot_tok)}", curses.A_BOLD), (" gen", DIM)]
            if _session["energy_j"] > 0:
                line += [(f" ({tot_tok / (_session['energy_j'] / 1000.0):.0f} tok/kJ)", OK)]
    sum_tg = sum((d.get("tg") or 0) for d in (llamas or []) if d.get("alive"))
    if sum_tg > 0 and total > 0:
        eff = sum_tg / (total / 1000.0)                # t/s per kW  ==  tokens/kJ
        if line:
            line.append(("      ", 0))
        # whole-system efficiency: Every server's tokens over the total system draw, as
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
        # A sampler value on the command line is what the server falls back to, not what
        # the model is running at: a client that puts temperature or repeat_penalty in the
        # request body wins, and /proc cannot see that. A slot visibly takes on one
        # client's values and then another's the moment the second one calls. True and
        # read as false is the expensive kind of wrong, and the correction costs one dim
        # line that appears only when it applies.
        if group == "sampling":
            rows.append([(" " * lab, 0),
                         ("↑ server defaults — a request carrying its own overrides them",
                          DIM)])
    return rows or [[("server started with all defaults", DIM)]]


def _llama_proc_rows(procs, width):
    """Per-process resident memory and the VRAM/GTT split from fdinfo, non-dumpable
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
                if p.get("gpu_accel"):
                    if p.get("dxg") or p.get("vram_src") == "nvidia-smi-na":
                        why = "not reported by driver (GPU-PV/WSL2)"
                    else:
                        why = "not reported by driver (NVIDIA?)"
                else:
                    why = "CPU only (no VRAM)"
            seg += [("   VRAM ", DIM), ("—", DIM), (f"  ({why})", DIM)]
        else:
            seg += [("   VRAM ", DIM), (f"{p['vram']}M", 0)]
            if p.get("vram_src") == "nvidia-smi":
                # this figure came from nvidia-smi, which reports memory on the card as one
                # number. The host-side spill this column exists for is not knowable from
                # it, so it says so. Printing 0 would claim "nothing spilled".
                seg += [("   GTT ", DIM), ("—", DIM),
                        ("  (nvidia-smi does not split vram/gtt)", DIM)]
            elif (p["gtt"] or 0) > 200:
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
        # the "last" marker rides on the unit, not on the note: the note is 26 cells wide
        # and min·avg·max already fills it, so a marker put there is the first thing
        # trimmed off, and a past reading that loses its marker is a past reading being
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
    time and 60.8% of its frames were IDENTICAL to the one before, frozen, then a jump.
    One column per tick satisfies it 98.7%.

    WHY THE SCALE IS SNAPPED. Fitting to the window keeps the detail (a series in a narrow
    band uses 7 of 9 glyphs instead of 3), but recomputing the exact min and max every tick
    moves the mapping constantly. Snapping both ends to a 1-2-5 step holds the mapping
    still until the range really changes: on the same run, 91.3% -> 98.7% scrolling with no
    loss of detail. A slow drift of the ends was tried first and measured WORSE than doing
    nothing (82.2%, and 56.6% on a bursty series), a bound that creeps every tick is the
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
    if maxv:                                   # absolute scale, percentages
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
    something changed, not just its value now. Every series is live even between
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
        # charted whenever the series exists, the decision about which servers get one
        # is made where the series is written (see _collect), not here, so there is one
        # place that knows. Unlike the two speeds above there is no peak test: a
        # generative server whose cache has genuinely stayed empty is a fact worth seeing,
        # where a speed that never occurred is a flat line about nothing. Absolute 0..100
        # scale, because it is a percentage and drawing 10% at full height would lie about
        # the quantity.
        if HIST.get(f"kv_series@{port}"):
            series.append((f"kv %{sfx}", f"kv_series@{port}", 100, 1))
    for i, s in enumerate(gpus_data):
        series.append((f"{s['vendor']} util %", f"gpu{i}_util", 100, 0))
    for i, s in enumerate(gpus_data):
        if s.get("vram_total"):
            series.append((f"{s['vendor']} vram %", f"gpu{i}_vram", 100, 0))
    series.append(("cpu util %", "cpu_util", 100, 0))
    series.append(("power W", "power_series", None, 0))
    series.append(("ram free MiB", "ram_series", None, 0))
    # the label column fits the longest label (e.g. "prefill t/s :8080"), so every
    # sparkline and value lines up in one clean column no matter how many servers run
    labw = max([15] + [_wwidth(lab) + 1 for lab, *_ in series])
    sw = max(16, width - labw - 38)
    rows = []
    for lab, key, mx, dec in series:
        hs = HIST.get(key)
        cur = hs[-1] if hs else 0
        bar, lo, hi = _sparkline(key, sw, mx)
        seg = [(f"{lab:<{labw}}", DIM)] + bar + [(f" {cur:>7.{dec}f}", 0)]
        # Peak is over the whole session, and it is back on every row. Replacing it with
        # the bar's own range was a bad trade: on a speed series the floor is almost always
        # zero (idle), so "0.0…35.9" is "peak 35.9" with noise in front of it, and the one
        # number worth keeping across a long run is the highest it ever went.
        _mn, _avg, peak = _extremes(key)
        if peak is not None:
            seg += [(f"  peak {peak:.{dec}f}", DIM)]
        # the bar shows the last `sw` ticks, so its floor is not the session's. Say where
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


def _fmt_tokens(n):
    """Format token count: exact with commas under 10k, then k / M."""
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{int(n):,}"


def _ctx_text(d):
    """/slots reports the per-SLOT context, the command line carries the total, and with
    -np 2 they differ by a factor of two under the same word (measured on :7795, 6144
    against 12288). Show both whenever they differ, so neither panel silently contradicts
    the other."""
    per, tot, sl = d.get("ctx"), d.get("ctx_total"), d.get("slots")
    if not per:
        return f"ctx {tot}" if tot else "ctx —"
    if tot and tot != per:
        # Two numbers that differ do not always differ for the same reason, and calling
        # both cases "per-slot versus total" produces nonsense: a server started with
        # -c 350000 and a single slot read "ctx 350208/slot, 350000 total", a part larger
        # than the whole. With several slots the split is the reason and the two multiply
        # out. With one slot they cannot, and what is left is llama.cpp rounding the
        # request up to its own block size: /slots reports what was allocated, the command
        # line what was asked for.
        # The allocated figure is the true one, so it leads, and the request is named as
        # a request instead of masquerading as a total.
        if sl and sl > 1:
            return f"ctx {per}/slot · {tot} total ({sl} slots)"
        return f"ctx {per} allocated (asked {tot})"
    return f"ctx {per}" + (f" ×{sl} slots" if sl and sl > 1 else "")


def _why(d, which):
    """Why a speed is unavailable, a dash with no reason just moves the puzzle."""
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
                # The two sources disagree, and the disagreement is the information.
                # `active` is counted from /slots by looking at which slots are working;
                # `active_metric` is llamacpp:requests_processing from /metrics. The first
                # is preferred because it needs no --metrics and cannot contradict the
                # phase drawn beside it, but the second was collected and never shown, a
                # field with no reader. requests_processing can stay at 1 after a request
                # has already returned, because that gauge follows the slot's release and
                # not its work. Showing the gap costs nothing when there is none, and when
                # there is one it says the server's own counter is lagging rather than
                # leaving two numbers to be reconciled by whoever notices.
                + ([(f"  (metrics says {d['active_metric']})", DIM)]
                   if (d.get("active_metric") is not None
                       and d.get("active") is not None
                       and d["active_metric"] != d["active"]) else [])
                + ([("   ⚠ stale", WARN)] if d.get("stale") else []))
    # prefill, generation (each live + windowed median±stddev) and the session lifetime
    # average (always available from the cumulative counters), packed onto one line
    speed = []
    # Three distinct states, and collapsing them is what made this panel lie: a live rate
    # (the request running right now), the last rate (the request that just finished,
    # true, but past, so it says so), and not measurable (no source exists on this build).
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
    # TTFT, and the label changes with the meaning rather than carrying a footnote. With a
    # single slot the interval covers one request and the number is the wait before its
    # first token. With two, the counters advanced for two requests and the sum is a total
    # prefill time, not a TTFT, so it is named that instead. Two names for two things is
    # what this file already does for per-slot versus total context.
    if d.get("ttft_last") is not None:
        _sl = d.get("ttft_slots") or 0
        if _sl > 1:
            speed.append([("prefill time ", DIM), (f"{d['ttft_last']:.2f}s", 0),
                          (f" ({_sl} slots, summed)", DIM)])
        else:
            speed.append([("ttft ", DIM), (f"{d['ttft_last']:.2f}s", 0), (" last", DIM)])
    s_tok = d.get("tokens_session")
    tot_tok = d.get("tokens_total")
    show_tok = s_tok is not None and (s_tok > 0 or tot_tok is not None)
    if show_tok:
        tok_cell = [("session ", DIM), (f"{_fmt_tokens(s_tok)}", curses.A_BOLD), (" gen tok", DIM)]
        if tot_tok is not None and tot_tok != s_tok:
            tok_cell.append((f" ({_fmt_tokens(tot_tok)} total)", DIM))
        speed.append(tok_cell)
    if d.get("pp_life") or d.get("tg_life"):
        speed.append([("avg ", DIM) if show_tok else ("session ", DIM),
                      (f"{_n(d.get('pp_life'))}/{_n(d.get('tg_life'), 1)}", 0),
                      (" t/s" if show_tok else " t/s avg", DIM)])
    elif d.get("metrics_off"):
        speed.append([("metrics off", WARN),
                      (" — start with --metrics for reuse, queue and session averages", DIM)])
    rows += _flow(speed, inner)
    # task progress bar (active task during prefill/gen or last completed task summary)
    task = d.get("task")
    if task:
        bw = max(8, min(16, inner // 6))
        t_id = f"#{task['id']} " if task.get("id") is not None else ""
        pct = task.get("pct", 0.0)
        eta = task.get("eta")
        eta_str = f"ETA {eta:.1f}s" if eta is not None and eta < 60 else f"ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta is not None else ""
        t_cells = []
        if task.get("phase") == "prefill":
            ptot = task.get("prompt_total", 0)
            pcache = task.get("prompt_cache", 0)
            pproc = task.get("prompt_proc", 0)
            cur_p = min(ptot, pcache + pproc)
            bar_cell = [("task ", DIM), (t_id, 0)] + _bar(pct, 1.0, bw, OK) + [(f" {pct * 100:.1f}%", 0)]
            t_cells.append(bar_cell)
            t_cells.append([(f"prefill {cur_p}/{ptot} tok", 0)])
            if pcache > 0:
                t_cells.append([(f"({pcache} cached)", DIM)])
        else:
            dec = task.get("decoded", 0)
            bg = task.get("budget", 0)
            if bg:
                bar_cell = [("task ", DIM), (t_id, 0)] + _bar(pct, 1.0, bw, OK) + [(f" {pct * 100:.1f}%", 0)]
                t_cells.append(bar_cell)
                t_cells.append([(f"gen {dec}/{bg} tok", 0)])
            else:
                t_cells.append([("task ", DIM), (t_id, 0), (f"gen {dec} tok", 0)])
        if eta_str:
            t_cells.append([(eta_str, OK)])
        rows += _flow(t_cells, inner)
    elif d.get("task_last"):
        tl = d["task_last"]
        t_id = f"#{tl['id']} " if tl.get("id") is not None else ""
        dec = tl.get("decoded", 0)
        ptot = tl.get("prompt_total", 0)
        pcache = tl.get("prompt_cache", 0)
        el = tl.get("elapsed")
        l_cells = [[("last task ", DIM), (t_id, 0), (f"{dec} gen tok", 0)]]
        if ptot:
            prompt_str = f"from {ptot} prompt"
            if pcache:
                prompt_str += f" ({pcache} cached)"
            l_cells.append([(prompt_str, DIM)])
        if el:
            rates = []
            if tl.get("pp"):
                rates.append(f"pp {tl['pp']:.0f} t/s")
            if tl.get("tg"):
                rates.append(f"gen {tl['tg']:.1f} t/s")
            rate_str = f" ({', '.join(rates)})" if rates else ""
            l_cells.append([(f"in {el:.1f}s{rate_str}", DIM)])
        rows += _flow(l_cells, inner)
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
    # ONE LINE PER SLOT, only when the server has more than one: with a single slot the
    # rows above ARE the slot, and a line repeating them would say the same thing twice
    # (that render stays byte-identical to the one before this block existed, by test).
    # An idle slot keeps its line, because it still holds its last conversation in the
    # cache and those tokens are what the kv bar above is made of. Cells flow like the
    # config panel's, so a 62-column terminal wraps a slot onto a second line instead of
    # losing the rate off the right edge. No rate on a stale tick: phase and occupancy
    # are the last good facts and the status line says so; a rate is a measurement, and
    # there was none.
    sls = d.get("slot_stats") or []
    if len(sls) > 1:
        sbw = max(6, min(10, inner // 8))
        for i, sl in enumerate(sls):
            ph = sl.get("phase") or "idle"
            cells = [[(f"{sl.get('id', i):>2} ", DIM),
                      (ph, OK if ph == "generating" else DIM if ph == "idle" else 0)]]
            used, nctx, fill = sl.get("used") or 0, sl.get("n_ctx") or 0, sl.get("fill")
            if nctx and fill is not None:
                c = CRIT if fill >= 0.95 else WARN if fill >= 0.85 else OK
                cells.append([("ctx ", DIM)] + _bar(used, nctx, sbw, c)
                             + [(f" {fill * 100:.1f}% {used}/{nctx}", 0)])
            else:
                cells.append([("ctx ", DIM), (f"{used} tok", 0)])
            if ph == "generating" and sl.get("budget"):
                q = (sl.get("decoded") or 0) / sl["budget"]
                cells.append([("budget ", DIM), (f"{sl.get('decoded') or 0}/{sl['budget']}",
                                                 WARN if q > 0.9 else 0)])
            if ph == "prefill":
                cells.append([("processed ", DIM), (f"{sl.get('processed') or 0} tok", 0)])
            if not d.get("stale"):
                for label, cur, last, dec in (("prefill", sl.get("pp"), sl.get("pp_last"), 0),
                                              ("gen", sl.get("tg"), sl.get("tg_last"), 1)):
                    if cur is not None:
                        cells.append([(f"{label} ", DIM), (f"{cur:.{dec}f}", 0), (" t/s", DIM)])
                    elif last is not None and ph == "idle":
                        cells.append([(f"{label} ", DIM), (f"{last:.{dec}f}", DIM),
                                      (" t/s last", DIM)])
            for j, r in enumerate(_flow(cells, inner - ET)):
                prefix = [(f"{'slots':<{ET}}", DIM)] if (i == 0 and j == 0) else [(" " * ET, 0)]
                rows.append(prefix + r)
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

# ------------------------------------------------------------- details menu
# The detail groups added on top of the original panel, each switchable from a dropdown
# (key m). Everything on by default, so a run that never opens the menu draws exactly
# what it drew before the menu existed; "Minimal" turns all nine off and gives back the
# panel as it was before any of them, plus the defect fixes, which are not details: the
# CPU Tctl cell and its limit colours, the fan % cell, the throughput bound, the bus-id
# correlation, the box-note rule and the box note itself stay whatever the menu says.
# Module-level state, the same idiom as _scroll and _rate.
_DETAIL_GROUPS = ("pcie", "aer", "clocks", "limits", "memory", "ecc", "video", "id", "sensors")
_details = {g: True for g in _DETAIL_GROUPS}
_menu = {"open": False, "cursor": 0}


def _show(group):
    """Whether a detail group is switched on; a name the menu does not know is shown."""
    return _details.get(group, True)


def _menu_items():
    """The entries in display order: the two presets, then one checkbox per group."""
    return [("All", "preset"), ("Minimal", "preset")] + [
        (f"[{'x' if _details[g] else ' '}] {g}", g) for g in _DETAIL_GROUPS]


def _menu_key(k):
    """One key while the menu is open: up/down (also k/j) move the cursor without
    wrapping, space or Enter toggles the group under it or applies the preset, and m, q
    or Esc close it. q closes the menu and does NOT quit the program while the menu is
    open: the key that opened a popup should never fall through to the key that ends the
    session. Esc closes it too; a bare Esc is otherwise ignored by the key loop because an
    arrow's escape prefix can arrive alone, and the worst that race can do here is close
    the menu, never quit."""
    last = len(_DETAIL_GROUPS) + 1
    if k in (curses.KEY_UP, ord("k")):
        _menu["cursor"] = max(0, _menu["cursor"] - 1)
    elif k in (curses.KEY_DOWN, ord("j")):
        _menu["cursor"] = min(last, _menu["cursor"] + 1)
    elif k in (ord(" "), 10, 13, curses.KEY_ENTER):
        c = _menu["cursor"]
        if c == 0:
            for g in _DETAIL_GROUPS:
                _details[g] = True
        elif c == 1:
            for g in _DETAIL_GROUPS:
                _details[g] = False
        else:
            g = _DETAIL_GROUPS[c - 2]
            _details[g] = not _details[g]
    elif k in (ord("m"), ord("M"), ord("q"), ord("Q"), 27):
        _menu["open"] = False


def _menu_lines():
    """The lines inside the popup, "> " before the cursor entry and "  " before the rest."""
    return [("> " if i == _menu["cursor"] else "  ") + label
            for i, (label, _) in enumerate(_menu_items())]


def _draw_menu(w, W, h):
    """The popup, top-right under the header, staged (noutrefresh) so the doupdate that
    follows paints it over the pad. Its own borders rather than _box: _put clips one
    column short of the window edge, which would cost the popup its right border. When
    the terminal is too short for every entry, the entries are windowed around the
    cursor so the cursor entry is always visible."""
    lines = _menu_lines()
    cursor = _menu["cursor"]
    room = max(1, h - 4)                       # header, footer and the two border rows
    start = 0
    if len(lines) > room:
        start = max(0, min(cursor - room // 2, len(lines) - room))
    shown = lines[start:start + room]
    title, foot = "details", "space · esc"
    width = min(W, max(_wwidth(x) for x in shown + [title, foot]) + 4)
    height = len(shown) + 2
    x, y = max(0, W - width), 1
    try:
        win = curses.newwin(height, width, y, x)
    except curses.error:
        return
    inner = width - 2
    top = "╭─ " + title + " " + "─" * max(0, inner - _wwidth(title) - 3) + "╮"
    bottom = "╰─ " + foot + " " + "─" * max(0, inner - _wwidth(foot) - 3) + "╯"
    # every addstr guarded like the rest of the file: the bottom-right cell of a window
    # is written and then reported as an error by curses, which is not a failure here
    for row, text in enumerate([top] + shown + [bottom]):
        try:
            if 1 <= row <= len(shown):
                entry = _wtrim(text, inner).ljust(inner)
                win.addstr(row, 0, "│", DIM)
                win.addstr(row, 1, entry, curses.A_REVERSE if start + row - 1 == cursor else 0)
                win.addstr(row, width - 1, "│", DIM)
            else:
                win.addstr(row, 0, _wtrim(text, width), DIM)
        except curses.error:
            pass
    win.noutrefresh()


def draw(w, gpus_data, cpu, mem, llamas, cfgs, procs, sensors=None):
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
    if sensors and _show("sensors"):
        blocks.append(("sensors", _sensor_rows(sensors, bw), _chips_note(len(sensors))))
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
    # the bar is a window now, not the whole session, so the note says its resolution:
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
    _on = sum(1 for g in _DETAIL_GROUPS if _details[g])
    foot = (f" q quit · ↑↓ PgUp/PgDn scroll · +/- rate {_interval():g}s · z reset"
            f" · m details {_on}/{len(_DETAIL_GROUPS)}   {more}")
    _put(w, h - 1, 0, [(foot.ljust(W - 1), curses.A_REVERSE)], W)
    try:
        w.chgat(h - 1, 0, W, curses.A_REVERSE)
    except curses.error:
        pass
    # Stage the screen, then paint. stdscr holds the header, the footer and a blank
    # body; the pad holds the body. Both are staged (noutrefresh) and a single
    # doupdate paints them together, with the pad staged last so it wins the body
    # region. Doing pad.refresh() before stdscr.refresh() (the obvious order) is the
    # trap: stdscr's erased body would then wipe the pad back out on the next update.
    w.noutrefresh()
    try:
        pad.noutrefresh(_scroll[0], 0, body_top, 0, body_bot, W - 1)
    except curses.error:
        pass
    # the details popup is staged last so it wins its region over the pad
    if _menu["open"]:
        _draw_menu(w, W, h)
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
    # every hwmon chip no panel above owns, re-read each tick like the rest (a handful
    # of sysfs files; the discovery glob is the same one the CPU reader runs at start)
    sensors = discover_hwmon_extra(_hwmon_exclude(gpus))
    for i, s in enumerate(gpus_data):
        # Not `or 0`: that converted None before record() could apply its own
        # hold-the-previous rule, so a card with no source for utilisation charted as a
        # card measured at 0 %. With a card reporting no utilisation and a reason for it,
        # six ticks produced six zeros and the row rendered as a flat measured idle,
        # while the panel one box above correctly printed a dash beside the reason. The
        # trend panel's own contract, repeated word for word in the README, is that a dot
        # means a true zero and nothing else.
        record(f"gpu{i}_util", s.get("util"))
        if s.get("vram_total"):
            record(f"gpu{i}_vram", 100.0 * (s.get("vram_used") or 0) / s["vram_total"])
    cpu = cpu_sample()
    mem = mem_sample()
    # the TUI hands in a feed thread and never waits; --once/--line have no loop to
    # protect and read directly
    llamas = lfeed.data if lfeed is not None else sample_llama_fleet(explicit_port)
    # hardware trend series, always live, so the trend panel is never blank
    total_w, _ = system_power(gpus_data, cpu)      # single source of truth for the whole-box draw
    record("power_series", total_w)
    record("ram_series", mem["free"])
    # as the gpu util above: None is not a zero, and record() knows what to do
    record("cpu_util", cpu.get("util"))
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
        # charge this server's energy line to the power of the GPUs it runs on (its -dev
        # list; all cards when -dev is absent), not the whole-box draw, which is the power
        # panel's job. So the per-session figure and the system figure are genuinely different
        # numbers: the session one omits the CPU and any card this server does not use.
        _devs = llama_devices_from_cmdline(port)
        # the names on ITS command line mean what ITS environment makes them mean, so the
        # map is the server's, not the panel's (see vulkan_map_for_pid)
        _vk = vulkan_map_for_pid(d.get("pid")).get("map") or {}
        _gp = [s for s in gpus_data if (not _devs)
               or (_vk.get(s.get("pci_addr")) is not None and f"Vulkan{_vk[s['pci_addr']]}" in _devs)]
        if _devs and not _gp:                 # -dev set but unmappable (the map says why) → all cards
            _gp = gpus_data
        d["power_w"] = sum((s.get("power") or 0) for s in _gp)
        cfgs[port] = llama_settings_from_cmdline(port) if d.get("alive") else None
        # each server's speed histories are tied to its model: reset on a swap
        model = d.get("model") or ""
        if model and model != _model_ports.get(port):
            # reset only the median histories, so median / min·avg·max describe the new
            # model. The trend sparklines (…_series) are deliberately not reset, they
            # roll continuously across model swaps, which is exactly what a "recent
            # history" view wants, and is why the trend no longer blanks when the
            # campaign moves to the next model.
            for k in (f"pp_gen@{port}", f"tg_gen@{port}"):
                HIST.pop(k, None)
            _last_pp.pop(port, None)
            _model_ports[port] = model
        # A prefill is an event, not a state, and that distinction decides whether its
        # trend line exists at all. Generation spans many ticks, so charting the live rate
        # draws it correctly. A prefill on a GPU is over before the second tick, four
        # thousand tokens in under two seconds, so the live slope is almost always absent
        # and a series fed only from it stays at zero forever, peaks at zero, and is
        # dropped by the "only chart a speed that has actually occurred" test below. The
        # line was therefore invisible on exactly the servers fast enough to be worth
        # watching. So: chart the live rate while one is running, and the completed rate
        # on the tick it first lands, giving one spike per prefill.
        pp_live, pp_done = d.get("pp"), d.get("pp_last")
        fresh = pp_done if (pp_done and abs(pp_done - _last_pp.get(port, 0.0)) > 1e-6) else None
        pp_now = pp_live if pp_live is not None else fresh
        # `and not stale` because the true statement spelled out just below, that an idle
        # server really is producing nothing, stops being true the moment the server goes
        # silent: the retry ladder restores the alive flag from the last good state while
        # the rate is forced to None, so a server that answered nothing charted as a
        # server measured at zero. A silence of over a minute during a prefill on CPU
        # would draw a minute of generating nothing that nothing observed. Idle and mute
        # are different facts.
        if d.get("alive") and not d.get("stale"):
            # a series records 0 while idle, that is a true statement about the server
            # (it is producing nothing) and it is what keeps the trend line continuous
            record(f"tg_series@{port}", d.get("tg") or 0)
            record(f"pp_series@{port}", pp_now or 0)
            # The KV must not copy the two lines above, and this is the whole subtlety.
            # They record 0 while the server is idle, which is a true statement about a
            # speed, since nothing is being produced. It is false about the cache: a slot
            # that is not working still holds its conversation, and those cells are still
            # taken. A server with both slots idle can still hold a couple of thousand
            # tokens, and recording 0 there would draw a collapse at every pause that
            # never happened. So the real value is recorded whatever the phase, and only
            # a KV that could not be computed at all is skipped.
            # ...and it is only charted for a generative server. An embedding or rerank
            # server has a KV figure too, but it never moves in a way anyone reads, and
            # this panel already made that exact call one line above ("so idle embedding /
            # rerank servers don't fill the panel with flat-zero lines"). The role is read
            # from the `role` group the config panel already builds, --embedding,
            # --reranking, --pooling, so this is the same fact, not a second opinion about
            # it. A server reached by port with no local process has no config: it is
            # treated as generative, which is the safe way to be wrong here.
            if d.get("kv") is not None and "role" not in (cfgs.get(port) or {}):
                record(f"kv_series@{port}", d["kv"] * 100)
        # the median window, unlike the trend, must only ever see real measurements
        if d.get("phase") == "generating" and (d.get("tg") or 0) > 0:
            record(f"tg_gen@{port}", d["tg"])
        if pp_now and abs(pp_now - _last_pp.get(port, 0.0)) > 1e-6:
            record(f"pp_gen@{port}", pp_now)
            _last_pp[port] = pp_now
    # remember every port that currently has a live server, so the trend keeps its line
    # (and its rolling history) while the server briefly restarts between runs. A port
    # gone for more than two minutes is genuinely finished and forgotten.
    _vulkan_forget_pids({d.get("pid") for d in llamas})
    for d in llamas:
        if d.get("alive"):
            _port_seen[d["port"]] = now
    for port in [p for p, t in list(_port_seen.items()) if now - t > 120]:
        # kv_series was added to the writer earlier today and not to this tuple, so a
        # retired port left its KV history in hist for the life of the process, never read
        # again either, since _trend_rows only looks up ports still in _port_seen. Bounded
        # per entry by the deque, unbounded in the number of distinct ports a long session
        # sees. The defect was mine and it is the exact shape a new field always takes:
        # the writer is updated and the consumer that retires it is not.
        for k in (f"pp_gen@{port}", f"tg_gen@{port}", f"pp_series@{port}",
                  f"tg_series@{port}", f"kv_series@{port}"):
            HIST.pop(k, None)
        _port_seen.pop(port, None)
        _model_ports.pop(port, None)
        _last_pp.pop(port, None)
    return gpus_data, cpu, mem, llamas, cfgs, procs, sensors


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
    # Input and data collection are decoupled. getch half-blocks (timeout) so keys,
    # including the multi-byte arrow/PgUp/End escape sequences, are assembled and
    # answered within 100 ms; scrolling redraws immediately from the data already in
    # hand. A full _collect() (which includes a blocking HTTP read of the server) runs
    # only on its own cadence, never in the keypress path, that coupling was what made
    # scrolling stutter and the screen jump. Bare esc no longer quits, so an arrow key
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
        # While the details menu is open every key goes to it, and the key is then
        # swallowed (k = -1) so the chain below cannot see it, q included. No `continue`:
        # the collection cadence at the bottom of the loop must keep running while the
        # menu is open, or the panel behind it would freeze.
        if _menu["open"] and k != -1:
            _menu_key(k)
            dirty = True
            k = -1
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
        elif k in (ord("m"), ord("M")):
            _menu["open"] = True; dirty = True
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


def _text_line(gpus_data, cpu, mem, llamas, cfgs=None, procs=None, sensors=None):
    # `sensors` is accepted because _collect now hands it over with the rest and this
    # function is called with the whole tuple, and deliberately unused: the one-line
    # format is what --line writes to logs, and a log format stays stable
    parts = []
    total_w, _ = system_power(gpus_data, cpu)
    # This function used to hold two standards. The llama half below already routed every
    # rate through _n(), which prints a dash when no source said anything, while this half
    # printed `or 0`, so a missing reading arrived as the number zero: the one thing the
    # program promises not to do. On a card whose hwmon exposes no energy counter, --probe
    # read the power as absent while this line printed 0 W for the same tick.
    # `is None` rather than truthiness, because 0 W, 0 % and an idle GPU are real
    # readings and have to survive.
    for s in gpus_data:
        # The memory half of this statement kept the `or 0` idiom when utilisation and
        # power moved onto _n() below, so a missing used figure still printed as zero over
        # the total: a partial fix, which is worse than none because the line then looks
        # repaired. It is reachable because the used and total figures are separate reads
        # and can disagree on presence, the same partial-hardware shape that slips past a
        # check written for missing hardware. _n() also drops the trailing ".0" that the
        # NVIDIA parser leaves on both numbers.
        vram = (f"{_n(s.get('vram_used'))}/{_n(s.get('vram_total'))}"
                if s.get("vram_total") else "-")
        parts.append(f"{s['vendor']} util {_n(s.get('util'))}% vram {vram} "
                     f"{s.get('temp_main') or '-'}C {_n(s.get('power'))}W")
    parts.append(f"CPU {_n(cpu.get('util'))}% {cpu.get('temp') or '-'}C "
                 f"{('%.0fW' % cpu['power']) if cpu.get('power') is not None else '-'}")
    parts.append(f"RAM {mem['free']}/{mem['total']}")
    parts.append(f"PWR {total_w:.0f}W")
    multi = len(llamas or []) > 1
    any_alive = any(d.get("alive") for d in (llamas or []))
    for d in (llamas or []):
        if not d.get("alive"):
            if any_alive:
                continue
            # The only server, and it is not answering. Skipping a dead server is right
            # while another one is alive: a log should not fill with a server that is
            # simply down. It was also what happened when the fleet held a single entry,
            # which became reachable the moment the port argument started to focus, and
            # then the line said nothing about llama at all, which cannot be told apart
            # from the tool having ignored the argument. The other two surfaces were
            # already honest here: --probe prints "alive=False phase=off" and the panel
            # prints "status off ... stale" with "prefill, (server did not answer)".
            # The reason is _why's own, not a sixth wording of the same fact.
            parts.append(f"llama :{d.get('port')} {d.get('phase') or 'off'} "
                         f"({_why(d, 'tg')})")
            continue
        sp = f" spec {d['spec'] * 100:.0f}%" if d.get("spec") is not None else ""
        tag = f":{d['port']} " if multi else ""
        # a log line has no room for a column that says "this one is from the request
        # that just ended", so it says it inline. Writing the last rate bare is the same
        # defect the panel was carrying: a past number wearing the present tense.
        def _sp(live, last, dec):
            if live is not None:
                return _n(live, dec)
            return f"{_n(last, dec)}(last)" if last is not None else "—"
        tok_str = ""
        s_tok = d.get("tokens_session")
        tot_tok = d.get("tokens_total")
        if s_tok is not None:
            if s_tok > 0:
                tok_str = f" tok {_fmt_tokens(s_tok)}"
            elif tot_tok:
                tok_str = f" tok 0 ({_fmt_tokens(tot_tok)})"
            else:
                tok_str = " tok 0"
        elif tot_tok:
            tok_str = f" tok {_fmt_tokens(tot_tok)}"
        parts.append(f"llama {tag}{d.get('phase')} "
                     f"pp {_sp(d.get('pp'), d.get('pp_last'), 0)} "
                     f"tg {_sp(d.get('tg'), d.get('tg_last'), 1)}{sp}{tok_str}")
    return time.strftime("%H:%M:%S") + " | " + " | ".join(parts)


def _maybe_enable_rapl():
    """CPU wattage comes from the RAPL energy counter, which most kernels expose only to
    root (the Platypus side-channel mitigation). If it is unreadable and we are on a
    terminal, offer to open it for this session with one sudo call, the same thing the
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
        print("TUI keys:  q quit   ↑↓ PgUp/PgDn Home/End scroll   +/- refresh rate   z reset history"
              "   m details menu")
        print("Honors NO_COLOR. No dependencies beyond the Python standard library;")
        print("optional helpers used when present: intel_gpu_top, nvtop, nvidia-smi.")
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

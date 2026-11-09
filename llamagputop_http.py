#!/usr/bin/env python3
"""Read-only HTTP wrapper for llamagputop's selected llama.cpp server."""

import argparse
import glob
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llamagputop as monitor


_CONFIG_FIELDS = {
    "ctx": ("loading", "ctx"),
    "slots": ("loading", "slots"),
    "ngl": ("loading", "ngl"),
    "flash-attn": ("loading", "flash-attn"),
    "threads": ("loading", "threads"),
    "batch": ("loading", "batch"),
    "temp": ("sampling", "temp"),
    "top-k": ("sampling", "top-k"),
    "top-p": ("sampling", "top-p"),
    "min-p": ("sampling", "min-p"),
    "repeat": ("sampling", "repeat"),
    "kv-k": ("cache", "kv k"),
    "kv-v": ("cache", "kv v"),
    "spec-type": ("speculative", "type"),
    "n-max": ("speculative", "n-max"),
    "draft-kv-k": ("speculative", "draft-kv k"),
    "draft-kv-v": ("speculative", "draft-kv v"),
}


def _setting(config, group, label):
    for name, value in config.get(group, ()):
        if name == label:
            return value
    return None


def _value(value):
    if value is None or not isinstance(value, str):
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _pair(config, first, second):
    left = _setting(config, *_CONFIG_FIELDS[first])
    right = _setting(config, *_CONFIG_FIELDS[second])
    if left is None and right is None:
        return None
    return f"{left if left is not None else 'null'}/{right if right is not None else 'null'}"


def _command_for_port(port):
    wanted = str(port)
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = [arg for arg in open(path, encoding="utf-8").read().split("\x00") if arg]
        except OSError:
            continue
        if "--port=" + wanted in cmd or "-p=" + wanted in cmd:
            return cmd
        for flag in ("--port", "-p"):
            if flag in cmd:
                index = cmd.index(flag)
                if index + 1 < len(cmd) and cmd[index + 1] == wanted:
                    return cmd
    return []


def _raw_flag(cmd, aliases):
    for index, flag in enumerate(cmd[:-1]):
        if flag in aliases:
            return cmd[index + 1]
        for alias in aliases:
            if flag.startswith(alias + "="):
                return flag.split("=", 1)[1]
    return None


def _raw_pair(port, first, second):
    cmd = _command_for_port(port)
    left = _raw_flag(cmd, first)
    right = _raw_flag(cmd, second)
    if left is None and right is None:
        return None
    return f"{left if left is not None else 'null'}/{right if right is not None else 'null'}"


def model_stats(llama):
    return {
        "prefill": llama.get("pp"),
        "gen": llama.get("tg"),
        "session-avg": llama.get("tg_life"),
        "reasoning": llama.get("reasoning_format") or "none",
        "draft-accepted-p": llama.get("spec"),
        "draft-accepted-tok": llama.get("tok_step"),
        "draft-accepted-total": llama.get("spec_acc"),
    }


def model_config(llama, config, port=None):
    ctx = llama.get("ctx_total") or llama.get("ctx") or _value(
        _setting(config, *_CONFIG_FIELDS["ctx"]))
    spec_type = llama.get("spec_type") or _setting(config, *_CONFIG_FIELDS["spec-type"])
    kv_pair = _pair(config, "kv-k", "kv-v")
    draft_kv_pair = _pair(config, "draft-kv-k", "draft-kv-v")
    if port is not None:
        kv_pair = kv_pair or _raw_pair(
            port, ("-ctk", "--cache-type-k"), ("-ctv", "--cache-type-v"))
        draft_kv_pair = draft_kv_pair or _raw_pair(
            port,
            ("-ctkd", "--cache-type-k-draft", "--spec-draft-type-k"),
            ("-ctvd", "--cache-type-v-draft", "--spec-draft-type-v"),
        )
    result = {
        "ctx": ctx,
        "ngl": _value(_setting(config, *_CONFIG_FIELDS["ngl"])),
        "flash-attn": _value(_setting(config, *_CONFIG_FIELDS["flash-attn"])),
        "threads": _value(_setting(config, *_CONFIG_FIELDS["threads"])),
        "batch": _value(_setting(config, *_CONFIG_FIELDS["batch"])),
        "slots": llama.get("slots") or _value(
            _setting(config, *_CONFIG_FIELDS["slots"])),

        "kv-k/v": kv_pair,
        "temp": _value(_setting(config, *_CONFIG_FIELDS["temp"])),
        "top-k": _value(_setting(config, *_CONFIG_FIELDS["top-k"])),
        "top-p": _value(_setting(config, *_CONFIG_FIELDS["top-p"])),
        "min-p": _value(_setting(config, *_CONFIG_FIELDS["min-p"])),
        "repeat": _value(_setting(config, *_CONFIG_FIELDS["repeat"])),
        "spec-type": spec_type or "none",
        "n-max": _value(llama.get("spec_nmax") or
                        _setting(config, *_CONFIG_FIELDS["n-max"])),
        "draft-kv": draft_kv_pair,
    }
    return result


def _selected(llamas, port=None):
    if port is None:
        return llamas[0] if llamas else {}
    wanted = str(port)
    return next((item for item in llamas if str(item.get("port")) == wanted), {})


def build_snapshot(gpus, cpu, memory, llamas, configs, processes, port=None):
    llama = _selected(llamas, port)
    key = str(llama.get("port", port))
    config = configs.get(key) or configs.get(llama.get("port")) or {}
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "llama": llama,
        "modelStats": model_stats(llama),
        "modelConfig": model_config(llama, config, port=key),
        "gpu": gpus,
        "cpu": cpu,
        "memory": memory,
        "power": {"watts": monitor.system_power(gpus, cpu)[0]},
        "processes": processes,
    }


class SnapshotStore:
    def __init__(self, port):
        self.port = str(port)
        self._lock = threading.Lock()
        self._snapshot = None
        self._published_at = None
        self._error = None

    def publish(self, snapshot):
        with self._lock:
            self._snapshot = snapshot
            self._published_at = time.monotonic()
            self._error = None

    def fail(self, error):
        with self._lock:
            self._error = str(error)

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def health(self):
        with self._lock:
            snapshot = self._snapshot or {}
            llama = snapshot.get("llama", {})
            return {
                "ok": True,
                "collector": "running",
                "snapshotAge": (None if self._published_at is None else
                                round(time.monotonic() - self._published_at, 3)),
                "error": self._error,
                "llama": {
                    "port": self.port,
                    "alive": bool(llama.get("alive")),
                    "stale": bool(llama.get("stale")),
                },
            }


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json(200, store.health())
                return
            snapshot = store.snapshot()
            if snapshot is None:
                self._json(503, {"error": "no snapshot available"})
            elif self.path == "/stats":
                self._json(200, snapshot)
            elif self.path == "/config":
                self._json(200, {"modelConfig": snapshot["modelConfig"]})
            else:
                self._json(404, {"error": "not found"})

        def _json(self, status, value):
            body = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    return Handler


def collect_snapshot(gpus, port):
    return build_snapshot(*monitor._collect(gpus, port), port=port)


def _collector(gpus, feeds, port, store, interval):
    try:
        while True:
            try:
                store.publish(collect_snapshot(gpus, port))
            except Exception as error:
                store.fail(error)
            time.sleep(interval)
    finally:
        for feed in feeds.values():
            feed.stop = True


def run(bind, http_port, llama_port, interval):
    gpus, feeds = monitor.discover_gpus()
    for feed in feeds.values():
        feed.start()
    store = SnapshotStore(llama_port)
    collector = threading.Thread(
        target=_collector, args=(gpus, feeds, str(llama_port), store, interval), daemon=True)
    collector.start()
    server = ThreadingHTTPServer((bind, http_port), handler_for(store))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        for feed in feeds.values():
            feed.stop = True


def main():
    parser = argparse.ArgumentParser(description="Serve llamagputop stats over HTTP")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--llama-port", default="8080")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    run(args.bind, args.http_port, args.llama_port, max(0.1, args.interval))


if __name__ == "__main__":
    main()

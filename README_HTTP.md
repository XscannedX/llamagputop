# llamagputop HTTP wrapper

`llamagputop_http.py` is a read-only HTTP wrapper around `llamagputop.py`. It exposes live hardware and llama.cpp information as JSON for another computer, dashboard, script, or browser on the network.

The wrapper is a separate process. It does not change the curses/TUI and does not scrape the TUI's rendered output.

## What it provides

- `GET /health` - wrapper and selected llama.cpp server status
- `GET /stats` - cached system data and selected model statistics
- `GET /config` - selected model configuration

The wrapper targets one llama.cpp server at a time. Configure that server with `--llama-port`.

## Requirements

- Linux, because the underlying monitor reads `/proc`, `/sys`, DRM, hwmon, and powercap data
- Python 3
- `llamagputop.py` and `llamagputop_http.py` in the same directory
- No third-party Python packages
- A running llama.cpp server for model data

For the most complete inference statistics, start llama.cpp with `--metrics`. The wrapper can still serve hardware data and partial model data without it.

The user running the wrapper must be able to read the relevant process and hardware data. If a value is unavailable, the API returns `null` or an empty value rather than pretending that the measurement is zero.

## Start the wrapper

Run it from the directory containing both Python files:

```bash
python3 llamagputop_http.py \
  --bind 0.0.0.0 \
  --http-port 4321 \
  --llama-port 7679
```

The options are:

| Option | Default | Meaning |
|---|---:|---|
| `--bind` | `127.0.0.1` | Local address on which HTTP listens |
| `--http-port` | `8765` | HTTP listening port |
| `--llama-port` | `8080` | One llama.cpp server to monitor |
| `--interval` | `1.0` | Collection interval in seconds; values below `0.1` are clamped to `0.1` |

For access from another machine, bind to the Fedora PC's LAN address or to all interfaces:

```bash
python3 llamagputop_http.py \
  --bind 0.0.0.0 \
  --http-port 4321 \
  --llama-port 7679 \
  --interval 1
```

Binding to `0.0.0.0` makes the service listen on every network interface. Prefer a specific trusted LAN address when possible.

Check the command-line options with:

```bash
python3 llamagputop_http.py --help
```

## Connect locally

From the Fedora PC:

```bash
curl -s http://127.0.0.1:4321/health
curl -s http://127.0.0.1:4321/stats
curl -s http://127.0.0.1:4321/config
```

With `jq` installed, format the responses:

```bash
curl -s http://127.0.0.1:4321/health | jq
curl -s http://127.0.0.1:4321/stats | jq
curl -s http://127.0.0.1:4321/config | jq
```

## Connect remotely

From another device on the same LAN, replace the address with the Fedora PC's LAN address:

```bash
curl -s http://192.168.1.111:4321/health | jq
curl -s http://192.168.1.111:4321/stats | jq
curl -s http://192.168.1.111:4321/config | jq
```

If the request times out, check these in order:

1. The wrapper is still running.
2. It was started with `--bind 0.0.0.0` or the Fedora PC's LAN address, not only `127.0.0.1`.
3. The Fedora firewall allows the HTTP port.
4. The client and Fedora PC can reach each other.
5. The selected llama.cpp port is correct. A wrong llama port affects model status, but should not prevent `/health` from responding.

## Endpoint reference

### `GET /health`

This checks the wrapper process and reports the state of the selected llama.cpp target.

Example:

```json
{
  "ok": true,
  "collector": "running",
  "snapshotAge": 0.4,
  "error": null,
  "llama": {
    "port": "7679",
    "alive": true,
    "stale": false
  }
}
```

Fields:

| Field | Meaning |
|---|---|
| `ok` | The HTTP wrapper process is alive |
| `collector` | Collector state; normally `running` |
| `snapshotAge` | Seconds since the last successful snapshot, or `null` before the first snapshot |
| `error` | Most recent collector error, or `null` |
| `llama.port` | Configured llama.cpp port |
| `llama.alive` | Whether the selected server responded during collection |
| `llama.stale` | Whether the monitor is retaining old data for the selected server |

The wrapper returns HTTP `200` when the wrapper itself is alive, even when the selected llama.cpp server is down. That makes `/health` useful as a process liveness check while the `llama.alive` field provides target status.

### `GET /stats`

Returns the latest cached snapshot. A request does not trigger a hardware or llama.cpp collection.

Top-level fields:

| Field | Meaning |
|---|---|
| `updatedAt` | UTC timestamp assigned when the snapshot was built |
| `llama` | Selected llama.cpp server data from the monitor |
| `modelStats` | Stable model statistics exposed by the wrapper |
| `modelConfig` | Stable model configuration exposed by the wrapper |
| `gpu` | GPU data from `llamagputop.py` |
| `cpu` | CPU data from `llamagputop.py` |
| `memory` | System memory data from `llamagputop.py` |
| `power` | Aggregated power data |
| `processes` | llama-related process data |

The `gpu`, `cpu`, `memory`, `power`, `processes`, and raw `llama` objects follow the structures returned by the current monitor and may gain fields as the monitor evolves. The `modelStats` and `modelConfig` sections are the intended stable wrapper interface.

#### `modelStats`

Example:

```json
{
  "modelStats": {
    "prefill": 24.5,
    "gen": 28.7,
    "session-avg": 26.1,
    "reasoning": "none",
    "draft-accepted-p": 0.72,
    "draft-accepted-tok": 1.72,
    "draft-accepted-total": 86
  }
}
```

| Field | Meaning | Source |
|---|---|---|
| `prefill` | Current prefill speed in tokens/second | `pp` |
| `gen` | Current generation speed in tokens/second | `tg` |
| `session-avg` | Session generation average | `tg_life` |
| `reasoning` | Reasoning format, or `none` | `reasoning_format` |
| `draft-accepted-p` | Speculative draft acceptance fraction from `0.0` to `1.0` | `spec` |
| `draft-accepted-tok` | Accepted draft tokens per speculative step | `tok_step` |
| `draft-accepted-total` | Cumulative accepted draft tokens | `spec_acc` |

A live rate may be `null` while the server is idle. `null` means that the monitor has no current measurement; it does not mean zero.

### `GET /config`

Returns only the configuration section:

```json
{
  "modelConfig": {
    "ctx": 131072,
    "ngl": 99,
    "flash-attn": "on",
    "threads": 8,
    "batch": "4096/1024",
    "slots": 1,
    "kv-k/v": "q5_1/q5_1",
    "temp": 0.6,
    "top-k": 20,
    "top-p": 0.95,
    "min-p": 0,
    "repeat": 1,
    "spec-type": "draft-mtp,ngram-map-k",
    "n-max": 3,
    "draft-kv": "q8_0/q8_0"
  }
}
```

| Field | Meaning |
|---|---|
| `ctx` | Context size |
| `ngl` | GPU layers |
| `flash-attn` | Flash attention setting |
| `threads` | CPU thread setting |
| `batch` | Batch configuration as reported by the monitor |
| `slots` | Number of llama.cpp slots |
| `kv-k/v` | Main KV cache K/V types, for example `q5_1/q5_1` |
| `temp` | Default temperature |
| `top-k` | Default top-k sampler value |
| `top-p` | Default top-p sampler value |
| `min-p` | Default min-p sampler value |
| `repeat` | Default repeat penalty |
| `spec-type` | Speculative decoding configuration |
| `n-max` | Maximum speculative draft length |
| `draft-kv` | Draft-model KV K/V types, for example `q8_0/q8_0` |

Configuration is read from the selected llama-server command line. The wrapper uses the monitor's normalized command-line parser first. For the four KV cache values, it also directly recognizes these aliases when the monitor version does not include them in its normalized output:

```text
-ctk   --cache-type-k
-ctv   --cache-type-v
-ctkd  --cache-type-k-draft  --spec-draft-type-k
-ctvd  --cache-type-v-draft  --spec-draft-type-v
```

Only explicitly available settings are reported. If a llama.cpp default cannot be verified from the process command line or monitor data, the field is `null` rather than an assumed default.

## HTTP status behavior

| Situation | Status | Body |
|---|---:|---|
| Wrapper alive and snapshot available | `200` | Endpoint JSON |
| `/health`, even if llama.cpp is unavailable | `200` | Health JSON with `llama.alive: false` |
| `/stats` or `/config` before first snapshot | `503` | `{"error":"no snapshot available"}` |
| Unknown path | `404` | `{"error":"not found"}` |

Only `GET` endpoints are implemented. The wrapper does not start, stop, or reconfigure llama.cpp.

## Fedora firewall

Open the port only to the LAN or VPN that needs access. Replace the source network with the correct one for your environment:

```bash
sudo firewall-cmd --permanent \
  --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="4321" protocol="tcp" accept'
sudo firewall-cmd --reload
```

Verify the rule:

```bash
sudo firewall-cmd --list-rich-rules
```

A broader temporary test rule is possible, but should not be left in place:

```bash
sudo firewall-cmd --add-port=4321/tcp
```

## Optional systemd service

For a persistent Fedora service, create `/etc/systemd/system/llamagputop-http.service` and adjust the paths and user:

```ini
[Unit]
Description=llamagputop HTTP wrapper
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=lines
WorkingDirectory=/home/lines/ai
ExecStart=/usr/bin/python3 /home/lines/ai/llamagputop_http.py --bind 0.0.0.0 --http-port 4321 --llama-port 7679
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Then enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llamagputop-http.service
```

Inspect status and logs:

```bash
systemctl status llamagputop-http.service
journalctl -u llamagputop-http.service -f
```

Use the Python path returned by `command -v python3` if `/usr/bin/python3` is not the interpreter that works with your installation. The service user must be able to read the llama-server command line and the hardware paths needed by `llamagputop.py`.

## Technical design

The wrapper is intentionally a thin adapter around the existing monitor.

### Startup

1. `argparse` reads the bind address, HTTP port, selected llama port, and collection interval.
2. `monitor.discover_gpus()` discovers GPUs and creates any optional hardware feeds.
3. Those feeds are started once.
4. A collector thread is started.
5. `ThreadingHTTPServer` listens for HTTP requests.

### Collection loop

The collector repeatedly calls:

```python
monitor._collect(gpus, llama_port)
```

That existing monitor function returns the structured collection tuple:

```text
(gpus, cpu, memory, llama_servers, configs, processes)
```

The wrapper selects the configured llama port, maps its fields into `modelStats` and `modelConfig`, adds the system sections, and publishes one JSON-ready snapshot.

The collector does not depend on `_LlamaFeed`. This keeps the wrapper compatible with monitor versions that expose `_collect()` but do not expose that optional private feed class. The existing synchronous collection path still performs the monitor's llama probing and hardware reads.

### Snapshot storage

`SnapshotStore` protects the current snapshot with a `threading.Lock`:

- The collector is the only writer.
- HTTP handlers are readers.
- Requests never mutate monitor state.
- A failed collection leaves the previous snapshot available and records the error for `/health`.
- The snapshot age is calculated from the last successful publish.

This prevents concurrent HTTP requests from racing with mutable monitor state such as probe caches, energy counters, and trend history.

### HTTP layer

The server uses Python's standard-library `ThreadingHTTPServer` and `BaseHTTPRequestHandler`:

- JSON is serialized with `json.dumps()`.
- Responses set `Content-Type: application/json` and `Content-Length`.
- Unknown paths return `404`.
- No third-party web framework is required.
- The server is read-only.

### Configuration mapping

The wrapper does not parse terminal output. It consumes the monitor's structured `_collect()` result and the monitor's command-line configuration parser. KV settings have a small direct `/proc/<pid>/cmdline` fallback because different `llamagputop.py` versions may not normalize every llama.cpp cache flag.

## Troubleshooting

### The collector thread reports `_LlamaFeed` is missing

Use the current wrapper version. The wrapper should call `_collect(gpus, port)` directly and must not require `_LlamaFeed`.

### `/health` works but `llama.alive` is false

Check that the selected port is the llama.cpp HTTP port, not the wrapper's HTTP port:

```bash
ss -ltnp | grep -E ':4321|:7679'
curl -s http://127.0.0.1:7679/health
```

Also check that the llama-server process is visible to the wrapper user.

### `/config` contains `null` values

The wrapper reports only values it can verify. Check the actual llama-server command line:

```bash
ps -ww -C llama-server -o pid=,args=
```

For KV settings, look for flags such as:

```text
-ctk q5_1 -ctv q5_1 -ctkd q8_0 -ctvd q8_0
```

If the process was launched by another user and `/proc/<pid>/cmdline` is unreadable, run the wrapper with an appropriate service user or permissions.

### Remote `curl` cannot connect

Test locally first:

```bash
curl -v http://127.0.0.1:4321/health
```

Then verify the listener:

```bash
ss -ltn | grep ':4321'
```

`127.0.0.1:4321` means only local clients can connect. For remote access, start with `--bind 0.0.0.0` or the Fedora machine's LAN address, then check `firewalld`.

### The wrapper starts but hardware fields are missing

Hardware fields depend on Linux permissions, driver interfaces, and optional helpers. The wrapper can still serve JSON when some readings are unavailable. Run the underlying monitor directly to compare output:

```bash
python3 llamagputop.py
```

The original monitor's README describes optional helpers such as `nvidia-smi`, `nvtop`, and `intel_gpu_top`.

## Testing

Run the wrapper tests:

```bash
python3 -m unittest tests.test_llamagputop_http -v
```

Run the full repository test suite:

```bash
python3 -m unittest discover -s tests
```

The wrapper tests cover JSON mapping, unavailable values, KV fallback parsing, cached HTTP responses, health responses, and unknown paths.

## Security limitations

The initial wrapper has no authentication, authorization, TLS, or request filtering. Do not expose it directly to the public internet. Use a trusted LAN/VPN, a restrictive firewall rule, or place it behind an authenticated reverse proxy if broader access is required.

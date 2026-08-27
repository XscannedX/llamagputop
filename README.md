llamagputop

This is a terminal monitor for Linux that tracks your GPUs and llama.cpp inference in a single Python file. It doesn't need any external dependencies beyond the standard library.
In order to work correctly and completely --metrics must be activated on your llama.cpp servers, chmod 755 and sudo will help with all the remaining metrics. Everything can work anyway (but with less data). 

```text
 llamagputop · 2 GPUs · llama: 5 servers       VRAM 14.8G free      RAM 16.2G free       142 W       14.2 t/s 

╭─ Intel · TigerLake-H GT1 [UHD Graphics] ──────────────────────────╮
│ util  ███░░░░░░░░░░░░░░░░   15%          clk 350/1450 MHz          │
│ vram  ███████░░░░░░░░░░░░   2104/15872 MiB                         │
│ power 12 W                                                         │
╰───────────────────────────────────────────────────────────────────╯
╭─ NVIDIA · GA106M [GeForce RTX 3060 Mobile / Max-Q] ───────────────╮
│ util  ████████████░░░░░░░   62%          clk 1425/7000 MHz         │
│ vram  ███████████████░░░░   4512/6144 MiB                          │
│ temp 68°C   power 85/115 W                                         │
╰───────────────────────────────────────────────────────────────────╯
╭─ llama processes ─────────────────────────────────────────────────╮
│ 1738957 GigaChat3.1-10B-A1.8B-q4_K RSS 6393M   VRAM — (CPU only)  │
│ 1730176 Ling-3.0-tiny-MXFP4_MOE    RSS 4209M   VRAM — (CPU only)  │
│ 1730591 gemma4-e4b-sft-claude-opus RSS 3542M   VRAM — (CPU only)  │
│ 1701115 bge-m3-clean-random-Q6_K   RSS 1272M   VRAM — (CPU only)  │
│ 1737341 bge-reranker-v2-m3-Q6_K    RSS  752M   VRAM — (CPU only)  │
╰───────────────────────────────────────────────────────────────────╯
╭─ power ───────────────────────────────────────────────────────────╮
│ Intel   ███░░░░░░░░░░░░░   12 / 45 W                               │
│ NVIDIA  ████████████░░░░   85 / 115 W                              │
│ CPU     ███████░░░░░░░░░   45 W                                    │
│ total   142 W    peak this session 180 W                           │
╰───────────────────────────────────────────────────────────────────╯
╭─ llama.cpp (1738957: 8080) ───────────────────────────────────────╮
│ status  generating  ctx 5120/slot · 10240 total (2 slots)  active 1 │
│ prefill 24 t/s last │ gen 28.7 t/s (med 28.7±0.3)                  │
│ kv ▎░░░░░░░░░░░  1.1%  115/10240 tok │ budget 107/150              │
╰───────────────────────────────────────────────────────────────────╯
╭─ llama.cpp (1730176: 8081) ───────────────────────────────────────╮
│ status    idle         ctx 8192     active 0, queued 0             │
╰───────────────────────────────────────────────────────────────────╯
╭─ trends ──────────────────────────────────────────────────────────╮
│ util %  ▂▃▃▄▅▆▇██████████████████████████████████████████████████ │
│ gen t/s ░░░░░░░░░░░▂▂▃▄▅▆▇███████████████████████████████████████ │
╰───────────────────────────────────────────────────────────────────╯
```

It watches every GPU in your machine including AMD, Intel, and NVIDIA, along with your CPU, RAM, and power draw. What sets this tool apart from others is the llama.cpp panel. It automatically detects every running llama server and shows you the metrics that actually matter when serving a model. You can see your prefill and generation speed with live medians and standard deviations. A speed is shown live only while a request is actually running; once it finishes, the rate it ran at is kept and marked "last" rather than left standing as if it were the current one. Start your server with --metrics and the completed-request figures come from the server's own counters, which is the exact number it timed itself. It also shows you your KV cache fill, aggregated across every slot, and it reads every slot rather than the first — a server started with -np 2 hands a request to whichever slot is free. If you use speculative decoding, it tells you which draft head is in use, its acceptance rate, the tokens per step, and the per position acceptance. These inference statistics are tied directly to the model, meaning they reset automatically when the model changes.

I designed this tool to read everything directly from the source. Data is pulled from sysfs, proc, and the driver. It can use optional helpers like intel gpu top or nvidia smi only if they are present on your system. It never blindly trusts a sensor. Any out of range readings are dropped and it just keeps the last good value. If a card, a tool, or a counter is missing, it honestly shows a dash with a reason instead of giving you a fake zero — a server started without --metrics has no prefill counter, and that is not the same thing as a prefill of zero. Inference is read on its own thread, so a server busy enough to stop answering slows nothing down on screen: the rest of the panel keeps its once-a-second refresh and the stalled server says so.

It features a summary strip at the top for quick glances at your generation speed, free VRAM, and total watts. Below that, it generates trend sparklines so you can see exactly when something changed. Each one is a strip chart: one column per refresh, newest on the right, sliding left as time passes. The scale is fitted to the range the data actually occupies rather than anchored at zero, so a metric living in a narrow band still shows its shape, and it is snapped to a round step so the graph scrolls instead of flickering. Every row carries the peak for the whole session, and a bar that does not start at zero says where its floor is. Percentages keep their absolute scale. A dot means a true zero and nothing else. The script enumerates your DRM cards and identifies the driver and vendor, bringing you all the deep hardware metrics like temperatures, VRM voltages, clocks, and GTT versus VRAM eviction data. 

Every server is found just by scanning the process list, so there are no ports to configure. If you run multiple servers, they each get their own dedicated panel automatically. The process list intelligently analyzes your active models and distinguishes between those running on the GPU and those running strictly on your CPU, explicitly labeling CPU-only RAM usages so you are never left guessing where your memory went. It even has a server config section that reads the settings from the command line so you know exactly how the model was loaded — every flag it was launched with, not a curated subset, so DRY, XTC, mirostat, pooling, LoRA and anything a newer llama.cpp adds all show up. API keys are masked. Sampler values there are the server's defaults, which a request carrying its own overrides, and the panel says so. 

For your CPU and RAM, it tracks utilisation, temperatures, frequency, RAPL power, and swap destinations like zram versus disk. The power section aggregates every readable watt meter to give you the cumulative energy drawn during your session and the energy efficiency of your models. 

The tool is completely portable. Since it discovers hardware at runtime, it runs on any Linux machine. 

To use it, just run the python script in your terminal. You can pass a port number to focus on a single server, or use the once or line flags for scripting and logging. The TUI is fully interactive. You can quit with q, scroll with your arrow keys, change the refresh rate with plus and minus, and reset the history with z.

If you want to track your CPU power, make sure your kernel allows reading the RAPL counter. You might need to add a udev rule to grant your user access to the powercap sysfs directory.

Copyright 2026 XscannedX. MIT License.

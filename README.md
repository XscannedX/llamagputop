llamagputop

This is a terminal monitor for Linux that tracks your GPUs and llama.cpp inference in a single Python file. It doesn't need any external dependencies beyond the standard library.
In order to work correctly and completely --metrics must be activated on your llama.cpp servers, chmod 755 and sudo will help with all the remaining metrics. Everything can work anyway (but with less data). 

```text
 llamagputop · 2 GPUs · llama: 5 servers                     14:23:07 

╭─ Intel · TigerLake-H GT1 [UHD Graphics] ──────────────────────────╮
│ util █▊░░░░░░░░░░░░  12% │ avg 9% │ core 350 MHz                   │
│ range 350–1450 eff 600 MHz                                         │
│ no data vram — memory is shared with system RAM                    │
│         temp, power — no hwmon node on this card                   │
╰───────────────────────────────────────────────────────────────────╯
╭─ NVIDIA · GA106M [GeForce RTX 3060 Mobile / Max-Q] ───────────────╮
│ util ████████▋░░░░░  62% │ avg 41% │ mem 18%                       │
│ vram ██████████▎░░░ 4512/6144 MiB │ gpu 68.0°C │ core 1425.0 MHz   │
│ vmem 7000.0 MHz │ power 85/115 W                                   │
╰───────────────────────────────────────────────────────────────────╯
╭─ llama processes ─────────────────────────────────────────────────╮
│ 1738957 GigaChat3.1-10B-A1.8B-q4_K RSS 6393M   VRAM — (CPU only)  │
│ 1730176 Ling-3.0-tiny-MXFP4_MOE    RSS 4209M   VRAM — (CPU only)  │
│ 1730591 gemma4-e4b-sft-claude-opus RSS 3542M   VRAM — (CPU only)  │
│ 1701115 bge-m3-clean-random-Q6_K   RSS 1272M   VRAM — (CPU only)  │
│ 1737341 bge-reranker-v2-m3-Q6_K    RSS  752M   VRAM — (CPU only)  │
╰───────────────────────────────────────────────────────────────────╯
╭─ power ───────────────────────────────────────────────────────────╮
│ NVIDIA 85/115 W │ CPU 45 W                                         │
│ total 130 W  avg 118 W  peak 180 W                                 │
│ session 1h04m  energy used 126.1 Wh                                │
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

It watches every GPU in your machine including AMD, Intel, and NVIDIA, along with your CPU, RAM, and power draw. What sets this tool apart from others is the llama.cpp panel. It automatically detects every running llama server and shows you the metrics that actually matter when serving a model. You can see your prefill and generation speed with live medians and standard deviations. A speed is shown live only while a request is actually running; once it finishes, the rate it ran at is kept and marked "last" rather than left standing as if it were the current one. Start your server with --metrics and the completed-request figures come from the server's own counters, which is the exact number it timed itself. The time to first token is read from those same counters rather than clocked here, because the server knows it exactly while polling would carry the refresh interval as its error bar; when several slots finished inside one interval the figure is a summed prefill time and is named as such, since a sum of waits is not a wait. It also shows you your KV cache fill, aggregated across every slot, and it reads every slot rather than the first — a server started with -np 2 hands a request to whichever slot is free. If you use speculative decoding, it tells you which draft head is in use, its acceptance rate, the tokens per step, and the per position acceptance. These inference statistics are tied directly to the model, meaning they reset automatically when the model changes.

I designed this tool to read everything directly from the source. Data is pulled from sysfs, proc, and the driver. It can use optional helpers like intel gpu top or nvidia smi only if they are present on your system. It never blindly trusts a sensor. Any out of range readings are dropped and it just keeps the last good value. If a card, a tool, or a counter is missing, it honestly shows a dash with a reason instead of giving you a fake zero — a server started without --metrics has no prefill counter, and that is not the same thing as a prefill of zero. That holds for the cards too, and the reason is written for the thing that is actually missing. Where the answer is a program, it names the BINARY and not a package, since package names differ between distributions while binaries do not: a machine without nvidia-smi reads "no data · util, vram, temp, power, clocks — needs nvidia-smi" instead of an empty panel. On AMD, though, almost everything comes from sysfs, so naming a binary would send you after a package that changes nothing; there the reason names the driver attribute or the missing node instead, as in "gpu_busy_percent not exposed by this driver" or "no power1_average on this hwmon node". An integrated GPU says its memory is the system's rather than drawing a bar against the whole of RAM, and it knows which it is by measurement rather than by its name — a discrete Arc is an Intel card with real VRAM, so what decides it is whether the total reported for the card is the machine's own RAM total. The same care applies in the other direction, to readings that are genuinely zero: a card in zero-fan mode reports 0 rpm and is shown as stopped rather than as having no tachometer, an idle GPU reads 0% and says so, and 0 watts is a measurement while an unreadable meter is a dash. Inference is read on its own thread, so a server busy enough to stop answering slows nothing down on screen: the rest of the panel keeps its once-a-second refresh and the stalled server says so.

It features a summary strip at the top for quick glances at your generation speed, free VRAM, and total watts. Below that, it generates trend sparklines so you can see exactly when something changed. Each one is a strip chart: one column per refresh, newest on the right, sliding left as time passes. The scale is fitted to the range the data actually occupies rather than anchored at zero, so a metric living in a narrow band still shows its shape, and it is snapped to a round step so the graph scrolls instead of flickering. Every row carries the peak for the whole session, and a bar that does not start at zero says where its floor is. The KV cache has a row of its own on generative servers, and unlike the speed rows it keeps its real value while the server is idle: a slot that is not working still holds its conversation, so recording a zero there would draw a collapse that never happened. Percentages keep their absolute scale. A dot means a true zero and nothing else. The script enumerates your DRM cards and identifies the driver and vendor, bringing you all the deep hardware metrics like temperatures, clocks, core voltage, and GTT versus VRAM eviction data. On AMD the VRM figures are temperatures and are shown as such: the binary metrics blob reports them in degrees, its own voltage fields read as unsupported on the cards checked, and the real core voltage comes from hwmon. 

Every server is found just by scanning the process list, so there are no ports to configure. If you run multiple servers, they each get their own dedicated panel automatically. The process list intelligently analyzes your active models and distinguishes between those running on the GPU and those running strictly on your CPU, explicitly labeling CPU-only RAM usages so you are never left guessing where your memory went. On AMD and Intel that split comes from the kernel, which reports device memory separately from the host-side spill; NVIDIA does not expose those counters at all, so there the figure comes from nvidia-smi and the spill is shown as unknown rather than as zero, because one number for memory on the card cannot be divided into two. It even has a server config section that reads the settings from the command line so you know exactly how the model was loaded — every flag it was launched with, not a curated subset, so DRY, XTC, mirostat, pooling, LoRA and anything a newer llama.cpp adds all show up. API keys are masked. Sampler values there are the server's defaults, which a request carrying its own overrides, and the panel says so. 

For your CPU and RAM, it tracks utilisation, temperatures, frequency, RAPL power, and swap destinations like zram versus disk. The power section aggregates every readable watt meter to give you the cumulative energy drawn during your session and the energy efficiency of your models. 

The tool is completely portable. Since it discovers hardware at runtime, it runs on any Linux machine. 

The tests live in `tests/` and use the standard library only, like the program: run them with `python3 -m unittest discover -s tests`. They need no particular hardware and touch no network, since every reader is driven against a temporary directory standing in for sysfs. Most of them come in pairs, because the difficult half of "a missing reading is a dash" is proving that a real zero still reads as zero: a parked fan, a power-gated GPU, an idle server and a freshly started card are all genuinely at zero and have to survive.

To use it, just run the python script in your terminal. You can pass a port number to focus on a single server, or use the once or line flags for scripting and logging. The TUI is fully interactive. You can quit with q, scroll with your arrow keys, change the refresh rate with plus and minus, and reset the history with z.

If you want to track your CPU power, make sure your kernel allows reading the RAPL counter. You might need to add a udev rule to grant your user access to the powercap sysfs directory.

Copyright 2026 XscannedX. MIT License.

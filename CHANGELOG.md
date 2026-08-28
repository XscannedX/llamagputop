# Changelog

All notable changes to this project will be documented in this file.

Versions have four components, `major.minor.patch.hotfix`, and releases so far move the
third. The scheme was widened from three components and applied to the whole history, so a
version read elsewhere as `1.6.0` is `1.0.6.0` here. No tag or release referenced the old
numbers.

Dates are the day the work was done, read from the commit history rather than chosen. The
one soft edge is 1.0.0.0, where a compatibility commit sits between it and 1.0.1.0 without
belonging to either, so its date is the repository's first day rather than a release event.

## Version 1.0.7.0 (2026-08-28)

### Added

* Per-process VRAM on NVIDIA cards. The panel reads `drm-memory-vram` from `/proc/PID/fdinfo`, which amdgpu and i915 export and the NVIDIA driver does not, so every CUDA process used to show a dash. nvidia-smi already runs once a second here for the card-level stats, so the per-process figure costs one more query on a thread that is already awake. The source of the number is recorded rather than blended, because the two do not answer the same question: fdinfo separates device memory from the host-side spill, which is the point of that column, while nvidia-smi reports a single figure for memory on the card. On NVIDIA the spill is therefore shown as a dash with that reason instead of a zero, since a zero would claim "nothing spilled" about something nobody measured. A reading older than six seconds is refused rather than shown, because a value from a process that has since exited is not a current value.

* Time to first token for the request that just finished, as the server timed it. The figure was already being computed and discarded: the prefill rate is prompt tokens over prompt seconds, and that denominator is the wait before the first token. Timing it here by polling would carry the poll interval as its error bar, which is a fifth of a two-second prefill, and manufacturing precision while the exact figure is already in hand is the defect this file fixed for prefill speed one release earlier. The label follows the meaning instead of carrying a footnote: with one slot working the interval covers one request and the number is that request's time to first token, while with several the counters advanced for several requests and the sum is a total prefill time, which is what it is then called.

* A trend row for the KV cache. It was the one number in the llama panel with no history, so the question of when it filled up had no answer. It does not copy the two speed rows beside it, and the difference matters: those record a zero while the server is idle, which is true of a speed and false of a cache, because a slot that is not working still holds its conversation. Only generative servers get the row, decided from the same role grouping the config panel already builds.

* Every empty field on a GPU panel now says why it is empty, which the README promised for the whole program while only the llama panels delivered it. The reason is written for the thing that is actually missing. Where the answer is a program it names the binary rather than a package, since the same tools ship under different package names across distributions while the binary stays put. On AMD, where almost everything comes from sysfs, naming a binary would send someone after a package that changes nothing, so the reason names the driver attribute or the missing node instead.

* The port argument focuses on one server, which is what the help text has always claimed. It used to only append: nothing filtered the list, so asking for one port on a machine running four still drew four panels. Both halves of that claim hold together, so nothing had to be traded away: the requested port is filtered to, and it is still probed when discovery did not find it. Focusing a port that is not answering now says so on the status line, where before the line said nothing at all about llama and silence could not be told apart from an ignored argument.

* A test suite, in `tests/`, standard library only so the program keeps its promise of needing nothing installed. It needs no particular hardware and opens no socket: every reader is driven against a temporary directory standing in for sysfs, or against a stub feed. Almost every case comes in a pair, because the difficult half of "a missing reading shows as a dash" is proving that a real zero still reads as zero, and a suite that checked only the first half would pass a change that turns every zero into a dash. Each fixture also carries a control proving it can express presence, since a fixture that quietly produces nothing makes every assertion about an empty field pass for the wrong reason.

### Fixed

* nvtop readings were matched to a card by looking for a vendor word in its name, and integrated graphics carry none of them, so an integrated GPU had its utilisation thrown away while it sat there for the taking. Matching is now on the normalised name, with the vendor test kept as a fallback for a card whose two sources genuinely disagree.

* An integrated GPU no longer draws a VRAM bar against the whole of system RAM. The guard for that rested on the idea that nvtop reports no used figure for integrated graphics, which turns out to be false at least some of the time: it can answer with a real used figure alongside a total that is exactly what `/proc/meminfo` reports as MemTotal. The test is now the total rather than the used figure, and it is a measurement rather than a guess at the card's nature, since a discrete Arc is an Intel card with real memory of its own and keeps its bar.

* Two defects in the AMD `gpu_metrics` decoder, both found the first time that branch met real hardware. Offsets 10 and 12 were labelled voltages and are temperatures, printed for as long as the branch existed as millivolts; in `gpu_metrics_v1_3` they are `temperature_vrgfx` and `temperature_vrsoc`, in degrees Celsius. And a throttle reading of zero was treated as a missing field, which sent every healthy card down a fallback path where a different offset was read as a throttle bitmask, inventing a reason to worry about a card that was fine.

* The AMD reader answered zero for a file that is not there. `read_int` defaults to 0, and six readings took it bare, so a card whose driver does not publish `gpu_busy_percent` reported 0 % utilisation and one without the memory counters reported 0 MiB: not a dash, but a full empty bar drawn at zero, which is the one thing the file's own sanity guard forbids.

* A stopped fan is readable again. Cards park their fans below a temperature threshold, so 0 rpm is the ordinary idle state and not an absence, and folding it into the same value as "no fan sensor" hid the row completely. The same line stood in both the AMD and the Intel readers, and the panel compounded it by guarding the row on truthiness.

* An idle Intel GPU's real 0 MHz is readable for the same reason. `rps_act_freq_mhz` is the actual frequency and a power-gated card sits at zero, while the setpoint files beside it report something else entirely. Four layers agreed in hiding it: a floor of 1 refused the reading, a trailing fallback would have killed it anyway, the read had no explicit default so an absent file also produced zero, and the panel guarded the row on truthiness. The missing default is why the floor could not simply be lowered.

* A held sensor reading now expires. Holding the last good value across one bad tick is the point of that mechanism; holding it forever is a different thing. With no age at all, one in-range reading meant a sensor that later died kept replaying that number for the rest of the run, styled exactly like a live one, which on a temperature display is the worst way to fail because the number still looks reasonable. The same omission stood on the GPU and CPU wattage caches, where the timestamp was already being kept and never consulted.

* A wrapped energy counter printed as 0 W. The counter only grows until it wraps, and clamping the negative delta to zero produced a fabricated reading indistinguishable from a card drawing nothing. The CPU path can correct the same event because intel-rapl publishes its range; hwmon declares no equivalent, so the wrap cannot be undone and the honest move is to keep the last good figure for one tick and re-baseline.

* The `--once` and `--line` output printed a fabricated zero where the panel printed a dash, and it was one function holding two standards: its llama half already routed every rate through the helper that prints a dash, while its GPU and CPU half printed a bare fallback to zero. Utilisation, power, CPU utilisation and the memory pair are all routed through the same helper now, so a missing reading is a dash and a real zero is still a zero.

* The trend panel charted zeros nobody measured, against a contract it states itself and the README repeats word for word. The recording helper already carried the right rule, that a bad sample holds the previous one because one column is one tick, and the call sites disabled it by converting an absent reading to zero before the helper could see it. Two of those sites were not the same case and were left alone: a server that is alive and idle genuinely produces nothing. What was wrong there is narrower, since a server that stops answering keeps its alive flag through the retry ladder while its rate is forced to absent, so a silence charted as a measured zero.

* An empty series is no longer seeded with a zero, and a retired server's KV history is now dropped with the rest of its series rather than left in memory for the life of the process.

* A sysfs clock reading was computed and discarded on every Intel card whose engine feed was alive, because `setdefault` fills a key that is absent and the line above it inserts the key whatever the feed reported. A key being present is not the same fact as a key having a value.

* Having an hwmon node is not having its files, and the reason-for-an-empty-field mechanism missed that on its first day. The check covered the case where the whole node is absent and assumed the rest followed; partial hardware is the shape that slips through a check written for missing hardware.

* Two context figures that differ do not always differ for the same reason, and calling both cases per-slot versus total printed a part larger than the whole. With several slots the split is the reason and the two multiply out; with one slot what is left is the server rounding the request up to its own block size, so the allocated figure leads and the requested one is named as such.

* The two sources for how many requests are running now say so when they disagree, instead of one being collected and never shown. The count from `/slots` is preferred because it needs no `--metrics` and cannot contradict the phase drawn beside it, while the metrics gauge follows the slot's release rather than its work and can lag behind a request that has already returned.

* The README's example panel was rewritten. It still showed an integrated GPU with a VRAM bar against half of system RAM, which is the first thing a reader sees and was the last place the old behaviour survived. Three other lines in the same picture were things the program does not print: a summary strip across the title row, which actually holds the title and a clock; a combined clock cell, where the panel prints core and memory separately; and a power section drawn as bars, where it prints plain cells and a total.

## Version 1.0.6.0 (2026-08-26)

### Fixed

* A non-finite reading can no longer kill the interface. `float()` accepts the strings "nan" and "inf" without raising, and three readers parse external tools that way — intel_gpu_top, nvtop and nvidia-smi — so a sensor printing either put it straight into the history; the nvidia-side helper only catches ValueError, which neither raises. Downstream, a NaN raised ValueError inside the sparkline and an infinity raised OverflowError in the step ladder, and both take the whole TUI down. `record()` is the only door into a series and now refuses them, holding the previous sample so a fixed-cadence series keeps one column per tick; the sparkline and the step ladder are defensive as well. The `sane()` guard that exists for exactly this class of problem did not cover the path: it is applied to clocks, temperatures, watts and voltage, not to utilisation or VRAM, which are what gets charted.
* Reading the reasoning format is guarded again. The guard was there, was removed in the 1.0.2.0 rewrite because the slot variable is now seeded to an empty object, and that covers a failed `/slots` but not a `/slots` that answers with a list whose items are not objects — there the attribute lookup raises and takes the whole sample with it, so a server that is merely odd was drawn as one that is not answering.
* A negative `n_ctx` from a slot is refused instead of stored, where it would have produced a negative KV percentage.
* The retry counter is capped. The ladder computes two to the power of the failure count and then clamps the result to five seconds; on a long outage the count grew without bound, so the exponent did too — a several-hundred-digit integer built and discarded on every attempt.

### Removed

* Dead code found by pyflakes and ruff: an unused `socket` import and a redundant module-level `urllib.error` import (the request helper imports it locally and uses it there), an `intel_cards` list that was built and never read, and an `if True:` wrapper left behind when the settings parser was split in 1.0.3.0. The file is now clean under pyflakes.

## Version 1.0.5.0 (2026-08-26)

### Fixed

* The peak is back on every trend row, and it is the whole session's. Version 1.0.4.0 replaced it with the bar's own low-to-high range, which was a bad trade: on a speed series the floor is almost always zero because idle records zero, so `0.0…35.9` was `peak 35.9` with noise in front of it, and the one number worth keeping across a long run is the highest the metric ever reached. A row whose bar does NOT start at zero now says where its floor is, as `base N`, since that is the case the range was there for.

### Changed

* Trend rows are strip charts: one column per tick, newest on the right, sliding left by exactly one column. They used to fold the WHOLE history into the available columns, so each column stood for a bucket whose boundaries moved as the history grew — the picture did not scroll, it re-quantised in place. Measured over 900 ticks, using "is this frame the previous one shifted left by one column" as the definition of scrolling: the old version satisfied it 7.9% of the time and 60.8% of its frames were byte-identical to the one before, which is what a graph that appears frozen and then jumps actually looks like. The new one satisfies it between 94% and 99% depending on the series.
* The fitted scale is snapped to a 1-2-5 step. Fitting to the visible window keeps the detail — a series in a narrow band uses seven of the nine glyphs instead of three — but recomputing the exact minimum and maximum every tick moves the mapping constantly. Snapping both ends holds it still until the range really changes: 91.3% to 98.7% scrolling on the same run, with no loss of detail. Scaling against the whole session's range was measured as an alternative and rejected: it scrolls just as well (98.9%) and collapses the same series to three glyphs, because one excursion half an hour ago flattens the view of now.
* A drifting auto-range was tried first and measured WORSE than doing nothing — 82.2% scrolling, and 56.6% on a bursty series — because a bound that creeps a little every tick is the flicker itself, not the cure. Recorded here so the idea is not tried again.
* The trend panel states its resolution (`1s per column`) and that the peak covers the whole session. A window that does not say how much time it spans cannot be read.
* Per-series bucket aggregation is gone. It existed to choose which extreme survived a bucket; with one sample per column there are no buckets.

## Version 1.0.4.0 (2026-08-26)

### Changed

* Trend sparklines are fitted to the range the data actually occupies instead of being anchored at zero. A series living in a narrow band high above zero had all its detail crushed into the top glyph, which is the same as not drawing it. Measured over 46 ticks before the change: `ram free` moved 454 MiB and used TWO of the nine glyphs — a straight line across a real change — while `power` used seven and `gen t/s` three. After: `ram free` uses five and the shape of the drop is visible. Percentages keep their absolute 0..100 scale, because drawing 5% at full height would lie about the quantity.
* Every fitted row prints the low and high end it was drawn against. Fitting to the window is not free: a 3 MiB wobble and a 400 MiB drop fill the same height, so the bar carries the shape and the numbers beside it carry the magnitude. Absolute rows print the peak instead, since a "0…100" next to a label ending in "%" says nothing.
* A dot means a true zero and nothing else. On a fitted row the bottom of the range is a low bar, because a dot there reads as "nothing happened" when what happened is "the lowest value in this window".
* Each series declares which extreme its buckets keep. A history longer than the bar has to fold many samples into one column, and keeping the maximum preserves bursts while erasing dips — right for utilisation, power and tokens a second, wrong for free memory, where the event IS the drop. Verified on a 3600-sample series with a single-tick collapse: kept by the minimum, invisible to the maximum.
* A completed prefill now appears in the trend. The series was fed only from the live rate, and a prefill on a GPU is over before the second tick (4001 tokens in 1.94 s, measured), so the line stayed at zero, peaked at zero, and was dropped by the "only chart a speed that has actually occurred" test — invisible on exactly the servers fast enough to be worth watching. It is charted as one spike per completed prefill, at the rate the server timed.

## Version 1.0.3.0 (2026-08-26)

### Added

* The server config panel shows every flag the model was launched with. It used to show a hardcoded subset, so a run started with `--dry-multiplier`, `--pooling`, `--jinja`, `--load-mode` or anything else outside that list had no line anywhere and nothing said so. Measured across four running servers: between six and eight settings displayed out of fifteen to eighteen flags actually passed. The themed groups are now the ORDER and the LABELS rather than the filter, and whatever they do not name lands in a final `other` group computed as the complement of what the groups consumed — so a flag added by a future llama.cpp release still appears, without anyone editing this file. Verified at 64 flags across those four servers with none dropped, and on a synthetic command line carrying the full DRY and XTC families, mirostat, rope/yarn, LoRA and two invented flags.
* Two new groups: `role` (pooling, embedding, reranking — what the server was started to do) and `server` (host, port, path, threads-http, metrics, slots and props endpoints, TLS).
* The sampling group carries a line saying those values are the server's DEFAULTS. A client that puts temperature or a penalty in the request body overrides them, and the command line cannot show that. The line appears only when sampling flags are actually present.
* API keys are masked. Showing every flag verbatim would otherwise print `--api-key` on a screen this tool is meant to be watched on.

### Fixed

* A negative value is no longer mistaken for the next flag. `-np -1` was read as a bare switch, so a server running with automatic parallelism displayed `slots: on`. A token starting with `-` now begins a new flag only when it is not a number.
* A negating switch is labelled by its concept rather than by the flag, so `--no-warmup` reads `warmup off` instead of `no-warmup off`, which said the opposite.
* Batch and micro-batch are labelled separately instead of being packed into one `4096/1024` cell.

## Version 1.0.2.0 (2026-08-26)

### Fixed

* Generation speed no longer decays to a false value and freezes there when a request ends. llama.cpp leaves the finished request's decoded-token count standing in the slot, so the rolling window kept sliding past the end of the work: the reading fell away and then stopped updating altogether. Measured against a server whose own timings reported 29.93 tokens a second, the panel walked down through 29.7, 27.8, 24.7, 21.1, 17.7, 14.2, 12.7, 9.4, 5.4 and settled on 1.606, where it stayed indefinitely. The live rate is now measured only while a slot is actually working, and the rate of the request that just finished is kept and labelled as such instead of being shown as the current one.
* Prefill speed is measured again. It had exactly one source, the cumulative counters, which a server started without --metrics does not expose; the endpoint answers 501 with a JSON body and the tool read that as "every counter is zero" rather than "there are no counters". The prefill figure was therefore a literal zero on every server, always. Servers with --metrics now report the completed-request prefill exactly (2068 against the server's own 2065.45), and a prefill slow enough to be sampled is also tracked live.
* Every slot is read, not just the first one. A server started with -np N hands a request to whichever slot is free, so reading slot 0 alone made the panel report "idle" while the server was generating on another slot, and show that slot's leftover decoded count. Confirmed on a two-slot server: a request ran to completion at 29.88 tokens a second while slot 0 sat idle holding a stale count. Generation, prefill, KV fill and the active-request count are now aggregated across all slots, and two concurrent requests read 39.4 against a true 39.60 aggregate.
* The context no longer resets to zero. Server facts — context size, slot count, model — describe the server rather than the current tick, but the sample was rebuilt from defaults on every pass, so a single failed or timed-out request published "ctx 0" along with an empty KV bar. Those facts now carry forward from the last good reading and a bad tick is marked stale instead of blanked.
* The unresponsive-server backoff can actually fire. It caught urllib.error.URLError while the request helper raised a bare TimeoutError, which is not a subclass of it, so a timeout fell through to a blanket handler on both the counters and the slots endpoint. The branch that worked was unreachable and the branch that ran was the broken one.
* Prompt reuse is no longer reported for builds that do not export the cache counter. Fifteen counters on one build against ten on another was being read as a measured zero.

### Changed

* The llama.cpp panel is read on its own thread, where every other reader in the tool already lives. It was the only one left on the draw path, which is why its numbers lagged behind the GPU, CPU and power figures while those kept ticking once a second. A server doing a prefill on CPU can stop answering entirely — measured, three consecutive twenty-second timeouts during a seventy-eight second prefill, with nothing at all under the old 1.2 second limit — so no timeout setting could have kept the panel live from the draw loop. The collection pass now stays around 120 milliseconds and never exceeded 411 across fifty-one measured ticks.
* Servers are probed concurrently. Read one after another, a single server that stops answering costs every other server its own timeout.
* The retry after a failed probe starts immediately and backs off geometrically to five seconds, replacing a flat sixty. A minute was both too long a penalty for one slow answer and no help at all against a server that will be silent for longer than that.
* Context is shown as allocated rather than inferred. The slots endpoint reports the per-slot size while the command line carries the total, and with -np 2 the two differ by a factor of two under the same word; the panel now shows both when they disagree. The total is read from -c, because deriving it as per-slot times slot count is right on one build and invents a number on another — a reranking server allocated 4096 reports 4096 across four slots, which that arithmetic turns into 16384.
* A speed with no source shows a dash and the reason, matching what the tool already did for a missing card or sensor. A number that nothing measured is not a zero.

### Removed

* A leftover debug write that appended to a file on the Desktop on every tick for every server.

## Version 1.0.1.0 (2026-08-25)

### Added
* Dynamic executable detection. The user interface now automatically detects the executable name (e.g., custom builds or forks) and uses it for highly accurate labeling instead of hardcoding the server name.
* Real time reasoning format detection. The system parses the slots endpoint to automatically identify the reasoning format during runtime. This completely eliminates the need for manual command line arguments.

### Changed
* Improved user interface responsiveness. A robust sixty second backoff mechanism was introduced. When a server becomes unresponsive due to high CPU loads, the interface displays a live countdown timer instead of freezing. This ensures smooth performance across the entire fleet.
* Enhanced KV cache user interface. The sparkline now displays both the absolute used and total token counts in addition to the percentage for better clarity.

### Fixed
* Full backward compatibility with older server builds.
* Graceful handling of HTTP 501 Not Implemented errors when the metrics endpoint is missing or disabled.
* Accurate generation phase detection utilizing the raw integer state when the modern boolean flag is unavailable.
* Reliable KV cache calculations utilizing decoded token counts when the prompt cache metrics are missing.

## Version 1.0.0.0 (2026-08-21)

### Added
* Initial release of the monitoring tool.
* Core terminal user interface for live tracking of GPU usage and fleet status.

# Changelog

All notable changes to this project will be documented in this file.

## Version 1.6.0

### Fixed

* A non-finite reading can no longer kill the interface. `float()` accepts the strings "nan" and "inf" without raising, and three readers parse external tools that way — intel_gpu_top, nvtop and nvidia-smi — so a sensor printing either put it straight into the history; the nvidia-side helper only catches ValueError, which neither raises. Downstream, a NaN raised ValueError inside the sparkline and an infinity raised OverflowError in the step ladder, and both take the whole TUI down. `record()` is the only door into a series and now refuses them, holding the previous sample so a fixed-cadence series keeps one column per tick; the sparkline and the step ladder are defensive as well. The `sane()` guard that exists for exactly this class of problem did not cover the path: it is applied to clocks, temperatures, watts and voltage, not to utilisation or VRAM, which are what gets charted.
* Reading the reasoning format is guarded again. The guard was there, was removed in the 1.2.0 rewrite because the slot variable is now seeded to an empty object, and that covers a failed `/slots` but not a `/slots` that answers with a list whose items are not objects — there the attribute lookup raises and takes the whole sample with it, so a server that is merely odd was drawn as one that is not answering.
* A negative `n_ctx` from a slot is refused instead of stored, where it would have produced a negative KV percentage.
* The retry counter is capped. The ladder computes two to the power of the failure count and then clamps the result to five seconds; on a long outage the count grew without bound, so the exponent did too — a several-hundred-digit integer built and discarded on every attempt.

### Removed

* Dead code found by pyflakes and ruff: an unused `socket` import and a redundant module-level `urllib.error` import (the request helper imports it locally and uses it there), an `intel_cards` list that was built and never read, and an `if True:` wrapper left behind when the settings parser was split in 1.3.0. The file is now clean under pyflakes.

## Version 1.5.0

### Fixed

* The peak is back on every trend row, and it is the whole session's. Version 1.4.0 replaced it with the bar's own low-to-high range, which was a bad trade: on a speed series the floor is almost always zero because idle records zero, so `0.0…35.9` was `peak 35.9` with noise in front of it, and the one number worth keeping across a long run is the highest the metric ever reached. A row whose bar does NOT start at zero now says where its floor is, as `base N`, since that is the case the range was there for.

### Changed

* Trend rows are strip charts: one column per tick, newest on the right, sliding left by exactly one column. They used to fold the WHOLE history into the available columns, so each column stood for a bucket whose boundaries moved as the history grew — the picture did not scroll, it re-quantised in place. Measured over 900 ticks, using "is this frame the previous one shifted left by one column" as the definition of scrolling: the old version satisfied it 7.9% of the time and 60.8% of its frames were byte-identical to the one before, which is what a graph that appears frozen and then jumps actually looks like. The new one satisfies it between 94% and 99% depending on the series.
* The fitted scale is snapped to a 1-2-5 step. Fitting to the visible window keeps the detail — a series in a narrow band uses seven of the nine glyphs instead of three — but recomputing the exact minimum and maximum every tick moves the mapping constantly. Snapping both ends holds it still until the range really changes: 91.3% to 98.7% scrolling on the same run, with no loss of detail. Scaling against the whole session's range was measured as an alternative and rejected: it scrolls just as well (98.9%) and collapses the same series to three glyphs, because one excursion half an hour ago flattens the view of now.
* A drifting auto-range was tried first and measured WORSE than doing nothing — 82.2% scrolling, and 56.6% on a bursty series — because a bound that creeps a little every tick is the flicker itself, not the cure. Recorded here so the idea is not tried again.
* The trend panel states its resolution (`1s per column`) and that the peak covers the whole session. A window that does not say how much time it spans cannot be read.
* Per-series bucket aggregation is gone. It existed to choose which extreme survived a bucket; with one sample per column there are no buckets.

## Version 1.4.0

### Changed

* Trend sparklines are fitted to the range the data actually occupies instead of being anchored at zero. A series living in a narrow band high above zero had all its detail crushed into the top glyph, which is the same as not drawing it. Measured over 46 ticks before the change: `ram free` moved 454 MiB and used TWO of the nine glyphs — a straight line across a real change — while `power` used seven and `gen t/s` three. After: `ram free` uses five and the shape of the drop is visible. Percentages keep their absolute 0..100 scale, because drawing 5% at full height would lie about the quantity.
* Every fitted row prints the low and high end it was drawn against. Fitting to the window is not free: a 3 MiB wobble and a 400 MiB drop fill the same height, so the bar carries the shape and the numbers beside it carry the magnitude. Absolute rows print the peak instead, since a "0…100" next to a label ending in "%" says nothing.
* A dot means a true zero and nothing else. On a fitted row the bottom of the range is a low bar, because a dot there reads as "nothing happened" when what happened is "the lowest value in this window".
* Each series declares which extreme its buckets keep. A history longer than the bar has to fold many samples into one column, and keeping the maximum preserves bursts while erasing dips — right for utilisation, power and tokens a second, wrong for free memory, where the event IS the drop. Verified on a 3600-sample series with a single-tick collapse: kept by the minimum, invisible to the maximum.
* A completed prefill now appears in the trend. The series was fed only from the live rate, and a prefill on a GPU is over before the second tick (4001 tokens in 1.94 s, measured), so the line stayed at zero, peaked at zero, and was dropped by the "only chart a speed that has actually occurred" test — invisible on exactly the servers fast enough to be worth watching. It is charted as one spike per completed prefill, at the rate the server timed.

## Version 1.3.0

### Added

* The server config panel shows every flag the model was launched with. It used to show a hardcoded subset, so a run started with `--dry-multiplier`, `--pooling`, `--jinja`, `--load-mode` or anything else outside that list had no line anywhere and nothing said so. Measured across four running servers: between six and eight settings displayed out of fifteen to eighteen flags actually passed. The themed groups are now the ORDER and the LABELS rather than the filter, and whatever they do not name lands in a final `other` group computed as the complement of what the groups consumed — so a flag added by a future llama.cpp release still appears, without anyone editing this file. Verified at 64 flags across those four servers with none dropped, and on a synthetic command line carrying the full DRY and XTC families, mirostat, rope/yarn, LoRA and two invented flags.
* Two new groups: `role` (pooling, embedding, reranking — what the server was started to do) and `server` (host, port, path, threads-http, metrics, slots and props endpoints, TLS).
* The sampling group carries a line saying those values are the server's DEFAULTS. A client that puts temperature or a penalty in the request body overrides them, and the command line cannot show that. The line appears only when sampling flags are actually present.
* API keys are masked. Showing every flag verbatim would otherwise print `--api-key` on a screen this tool is meant to be watched on.

### Fixed

* A negative value is no longer mistaken for the next flag. `-np -1` was read as a bare switch, so a server running with automatic parallelism displayed `slots: on`. A token starting with `-` now begins a new flag only when it is not a number.
* A negating switch is labelled by its concept rather than by the flag, so `--no-warmup` reads `warmup off` instead of `no-warmup off`, which said the opposite.
* Batch and micro-batch are labelled separately instead of being packed into one `4096/1024` cell.

## Version 1.2.0

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

## Version 1.1.0

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

## Version 1.0.0

### Added
* Initial release of the monitoring tool.
* Core terminal user interface for live tracking of GPU usage and fleet status.

# Changelog

All notable changes to this project will be documented in this file.

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

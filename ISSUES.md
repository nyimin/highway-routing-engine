# Known Issues Log

## 1. Bridge Siting Utility (`bridge_siting.py`) Deprecated

- **Status**: OPEN / DEPRECATED
- **Description**: The standalone `bridge_siting.py` utility is currently disconnected from the main pipeline and serves no practical purpose. It has been rendered obsolete by the inline `multi_pass_routing` strategy in `routing.py` and the structure detection logic in `structures.py`.
- **Impact**: Code bloat.
- **Action Required**: Wait for future architectural cleanup to either delete the file or completely overhaul it to integrate with the new multi-bridge flow. For the moment, it should not be used.

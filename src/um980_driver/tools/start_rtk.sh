#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${RTK_CONFIG_PATH:-$PACKAGE_DIR/config/rtk_client.ini}"
GGA_INTERVAL="${RTK_GGA_INTERVAL:-1}"
UM980_STATUS_INTERVAL="${UM980_STATUS_INTERVAL:-0}"

exec python3 "$SCRIPT_DIR/ntrip_rtk_bridge.py" \
  --config "$CONFIG_PATH" \
  --gga-interval "$GGA_INTERVAL" \
  --um980-status-interval "$UM980_STATUS_INTERVAL" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RULE_SRC="$PACKAGE_DIR/udev/99-rtk.rules"
RULE_DST="/etc/udev/rules.d/99-rtk.rules"

install -m 0644 "$RULE_SRC" "$RULE_DST"
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty

echo "Installed $RULE_DST"
ls -l "$RULE_DST" /dev/ttyrtk

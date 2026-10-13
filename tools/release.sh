#!/usr/bin/env bash
# Cut a release: bump the version, build it, install it.
#
# The installed file is always ~/.local/bin/show — never a versioned name.
# The desktop entry and any taskbar pin point at that path, and renaming the
# binary per release would break both every time.
#
#   tools/release.sh           bump the minor version and release
#   tools/release.sh --major   bump the major version instead
#   tools/release.sh --patch   bump the patch version instead
#   tools/release.sh --keep    release without bumping anything
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUMP="minor"
case "${1:-}" in
    --major) BUMP="major" ;;
    --patch) BUMP="patch" ;;
    --keep)  BUMP="none" ;;
    --minor|"") BUMP="minor" ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
esac

# Tests first. A release that does not pass them is not a release.
echo "==> testing"
cargo test --release --quiet

if [[ "$BUMP" != "none" ]]; then
    echo "==> bumping the $BUMP version"
    python3 - "$BUMP" <<'PY'
import re, sys

bump = sys.argv[1]
text = open("Cargo.toml").read()
# Only the [package] version, not a dependency that happens to match.
package = re.search(r"^\[package\](.*?)^\[", text, re.S | re.M)
if package is None:
    raise SystemExit("no [package] section in Cargo.toml")

match = re.search(r'^version = "(\d+)\.(\d+)\.(\d+)"', package.group(1), re.M)
if match is None:
    raise SystemExit("no version in [package]")

major, minor, patch = (int(g) for g in match.groups())
if bump == "major":
    major, minor, patch = major + 1, 0, 0
elif bump == "minor":
    minor, patch = minor + 1, 0
else:
    patch += 1

old = match.group(0)
new = f'version = "{major}.{minor}.{patch}"'
text = text[: package.start(1)] + package.group(1).replace(old, new, 1) + text[package.end(1) :]
open("Cargo.toml", "w").write(text)
print(f"  {match.group(1)}.{match.group(2)}.{match.group(3)} -> {major}.{minor}.{patch}")
PY
fi

echo "==> building"
cargo build --release --quiet

echo "==> installing"
./target/release/show --install

echo "==> archiving the source for this build"
"$ROOT/tools/backup.sh" >/dev/null

echo
echo "released: $(./target/release/show --version)"
echo "installed: $(~/.local/bin/show --version 2>/dev/null || echo 'not on PATH')"

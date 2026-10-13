#!/usr/bin/env bash
# Archive the source, and only the source.
#
# What is deliberately left out is most of the disk: `target` is rebuildable,
# `vendor` is a 200 MB FFmpeg that `build-ffmpeg.sh` reproduces, and `testdata`
# is downloadable by `fetch-corpus.sh`. Backing those up would turn a 200 KB
# archive into hundreds of megabytes of things that are not the work.
#
# Archives are named by build number, so one can be matched to the binary it
# produced.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$ROOT")"
DEST="${SHOW_BACKUP_DIR:-$(dirname "$ROOT")/$NAME-backups}"

BUILD="$(cat "$ROOT/build-number" 2>/dev/null || echo 0)"
STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$DEST/$NAME-build$BUILD-$STAMP.tar.zst"

# Everything that is the work, and nothing that is a build product.
# `assets` is not decoration: the subtitle face is compiled into the binary
# with include_bytes!, so an archive without it does not build.
INCLUDE=(
    src assets tools docs
    Cargo.toml Cargo.lock build.rs build-number
    .cargo .gitignore
)

mkdir -p "$DEST"
cd "$ROOT"

# Anything the source compiles in has to travel with it. Checked rather than
# remembered: the font was added months after this list was written, and an
# archive that cannot rebuild is worse than no archive, because it looks fine.
missing=()
while read -r needed; do
    [ -e "$needed" ] || continue
    for path in "${INCLUDE[@]}"; do
        case "$needed" in "$path"/*|"$path") continue 2 ;; esac
    done
    missing+=("$needed")
done < <(grep -rhoP 'include_(bytes|str)!\("\K[^"]+' src/ 2>/dev/null |
         sed 's|^\.\./||' | while read -r f; do
             [ -e "src/$f" ] && echo "src/$f" || echo "$f"
         done | sort -u)

if [ ${#missing[@]} -gt 0 ]; then
    echo "backup: these are compiled into the binary but not in the archive:" >&2
    printf "  %s\n" "${missing[@]}" >&2
    exit 1
fi

present=()
for path in "${INCLUDE[@]}"; do
    [[ -e "$path" ]] && present+=("$path")
done

if command -v zstd >/dev/null 2>&1; then
    tar --zstd -cf "$ARCHIVE" "${present[@]}"
else
    ARCHIVE="${ARCHIVE%.zst}.gz"
    tar -czf "$ARCHIVE" "${present[@]}"
fi

echo "$ARCHIVE"
du -h "$ARCHIVE" | cut -f1 | sed 's/^/  size: /'
tar -tf "$ARCHIVE" | wc -l | sed 's/^/  files: /'

# Keep the last twenty; a source tree this size is cheap, but not unbounded.
mapfile -t old < <(ls -1t "$DEST"/"$NAME"-build*.tar.* 2>/dev/null | tail -n +21)
if ((${#old[@]})); then
    rm -f "${old[@]}"
    echo "  pruned ${#old[@]} older archive(s)"
fi

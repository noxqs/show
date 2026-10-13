#!/usr/bin/env bash
# Build the static FFmpeg this project links against.
#
# Distributions are years behind — Ubuntu 24.04 ships 6.1 — and a viewer built
# here has to run on machines with a different one, so FFmpeg is vendored and
# linked statically rather than picked up from the system.
#
# Decoding is kept whole and everything else is dropped. Curating a decoder
# list looks tidy until someone opens a .wmv and the viewer says it cannot read
# a file every other player on the machine handles; the honest trade is to keep
# every decoder, demuxer and parser, and to throw away encoders, muxers,
# filters and devices, which a viewer never needs.
set -euo pipefail

VERSION="${FFMPEG_VERSION:-9.0.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/vendor/build"
PREFIX="$ROOT/vendor/ffmpeg9"

mkdir -p "$BUILD"
cd "$BUILD"

if [[ ! -d "ffmpeg-$VERSION" ]]; then
    echo "==> fetching FFmpeg $VERSION"
    curl -fsSL -o "ffmpeg-$VERSION.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz"
    tar xf "ffmpeg-$VERSION.tar.xz"
fi

cd "ffmpeg-$VERSION"
echo "==> configuring"
./configure --prefix="$PREFIX" \
    --disable-programs --disable-doc --disable-network --disable-autodetect \
    --disable-encoders --disable-muxers --disable-filters --disable-devices \
    --enable-protocol=file,pipe \
    --enable-static --disable-shared --enable-pic \
    --enable-vaapi

echo "==> building"
make -j"$(nproc)"
make install
echo "==> installed to $PREFIX"

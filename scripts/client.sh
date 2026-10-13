#!/usr/bin/env bash
# Start the native client, building it first if needed. usage: scripts/client.sh [--server ws://HOST:PORT] [--name NICK] ...
# On NixOS the X11/GL libraries macroquad loads at run time are not on a standard path; they are taken from nixpkgs.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
bin=$root/rust/game_client/target/release/game_client
[ -x "$bin" ] || cargo build --release --manifest-path "$root/rust/game_client/Cargo.toml"
if [ -e /etc/NIXOS ]; then
  libs=$(nix build --no-link --print-out-paths nixpkgs#libx11.out nixpkgs#libxi.out nixpkgs#libGL.out nixpkgs#libxkbcommon.out | sed 's|$|/lib|' | paste -sd:)
  export LD_LIBRARY_PATH=$libs:/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
exec "$bin" "$@"

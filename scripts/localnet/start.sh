#!/usr/bin/env bash
# Start a disposable 3-authority local subtensor natively (no Docker daemon).
# Usage: NODE_DIR=/path/with/rootfs DATA=/tmp/reax-localnet RPC_PORT=9951 start.sh
set -euo pipefail
NODE_DIR=${NODE_DIR:?set NODE_DIR to the directory filled by fetch_node.py}
DATA=${DATA:-/tmp/reax-localnet}
RPC=${RPC_PORT:-9951}
P2P=${P2P_PORT:-30361}
R="$NODE_DIR/rootfs"
LD="$R/lib64/ld-linux-x86-64.so.2"
BIN="$R/target/${FLAVOR:-fast-runtime}/release/node-subtensor"
node() { "$LD" --library-path "$R/usr/lib/x86_64-linux-gnu:$R/lib/x86_64-linux-gnu" "$BIN" "$@"; }
rm -rf "$DATA"; mkdir -p "$DATA"
node build-spec --disable-default-bootnode --raw --chain local > "$DATA/local.json" 2>/dev/null
node key insert --base-path "$DATA/three" --chain="$DATA/local.json" --scheme Sr25519 --suri "//Three" --key-type aura
node key insert --base-path "$DATA/three" --chain="$DATA/local.json" --scheme Ed25519 --suri "//Three" --key-type gran
common=(--chain="$DATA/local.json" --validator --rpc-cors=all --allow-private-ipv4 --discover-local --unsafe-force-node-key-generation --rpc-max-connections 2000)
i=0
for who in "--one" "--two" "--name Three"; do
  name=$(echo $who | sed 's/--name //; s/--//' | tr 'A-Z' 'a-z')
  setsid nohup "$LD" --library-path "$R/usr/lib/x86_64-linux-gnu:$R/lib/x86_64-linux-gnu" "$BIN" \
    --base-path "$DATA/$name" $who --port $((P2P+i)) --rpc-port $((RPC+i)) "${common[@]}" > "$DATA/$name.log" 2>&1 &
  echo $! >> "$DATA/pids"
  i=$((i+1))
done
echo "local subtensor starting: ws://127.0.0.1:$RPC (logs in $DATA)"

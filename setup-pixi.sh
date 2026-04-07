#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/setup-backends.sh"

# Pick a machine-local root. Adjust these if you prefer another local filesystem.
if [[ -d "/tmp/$USER" ]]; then
  LOCAL_PIXI_ROOT="/tmp/$USER/pixi-$HOST"
elif [[ -d "/var/tmp/$USER" ]]; then
  LOCAL_PIXI_ROOT="/var/tmp/$USER/pixi-$HOST"
else
  LOCAL_PIXI_ROOT="$SCRIPT_DIR/.pixi-local-$HOST"
fi

export PIXI_HOME="$LOCAL_PIXI_ROOT/home"
DETACHED_ENVS_DIR="$LOCAL_PIXI_ROOT/envs"

mkdir -p "$PIXI_HOME" "$DETACHED_ENVS_DIR"

cat > "$PIXI_HOME/config.toml" <<EOF
detached-environments = "$DETACHED_ENVS_DIR"
EOF

echo "HOST=$HOST"
echo "PIXI_HOME=$PIXI_HOME"
echo "detached-environments=$DETACHED_ENVS_DIR"

exec pixi "$@"

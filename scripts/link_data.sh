#!/usr/bin/env bash
# Create symlinks inside the repo root pointing into the cluster storage:
#   data    -> $DIFIX3D_ROOT/data
#   outputs -> $DIFIX3D_ROOT/outputs
# Override the storage root with: DIFIX3D_ROOT=/some/path scripts/link_data.sh
set -euo pipefail

ROOT="${DIFIX3D_ROOT:-/cluster/angmar/rborth/difix3d}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

link_dir() {
  local name="$1"
  local target="$ROOT/$name"
  local link="$REPO_ROOT/$name"

  mkdir -p "$target"

  if [[ -L "$link" ]]; then
    echo "Replacing existing symlink: $link -> $(readlink "$link")"
    rm "$link"
  elif [[ -d "$link" ]]; then
    echo "Migrating existing directory $link -> $target"
    # Move contents (including dotfiles) into the target, then remove dir.
    shopt -s dotglob nullglob
    local entries=("$link"/*)
    if (( ${#entries[@]} > 0 )); then
      mv "${entries[@]}" "$target/"
    fi
    shopt -u dotglob nullglob
    rmdir "$link"
  elif [[ -e "$link" ]]; then
    echo "Error: $link exists and is not a directory or symlink. Aborting." >&2
    exit 1
  fi

  ln -s "$target" "$link"
  echo "Linked: $link -> $target"
}

link_dir data
link_dir outputs

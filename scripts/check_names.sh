#!/usr/bin/env bash
# Fails if any pattern from the local, gitignored .names-denylist appears in a tracked file.
# No allowlist: only this script is excluded. Without .names-denylist (CI, fresh clones) the
# check is skipped. Each non-comment line is "<git grep flags> <regex>", e.g. "-iE foo|bar".
# Uses git grep (no xargs). Any grep error fails the check (exit 2) instead of passing silently.
set -uo pipefail
root=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "names: not inside a git repo" >&2; exit 2; }
cd "$root"

list=.names-denylist
if [ ! -f "$list" ]; then echo "names: skipped (no $list)"; exit 0; fi

excludes=(-- . ':(exclude)scripts/check_names.sh')
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT

n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  flags=${line%% *}; pattern=${line#* }
  if [ "$flags" = "$line" ] || [ -z "$pattern" ]; then echo "names: bad line in $list: $line" >&2; exit 2; fi
  git grep -nI "$flags" "$pattern" "${excludes[@]}" >>"$tmp"
  rc=$?
  if [ $rc -ge 2 ]; then echo "names: git grep failed (exit $rc)" >&2; exit 2; fi
  n=$((n + 1))
done <"$list"
if [ $n -eq 0 ]; then echo "names: $list has no patterns" >&2; exit 2; fi

hits=$(sort -u "$tmp")
if [ -n "$hits" ]; then
  echo "Banned names found (no file may mention them):"
  printf '%s\n' "$hits"
  exit 1
fi
echo "names: clean ($n patterns)"

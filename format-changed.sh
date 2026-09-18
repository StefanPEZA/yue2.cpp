#!/bin/bash
# Format only the C++ files this branch touches.
#
# format.sh rewrites every file in the tree, and the clang-format available
# here (23.x) aligns trailing comments differently from the one the repo was
# last formatted with, so a full run buries a change under unrelated churn.
# Limit it to what we actually edited.

set -e
cd "$(dirname "$0")"

files=$(git diff --name-only HEAD; git diff --name-only --cached; git ls-files --others --exclude-standard)
echo "$files" | sort -u | grep -E '\.(cpp|h)$' | grep -v -e '^build/' -e '^ggml/' -e '^vendor/' |
    while read -r f; do
        [ -f "$f" ] && clang-format -i "$f" && echo "formatted $f"
    done

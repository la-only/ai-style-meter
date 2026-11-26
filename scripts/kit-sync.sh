#!/usr/bin/env bash
# kit-sync — synchronize the protocol from the kit submodule to its parent project.
#
# opencode reads AGENTS.md and .opencode/ from the project root, while the
# submodule is a child directory. The project root is a working copy and the
# submodule is the source of protocol updates.
#
# Usage from the project root:
#   make kit-sync                      # submodule in ./paradigm
#   make kit-sync KIT_DIR=opencode-team-starter
# Direct invocation:
#   bash paradigm/scripts/kit-sync.sh paradigm
set -euo pipefail

KIT_DIR="${1:-paradigm}"

if [[ ! -f "$KIT_DIR/AGENTS.md" || ! -d "$KIT_DIR/.opencode" ]]; then
  echo "kit-sync: в '$KIT_DIR' не найден кит (нужны AGENTS.md и .opencode/)" >&2
  exit 1
fi

if [[ ! -d "$KIT_DIR/scripts" || ! -f "$KIT_DIR/Makefile.target" || ! -f "$KIT_DIR/__version__.py" ]]; then
  echo "kit-sync: в '$KIT_DIR' не найдены scripts/, Makefile.target или __version__.py" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$KIT_DIR/.." && pwd)"

mkdir -p "$PROJECT_DIR/scripts"
cp -R "$KIT_DIR/scripts/." "$PROJECT_DIR/scripts/"
cp "$KIT_DIR/Makefile.target" "$PROJECT_DIR/Makefile"

mkdir -p \
  "$PROJECT_DIR/docs" \
  "$PROJECT_DIR/docs/adr" \
  "$PROJECT_DIR/docs/reviews" \
  "$PROJECT_DIR/docs/risks" \
  "$PROJECT_DIR/docs/specs" \
  "$PROJECT_DIR/docs/tasks"

if [[ ! -e "$PROJECT_DIR/AGENTS.md" ]]; then
  cp "$KIT_DIR/AGENTS.md" "$PROJECT_DIR/AGENTS.md"
fi

if [[ ! -e "$PROJECT_DIR/VERSION" ]]; then
  : > "$PROJECT_DIR/VERSION"
fi

if [[ ! -e "$PROJECT_DIR/README.md" ]]; then
  : > "$PROJECT_DIR/README.md"
fi

if [[ ! -e "$PROJECT_DIR/__version__.py" ]]; then
  cp "$KIT_DIR/__version__.py" "$PROJECT_DIR/__version__.py"
fi

rm -rf "$PROJECT_DIR/.opencode"
cp -R "$KIT_DIR/.opencode" "$PROJECT_DIR/.opencode"

VERSION="$(tr -d '[:space:]' < "$KIT_DIR/VERSION" 2>/dev/null || echo 'unknown')"
COMMIT="$(git -C "$KIT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'n/a')"

echo "kit-sync: протокол версии $VERSION (commit $COMMIT) синхронизирован в $PROJECT_DIR"
echo "kit-sync: не редактируй файлы внутри $KIT_DIR/ — изменения протокола делаются PR в репозиторий кита"

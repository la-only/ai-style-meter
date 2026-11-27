PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
VERSION := $(shell $(PYTHON) -c 'from __version__ import __version__; print(__version__)')
.PHONY: test lint typecheck check run task review kit-sync codeaudit bandit

# Путь к сабмодулю кита внутри проекта (переопределяется: make kit-sync KIT_DIR=...)
KIT_DIR ?= paradigm

# --- Гейты качества (поправь под свой проект) ---

test:
	python -m pytest -q

bandit:
	bandit -r .

codeaudit:
	codeaudit filescan .

lint:
	ruff check .

typecheck:
	mypy scripts tests stylometry

check: lint typecheck test

run: build
	$(PYTHON) run.py

# --- Цикл «команды» ---

# Реализация задачи агентом:
#   git checkout -b task/TASK-001-xxx
#   make task T=docs/tasks/TASK-001-xxx.md
task:
	opencode run --agent implementer --file $(T) \
		"Реализуй задачу из приложенного файла по процессу из AGENTS.md."

# Независимое ревью последнего коммита вторым агентом:
#   make review T=docs/tasks/TASK-001-xxx.md
review:
	opencode run --agent reviewer --file $(T) \
		"Проведи ревью diff HEAD~1..HEAD по задаче из файла. Отчёт — в docs/reviews/."

# Синхронизация протокола из сабмодуля кита в корень проекта:
#   make kit-sync                  # сабмодуль в ./paradigm
#   make kit-sync KIT_DIR=путь/к/киту
kit-sync:
	bash $(KIT_DIR)/scripts/kit-sync.sh $(KIT_DIR)

# Статистика токенов OpenCode:
#   make extractt_token_stat_for ARGS="--session ses_..."
#   make extractt_token_stat_for ARGS="--date 2026-11-13"
extractt_token_stat_for:
	$(PYTHON) scripts/opencode_token_statistics_extracter.py $(ARGS)

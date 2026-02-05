# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Commands

- **Run application:** `uv run main.py`
- **Lint and fix:** `uv tool run ruff check --fix`
- **Markdown lint and fix:** `bunx -bun markdownlint-cli2 --fix "**/*.md"`

## Commit Convention

- 형식: `type: 한국어 설명`
- 타입: feat, fix, refactor, chore, docs, test
- 커밋은 의미 있는 단위로 분리

## Code Style

- **`__init__.py` 금지**: 패키지에 `__init__.py` 파일을 생성하지 않음. 직접 모듈
  경로로 import (예: `from src.search.graph.builder import build_search_graph`)

## Architecture

This is a Python project using UV as the build system, targeting Python 3.13+.

### Documentation Reference (docs/)

The `docs/` directory contains architecture decisions and patterns:

- **adr-001-model-selection.md**: 검색 에이전트 모델 선택 결정 기록
- **benchmark-model-comparison.md**: 모델 벤치마크 비교 결과
- **langgraph-patterns.md**: LangGraph 패턴 정리

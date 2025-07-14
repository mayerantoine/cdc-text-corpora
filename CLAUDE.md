# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project is a PYTHON package (`cdc-text-corpora`) that provides a beginner-friendly interface to the CDC Text Corpora dataset, enabling easy access to three fixed collections (PCD, EID, and MMWR) with built-in semantic search and RAG-based question-answering capabilities with citations.

## **Standard Workflow**

- First think through the problem, read the codebase for relevant files, and review the current plan in file **@/tasks/todo.md**.
- The plan should have a list of todo items that you can check off as you complete them
- Before writing any code for a task, you MUST create a plan and ask for approval.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome for a task, breakdown the task if necessary.
- Every change should impact as little code as possible. Everything is about simplicity.
- Implement ONE task at a time in the todo.md—we will test and validate each task before moving on to the next, marking them as complete as you go.
- Ask approval to update the todo for any missing tasks.
- ALWAYS ask for clarification rather than making assumptions.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST NEVER make code changes unrelated to your current task.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST commit frequently throughout the development process, even if your high-level tasks are not yet done.

## Common Development Commands

**Package Management:**
- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

**Running Code:**
- Run Python script: `uv run <script-name>.py`
- Run tests: `uv run pytest`
- Run linting: `uv run flake8`
- Run type checking: `uv run mypy`
- Launch Python REPL: `uv run python`

**CLI Commands:**
- Main CLI: `uv run cdc-corpus --help`
- Download data: `uv run cdc-corpus download -c all`
- Parse articles: `uv run cdc-corpus parse -c all`
- Start Q&A: `uv run cdc-corpus qa`

**Testing:**
- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest tests/test_pcd_parser.py`
- Run tests by marker: `uv run pytest -m unit`
- Skip slow tests: `uv run pytest -m "not slow"`

## Package Architecture

### Core Design Pattern

The package follows a **layered architecture** with clear separation of concerns:

1. **CLI Layer** (`cli/`): User interface and command handling
2. **Core Layer** (`core/`): Business logic for downloading, parsing, and managing data
3. **QA Layer** (`qa/`): RAG-based question answering and semantic search
4. **Utils Layer** (`utils/`): Shared utilities and configuration

### Key Architectural Components

**CDCCorpus (core/datasets.py)**: Main orchestrator class that coordinates all operations
- Handles downloading collections via `download_html_articles()`
- Manages parsing with `load_parse_save_html_articles()`
- Provides data access through `load_json_articles_as_dataframe()` and `load_json_articles_as_iterable()`
- Tracks collection status and metadata

**ArticleCollection (core/datasets.py)**: Memory-efficient article iterator
- Uses generators to process large collections without loading everything into memory
- Filters by collection and language
- Implements standard Python iteration protocols

**Parser Architecture (core/parser.py)**: Collection-specific parsing
- Abstract base class `CDCArticleParser` with collection-specific implementations
- Factory pattern via `create_parser()` function
- Handles HTML extraction, content cleaning, and validation

**RAG Engine (qa/rag_engine.py)**: Semantic search and question answering
- Vector database integration with ChromaDB
- LLM integration supporting OpenAI and Anthropic
- Citation tracking and context management

### Data Flow

1. **Download**: HTML collections are downloaded as ZIP files to `cdc-corpus-data/html-outputs/`
2. **Extract**: ZIP files are extracted to `cdc-corpus-data/json-html/`
3. **Parse**: HTML files are parsed into structured JSON in `cdc-corpus-data/json-parsed/`
4. **Index**: Articles are vectorized and stored in `cdc-corpus-data/chroma_db/`
5. **Query**: RAG engine retrieves relevant articles and generates answers

## Python Package Management with uv

- Use uv exclusively for Python package management in this project.
- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management.
- The project uses pyproject.toml for configuration and uv.lock for dependency locking.

## Testing Architecture

Tests are organized by collection type with shared test data:

- **Unit Tests**: `test_pcd_parser.py`, `test_eid_parser.py`, `test_mmwr_parser.py`
- **Test Data**: `tests/articles/` contains sample HTML and expected JSON outputs
- **Integration Tests**: Use `@pytest.mark.integration` marker
- **Slow Tests**: Use `@pytest.mark.slow` marker (can be skipped with `-m "not slow"`)

**Test Configuration (pyproject.toml):**
- Tests exclude `tests/articles/` and `tests/ignore/` directories
- Strict configuration with comprehensive markers
- Type checking with mypy enabled
- Linting with flake8 configured

## Configuration Management

**Environment Variables:**
- `OPENAI_API_KEY`: OpenAI API key for RAG functionality
- `ANTHROPIC_API_KEY`: Anthropic API key for RAG functionality
- `DEFAULT_LLM_PROVIDER`: Default LLM provider (openai/anthropic)
- `DEFAULT_LLM_MODEL`: Default model name
- `DEFAULT_EMBEDDING_MODEL`: Embedding model for semantic search

**Configuration Files:**
- `pyproject.toml`: Project configuration, dependencies, and tool settings
- `uv.lock`: Locked dependency versions
- `.env`: Environment variables (not tracked in git)

## Data Collections

The package supports three fixed CDC collections:

1. **PCD (Preventing Chronic Disease)**: 2004-2023, English/Spanish
2. **EID (Emerging Infectious Diseases)**: 1995-2023, English only
3. **MMWR (Morbidity and Mortality Weekly Report)**: 1982-2023, Multi-language

Each collection has specific parsing logic to handle different HTML structures and metadata formats.

## Memory Management

The package is designed to handle large datasets efficiently:

- **Streaming Parsing**: ArticleCollection uses generators to process articles one at a time
- **Lazy Loading**: JSON files are only loaded when accessed
- **Vector Database**: ChromaDB handles efficient storage and retrieval of embeddings
- **Progress Tracking**: Rich progress bars for long-running operations

## Error Handling Patterns

- **Graceful Degradation**: Continue processing even if individual articles fail
- **Validation**: Optional article validation during parsing with detailed error reporting
- **User Feedback**: Rich console output with clear error messages and suggestions
- **Logging**: Comprehensive error tracking throughout the pipeline

## CLI Design Philosophy

The CLI follows a **three-step workflow**:
1. **Download**: Get raw HTML data from CDC sources
2. **Parse**: Convert HTML to structured JSON
3. **Query**: Interactive RAG-based question answering

Each command is designed to be:
- **Idempotent**: Can be run multiple times safely
- **Resumable**: Can continue from where it left off
- **Informative**: Provides clear feedback and progress indicators
- **Flexible**: Supports filtering by collection and language

## Development Best Practices

- **Type Safety**: Full mypy coverage with strict configuration
- **Code Quality**: Flake8 linting with reasonable line length limits
- **Testing**: Comprehensive test coverage with real CDC data samples
- **Documentation**: Inline docstrings and clear API documentation
- **Dependencies**: Minimal external dependencies, prefer standard library when possible
- **Performance**: Memory-efficient processing for large datasets
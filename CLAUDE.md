## Overview

This project is a PYTHON package (`cdc-text-corpora`) that provides a beginner-friendly interface to the CDC Text Corpora dataset, enabling easy access to three fixed collections (PCD, EID, and MMWR) with built-in semantic search and RAG-based question-answering capabilities with citations.

## **Standard Workflow**

- First think through the problem, read the codebase for relevant files, and review the current plan in file **@/tasks/[todo.md](http://todo.md/).**
- The plan should have a list of todo items that you can check off as you complete them
- Before writing any code for a task, you MUST create a plan and ask for approval.
- YOU MUST make the SMALLEST reasonable changes to achieve the desired outcome for a task, breakdown the task if necessary.
- Every change should impact as little code as possible. Everything is about simplicity.
- Implement ONE task at a time in the todo.md—we will test and validate each taks before moving on to the next, marking them as complete as you go.
- Ask approval to update the todo for any missing tasks.
- ALWAYS ask for clarification rather than making assumptions.
- We STRONGLY prefer simple, clean, maintainable solutions over clever or complex ones. Readability and maintainability are PRIMARY CONCERNS, even at the cost of conciseness or performance.
- YOU MUST NEVER make code changes unrelated to your current task.
- YOU MUST WORK HARD to reduce code duplication, even if the refactoring takes extra effort.
- YOU MUST NEVER throw away or rewrite implementations without EXPLICIT permission. If you're considering this, YOU MUST STOP and ask first.
- YOU MUST commit frequently throughout the development process, even if your high-level tasks are not yet done.

## Python Package Management with uv

- Use uv exclusively for Python package management in this project.
- Make sure that there is a pyproject.toml file in the root directory.
- If there isn't a pyproject.toml file, create one using uv by running uv init.
- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management.
    - Use these commands:
        - Install dependencies: `uv add <package>`
        - Remove dependencies: `uv remove <package>`
        - Sync dependencies: `uv sync`
    - Running Python Code
        - Run a Python script with `uv run <script-name>.py`
        - Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
        - Launch a Python repl with `uv run python`
    - Managing Scripts with PEP 723 Inline Metadata
        - Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
        - You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
        - Or using uv CLI:
            - `uv add package-name --script script.py`
            - `uv remove package-name --script script.py`

## Package Architecture

### Core Modules Structure

```

cdc_text_corpora/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py             # Main CLI entry point
├── core/
│   ├── __init__.py
│   ├── downloader.py       # Data downloading and caching
│   ├── parser.py           # Document parsing and content extraction
│   └── json.py             # JSON Local storage management
|   └── collections.py      # main module for collections 
├── qa/
│   ├── __init__.py
│   ├── rag_engine.py       # RAG-based QA engine using LangChain
│   ├── retriever.py        # Document retrieval for RAG
│   ├── citation.py         # Citation generation and tracking
│   └── context.py          # Context management for QA
├── utils/
│   ├── __init__.py
│   ├── cache.py            # Caching utilities
│   ├── config.py           # Configuration management
│   ├── validation.py       # Input validation
│   └── embeddings.py       # Embedding utilities


```

### Class Responsibilities

### CLI Components

- **CLIManager**: Main CLI interface coordinator
- **DownloadCommand**: Handles dataset downloading via CLI
- **ParseCommand**: Manages document parsing operations
- **QACommand**: Interactive QA interface

### Core Components

- **CDCCorpusDownloader**: Downloads from the three fixed collections
- **DocumentParser**: Extracts content from HTML documents using LangChain
- **StorageManager**: Manages local storage and metadata


### QA Components
- **VectorDatabase**: Manages vector embeddings storage (ChromaDB/Pinecone)
- **SemanticSearcher**: Implements semantic search using embeddings
- **HybridSearcher**: Combines keyword and semantic search
- **RAGEngine**: LangChain-based RAG implementation for QA
- **DocumentRetriever**: Retrieves relevant documents for RAG
- **CitationManager**: Generates and tracks citations in responses
- **ContextManager**: Manages document context for QA


# CDC Text Corpora - Component Architecture

## Key Module Relationships

### 1. **Entry Points**
```
cli/main.py → core/datasets.py (CDCCorpus)
```
- CLI commands route to CDCCorpus methods
- CDCCorpus is the main orchestrator

### 2. **Data Pipeline**
```
downloader.py → parser.py → models.py
```
- Downloader fetches data from CDC
- Parser converts HTML to structured data  
- Models define data structures

### 3. **Q&A System**
```
rag_pipeline.py → rag_engine.py → datasets.py
rag_agent.py → rag_engine.py → datasets.py
```
- Both QA modes use RAG Engine
- RAG Engine loads data via CDCCorpus

### 4. **Support Systems**
```
config.py → All modules (configuration)
validation.py → parser.py (data quality)
```
- Config provides settings and paths
- Validation ensures data quality

## File Structure and Responsibilities

```
src/cdc_text_corpora/
├── cli/
│   └── main.py              # 🖥️ CLI commands and user interface
├── core/
│   ├── datasets.py          # 📊 CDCCorpus - main orchestrator
│   ├── downloader.py        # ⬇️ Download from CDC APIs  
│   ├── parser.py            # 🔧 Parse HTML to JSON
│   └── models.py            # 📄 Data structures (Article, etc.)
├── qa/
│   ├── rag_engine.py        # 🧠 Vector search + LLM integration
│   ├── rag_pipeline.py      # 🔄 Sequential Q&A workflow
│   └── rag_agent.py         # 🤖 Multi-agent research system
└── utils/
    ├── config.py            # ⚙️ Configuration and settings
    └── validation.py        # ✅ Data quality and validation
```

## Key Design Patterns

### 1. **Orchestrator Pattern**
- `CDCCorpus` coordinates all operations
- Single entry point for data operations

### 2. **Factory Pattern**  
- `create_parser()` creates collection-specific parsers
- Handles PCD/EID/MMWR differences

### 3. **Strategy Pattern**
- Different QA modes (Sequential vs Agentic)
- Different LLM providers (OpenAI vs Anthropic)

### 4. **Repository Pattern**
- File system abstraction through CDCCorpus
- Consistent data access interface

This diagram shows the **essential connections** and **data flow** between modules, making it clear how the codebase works without overwhelming detail.
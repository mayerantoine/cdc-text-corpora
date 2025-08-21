# CDC Text Corpora

A Python package that provides easy access to CDC (Centers for Disease Control and Prevention) text corpora with built-in semantic search and RAG-based question-answering capabilities.

## Overview

This package enables researchers, developers, and health professionals to easily access and analyze CDC publications from three major collections:

- **PCD** - Preventing Chronic Disease (2004-2023)
- **EID** - Emerging Infectious Diseases (1995-2023) 
- **MMWR** - Morbidity and Mortality Weekly Report (1982-2023)

### Key Features

- **Easy Data Access**: Simple CLI commands to download and parse CDC publications
- **Semantic Search**: Vector-based search using modern embedding models
- **RAG Q&A**: Interactive question-answering with citation support
- **Multi-language**: Support for English, Spanish, French, and Chinese content
- **Fast Processing**: Efficient parsing and indexing with progress tracking
- **Flexible Configuration**: Support for OpenAI and Anthropic LLMs

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cdc-text-corpora

# Install with uv (recommended)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or on Windows:
# .venv\Scripts\activate

# Or install in development mode
uv pip install -e .
```

## Quick Start

### First Step: Setup API Keys (Required for RAG Features)

⚠️ **Important**: Before using RAG features (semantic search and Q&A), you must configure API keys for LLM providers:

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Add your API keys (at least one is required for RAG features)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Configure default settings
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
```

**Note**: You can download and parse collections without API keys, but the `qa` command requires at least one LLM provider configured.

### Option 1: Interactive Setup (Recommended)

Run the guided setup process that handles everything automatically:

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # On Linux/macOS (.venv\Scripts\activate on Windows)

# Launch interactive setup
cdc-corpus run
# or simply
cdc-corpus
```

This will guide you through:
1. **API Key Setup**: Configure LLM providers for RAG features
2. **Collection Selection**: Choose PCD, EID, MMWR, or all collections
3. **Language Selection**: Pick from available languages
4. **Data Download**: Automatically downloads selected collections
5. **Parsing**: Extracts and structures articles from HTML
6. **Indexing**: Creates vector database for semantic search
7. **Q&A Launch**: Starts interactive question-answering session

### Option 2: Manual Setup

#### Option 2A: Quick Start with Bundled Index

Get started immediately using the pre-built index included with the package:

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # On Linux/macOS (.venv\Scripts\activate on Windows)

# Extract bundled pre-built index for immediate use
cdc-corpus index --use-bundled

# Start Q&A session right away (requires API keys)
cdc-corpus qa
```

**Benefits of Bundled Index:**
- ✅ Instant setup - no download/parsing required
- ✅ Pre-built from all collections and languages  
- ✅ Professionally optimized embeddings
- ⚠️ ~1.2GB disk space required after extraction

#### Option 2B: Build Fresh Index from Data

Build your own index from the latest CDC data:

##### 1. Test API Connection

```bash
# Test your API configuration
uv run tests/test_api_connection.py
```

##### 2. Download Data

```bash
# Download all collections
cdc-corpus download --collection all

# Download specific collection
cdc-corpus download --collection pcd
cdc-corpus download --collection eid
cdc-corpus download --collection mmwr
```

##### 3. Parse Articles

```bash
# Parse all collections (English)
cdc-corpus parse --collection all

# Parse specific collection and language
cdc-corpus parse --collection pcd --language en
cdc-corpus parse --collection eid --language es
```

##### 4. Index and Start Q&A Session

```bash
# Interactive Q&A (will auto-build index from parsed articles)
cdc-corpus qa

# Or manually build index first
cdc-corpus index --collection pcd
cdc-corpus qa --collection pcd
```

## CLI Reference

### Download Command

Download CDC text corpora collections:

```bash
cdc-corpus download [OPTIONS]

Options:
  -c, --collection [pcd|eid|mmwr|metadata|all]  Collection to download (default: all)
  -v, --verbose                                 Show verbose output
  --help                                        Show help message
```

### Parse Command

Parse HTML articles into structured JSON data:

```bash
cdc-corpus parse [OPTIONS]

Options:
  -c, --collection [pcd|eid|mmwr|all]          Collection to parse (default: all)
  -l, --language [en|es|fr|zhs|zht|all]        Language filter (default: all)
  -j, --save-json / --no-save-json             Save as JSON files (default: True)
  -o, --output-dir TEXT                        Custom output directory
  -v, --verbose                                Show verbose output
```

### Index Command

Create and manage vector indexes for semantic search:

```bash
cdc-corpus index [OPTIONS]

Options:
  -c, --collection [pcd|eid|mmwr|all]          Collection to index (default: all)
  -l, --language [en|es|fr|zhs|zht|all]        Language filter (default: en)
  -s, --source-type [json|html]               Source type (default: json)
  --use-bundled                               Extract pre-built bundled index
  --status                                    Show index and data status
  -v, --verbose                               Show verbose output
```

**Index Management:**
- **Bundled Index**: Pre-built index included with package for immediate use
- **Fresh Index**: Build index from your downloaded/parsed articles
- **Status Check**: Review current index and data availability

### QA Command

Interactive RAG-based question answering with two modes:

```bash
cdc-corpus qa [OPTIONS]

Options:
  -c, --collection [pcd|eid|mmwr|all]          Collection to query (default: all)
  -l, --language [en|es|fr|zhs|zht|all]        Language filter (default: en)
  -m, --mode [agentic]                         QA mode (default: agentic)
  -d, --data-dir TEXT                          Custom data directory
  -v, --verbose                                Show verbose output
```

**QA Mode:**
- **Agentic**: Multi-agent research system with evidence gathering and synthesis

### Run Command

Interactive guided setup (default when no command specified):

```bash
cdc-corpus run [OPTIONS]
# or simply: cdc-corpus

Options:
  -v, --verbose    Show verbose output
  --help          Show help message
```

## Complete Process Flow and Architecture

### Data Pipeline Overview

The CDC Text Corpora package follows a comprehensive 5-stage pipeline:

```
1. DOWNLOAD:  CDC APIs → ZIP files → html-outputs/
2. EXTRACT:   ZIP files → HTML files → json-html/  
3. PARSE:     HTML → structured Article objects → json-parsed/
4. INDEX:     Articles → embeddings → chroma_db/
5. QUERY:     Question → semantic search → LLM → answer + citations
```

### Detailed Command Flows

#### 1. Download Command Flow

**Execution Path**: `CLI → Core Downloader → File System`

```
cli/main.py:download()
  ↓
core/downloader.py:download_collection()
  ↓
├─ Metadata: CDC Socrata API → cdc-corpus-data/cdc_corpus_df.csv
└─ Collections: ZIP downloads → cdc-corpus-data/html-outputs/[collection].zip
```

**Data Sources:**
- **PCD**: Preventing Chronic Disease articles (2004-2023)
- **EID**: Emerging Infectious Diseases articles (1995-2023)
- **MMWR**: Morbidity and Mortality Weekly Report (1982-2023)
- **Metadata**: Article metadata via Socrata API

#### 2. Parse Command Flow

**Execution Path**: `ZIP → HTML → Structured JSON`

```
cli/main.py:parse()
  ↓
core/datasets.py:CDCCorpus.load_parse_save_html_articles()
  ↓
├─ HTMLArticleLoader: Extract ZIPs → json-html/
├─ Collection-specific parsers → Article dataclasses
└─ JSON output → json-parsed/[collection]_[language]_[timestamp].json
```

**Processing Details:**
- **HTML Extraction**: Collection-specific folder structures
  - PCD: `issues/` folder
  - EID: `article/` folder
  - MMWR: `preview/mmwrhtml/` and `volumes/` folders
- **Language Filtering**: `_es.htm`, `_fr.htm`, `_zhs.htm`, `_zht.htm` suffixes
- **Article Parsing**: Title, abstract, full text, authors, references, metadata
- **Validation**: Optional article validation with detailed error reporting

#### 3. QA Command Flow

##### Agentic RAG Mode

**Execution Path**: `Multi-Agent Research → Evidence Synthesis → Comprehensive Answer`

```
cli/main.py:qa()
  ↓
qa/rag_agent.py:AgenticRAG
  ↓
Multi-agent system with specialized tools:
├─ Search Tool: Semantic search → passages
├─ Evidence Tool: Relevance scoring (1-10) → filtered evidence  
└─ Answer Tool: Evidence synthesis → comprehensive answer
  ↓
Orchestrator: Search → Gather Evidence → Generate Answer
```

**Agent Capabilities:**
- **Collection-specific instructions**: Tailored for PCD/EID/MMWR content
- **Multi-turn workflow**: Iterative research and evidence gathering
- **Configurable parameters**: Evidence pieces, relevance cutoff, search attempts
- **Async execution**: Parallel processing with rich status updates

#### 4. Run Command Flow (Interactive Setup)

**Execution Path**: `Guided Workflow → Complete Pipeline → Q&A Session`

```
cdc-corpus run
  ↓
Interactive guided workflow:
├─ 1. Collection selection & download
├─ 2. Language selection & parsing  
├─ 3. Vector indexing
└─ 4. Launch agentic Q&A
```

### Architecture Components

#### Data Management Layer
- **CDCCorpus**: Main orchestrator for all data operations
- **ArticleCollection**: Memory-efficient iterator using generators
- **HTMLArticleLoader**: ZIP extraction and HTML file loading
- **Data Directories**: Standardized via `utils/config.py`

#### Parsing Architecture
- **Collection-specific parsers**: PCD/EID/MMWR specialized parsing logic
- **Article dataclass**: Structured representation with validation
- **Factory pattern**: `create_parser()` for parser instantiation
- **Error handling**: Graceful degradation with user feedback

#### RAG Architecture
- **RAGEngine**: Core semantic search and QA engine
  - LangChain integration for LLM orchestration
  - ChromaDB for vector storage and retrieval
  - HuggingFace embeddings for semantic search
  - Supports OpenAI and Anthropic LLMs
- **AgenticRAG**: Multi-agent research system with specialized tools

#### Configuration Management
- **Environment variables**: API keys, model configurations
- **Data directory resolution**: User-centric data storage
- **Collection validation**: Ensures valid collection/language combinations

### Memory Management and Performance

- **Streaming Processing**: ArticleCollection uses generators for large datasets
- **Lazy Loading**: JSON files loaded on-demand
- **Progress Tracking**: Rich progress bars throughout pipeline
- **Error Recovery**: Graceful handling with detailed user feedback
- **Vector Caching**: Persistent ChromaDB storage for efficient retrieval

## Programming Interface

### Basic Usage

```python
from cdc_text_corpora import CDCCorpus
from cdc_text_corpora.qa import RAGEngine

# Initialize corpus manager
corpus = CDCCorpus()

# Download and parse data
corpus.download_html_articles("pcd")
result = corpus.load_parse_save_html_articles("pcd", language="en")

# Load articles as DataFrame
df = corpus.load_json_articles_as_dataframe(collection="pcd")

# Get collection statistics
stats = corpus.get_collection_stats("pcd")
```

### RAG Question Answering

```python
from cdc_text_corpora.qa import RAGEngine

# Initialize RAG engine
rag = RAGEngine(corpus, llm_provider="anthropic")

# Index articles for semantic search
rag.index_articles(collection="pcd")

# Ask questions
result = rag.ask_question(
    "What are the main risk factors for diabetes?",
    collection_filter="pcd",
    include_sources=True
)

print(result['answer'])
for source in result['sources']:
    print(f"Source: {source['title']}")
```

### Interactive Agentic RAG

```python
from cdc_text_corpora.qa import AgenticRAG, AgentConfig

# Create agent configuration
config = AgentConfig(
    collection_filter="eid",
    relevance_cutoff=8,
    max_evidence_pieces=5
)

# Initialize and run agentic RAG
agentic_rag = AgenticRAG(config=config)
agentic_rag.run()  # Starts interactive Q&A session
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | `openai` |
| `DEFAULT_LLM_MODEL` | Default model name | `gpt-4o-mini` |
| `DEFAULT_EMBEDDING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `DEFAULT_CHUNK_SIZE` | Text chunk size | `1000` |
| `DEFAULT_CHUNK_OVERLAP` | Chunk overlap | `200` |

### Data Directory Structure

```
cdc-corpus-data/
├── cdc_corpus_df.csv           # Metadata
├── html-outputs/               # Downloaded ZIP files
│   ├── pcd.zip
│   ├── eid.zip
│   └── mmwr.zip
├── json-html/                  # Extracted HTML files
│   ├── pcd/
│   ├── eid/
│   └── mmwr/
├── json-parsed/                # Parsed JSON articles
│   ├── pcd_en_20240101_120000.json
│   └── eid_en_20240101_120000.json
└── chroma_db/                  # Vector database
```

## Supported Models

### LLM Providers

**OpenAI:**
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

**Anthropic:**
- claude-3-5-sonnet, claude-3-5-haiku, claude-3-sonnet, claude-3-haiku, claude-3-opus

### Embedding Models

- all-MiniLM-L6-v2 (default)
- Any HuggingFace sentence-transformers model

## Data Collections

### Preventing Chronic Disease (PCD)
- **Years**: 2004-2023
- **Focus**: Chronic disease prevention research
- **Languages**: English, Spanish
- **Content**: Research articles, case studies, reviews

### Emerging Infectious Diseases (EID)
- **Years**: 1995-2023
- **Focus**: Infectious disease research and surveillance
- **Languages**: English
- **Content**: Research articles, outbreak reports, reviews

### Morbidity and Mortality Weekly Report (MMWR)
- **Years**: 1982-2023
- **Focus**: Public health surveillance and recommendations
- **Languages**: English, Spanish, French, Chinese
- **Content**: Surveillance reports, recommendations, case reports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{cdc_text_corpora,
  title={CDC Text Corpora: A Python Package for Accessing CDC Publications},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/cdc-text-corpora}
}
```

## Data Source

This package uses data from the CDC Text Corpora dataset:
- **Source**: [CDC Text Corpora for Learners](https://data.cdc.gov/National-Center-for-State-Tribal-Local-and-Territo/CDC-Text-Corpora-for-Learners-HTML-Mirrors-of-MMWR/ut5n-bmc3)
- **License**: Public Domain
- **Last Updated**: 2023

## Support

For issues, questions, or contributions:
- **Email**: [your-email]
- **Issues**: [GitHub Issues](https://github.com/your-username/cdc-text-corpora/issues)
- **Documentation**: [Project Wiki](https://github.com/your-username/cdc-text-corpora/wiki)
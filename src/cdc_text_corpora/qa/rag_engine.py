"""Refactored RAG Engine for CDC Text Corpora using composition pattern."""

from typing import List, Dict, Any, Optional
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.utils.config import ensure_data_directory
from .article_processor import ArticleProcessor
from .vector_store import VectorStoreManager

# Load environment variables from .env file
load_dotenv()


class RAGEngine:
    """
    RAG-based question answering engine for CDC Text Corpora.

    This refactored class uses composition to delegate vector operations to VectorStoreManager
    and document processing to ArticleProcessor, while focusing on LLM operations and
    question-answering pipeline management.
    """

    def __init__(
        self,
        corpus_manager: CDCCorpus,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        persist_directory: Optional[str] = None,
        # Agentic configuration parameters
        default_collection_filter: str = 'all',
        relevance_cutoff: int = 8,
        search_k: int = 10,
        max_evidence_pieces: int = 5,
        max_search_attempts: int = 3
    ) -> None:
        """
        Initialize the RAG engine.

        Args:
            corpus_manager: CDCCorpus instance for accessing parsed articles
            embedding_model: Name of the embedding model to use (defaults to env var or all-MiniLM-L6-v2)
            llm_model: Name of the LLM model for question answering (defaults to env var or gpt-4o-mini)
            llm_provider: LLM provider ('openai' or 'anthropic') (defaults to env var or openai)
            chunk_size: Size of text chunks for embedding (defaults to env var or 1000)
            chunk_overlap: Overlap between chunks (defaults to env var or 200)
            persist_directory: Directory to persist ChromaDB data (defaults to env var or data_dir/chroma_db)
            default_collection_filter: Default collection filter for agentic RAG
            relevance_cutoff: Minimum relevance score for evidence (1-10)
            search_k: Number of search results to retrieve
            max_evidence_pieces: Maximum pieces of evidence to gather
            max_search_attempts: Maximum search attempts before generating answer
        """
        self.corpus_manager = corpus_manager

        # Load configuration from environment variables with fallbacks
        self.embedding_model_name = embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
        self.llm_model_name = llm_model or os.getenv("DEFAULT_LLM_MODEL") or "gpt-4o-mini"
        provider = llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.llm_provider = provider.lower() if provider else "openai"
        self.chunk_size = chunk_size or int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

        # Store agentic configuration as instance properties
        self.default_collection_filter = default_collection_filter
        self.relevance_cutoff = relevance_cutoff
        self.search_k = search_k
        self.max_evidence_pieces = max_evidence_pieces
        self.max_search_attempts = max_search_attempts

        # Validate LLM provider
        if self.llm_provider not in ["openai", "anthropic"]:
            raise ValueError(f"Invalid LLM provider '{self.llm_provider}'. Must be 'openai' or 'anthropic'")

        # Set up persistence directory
        if persist_directory is None:
            # Use the corpus manager's data directory to ensure consistency
            corpus_data_dir = corpus_manager.get_data_directory()
            ensure_data_directory(str(corpus_data_dir))

            chroma_dir = corpus_data_dir / "chroma_db"
            chroma_dir.mkdir(exist_ok=True)
            persist_directory = str(chroma_dir)

        self.persist_directory = persist_directory

        # Initialize composed components
        self.article_processor = ArticleProcessor(
            corpus_manager=corpus_manager,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.vector_store = VectorStoreManager(
            embedding_model=self.embedding_model_name,
            persist_directory=self.persist_directory
        )

        # Initialize LLM components
        self.llm: Any = None

        # Agentic RAG components (initialized lazily)
        self._agent_config: Any = None
        self._agentic_rag: Any = None

        # Initialize LLM
        self._initialize_llm_components()

    def _initialize_llm_components(self) -> None:
        """Initialize LLM components."""
        # Initialize LLM based on provider
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> Any:
        """Initialize LLM based on the specified provider."""
        if self.llm_provider == "openai":
            return ChatOpenAI(
                model=self.llm_model_name,
                temperature=0.1
            )
        elif self.llm_provider == "anthropic":
            # Map common model names to Anthropic model names
            anthropic_models = {
                "claude-3-sonnet": "claude-3-sonnet-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3-opus": "claude-3-opus-20240229",
                "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku": "claude-3-5-haiku-20241022"
            }

            # Use the provided model name or map it if it's a common alias
            model_name = anthropic_models.get(self.llm_model_name, self.llm_model_name)

            return ChatAnthropic(
                model=model_name,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    @property
    def agent_config(self) -> Any:
        """
        Lazy property for AgentConfig initialization.

        Returns:
            AgentConfig instance with current RAGEngine settings
        """
        if self._agent_config is None:
            # Import here to avoid circular imports
            from .rag_agent import AgentConfig

            self._agent_config = AgentConfig(
                collection_filter=self.default_collection_filter,
                relevance_cutoff=self.relevance_cutoff,
                search_k=self.search_k,
                max_evidence_pieces=self.max_evidence_pieces,
                max_search_attempts=self.max_search_attempts,
                model_name=self.llm_model_name
            )

        return self._agent_config

    @property
    def agentic_rag(self) -> Any:
        """
        Lazy property for AgenticRAG initialization.

        Returns:
            AgenticRAG instance configured with this RAGEngine
        """
        if self._agentic_rag is None:
            # Import here to avoid circular imports
            from .rag_agent import AgenticRAG

            self._agentic_rag = AgenticRAG(
                corpus=self.corpus_manager,
                config=self.agent_config
            )

        return self._agentic_rag

    def index_articles(
        self,
        collection: Optional[str] = None,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Index articles from parsed JSON files into the vector database.

        Args:
            collection: Collection to index ('pcd', 'eid', 'mmwr'). If None, indexes all
            language: Language filter for articles

        Returns:
            Dictionary with indexing statistics
        """
        # Load articles using article processor
        articles = self.article_processor.load_json_articles(
            collection=collection,
            language=language
        )

        if not articles:
            print("No articles found to index.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }

        # Convert articles to documents
        documents = self.article_processor.create_documents(articles)

        # Chunk documents
        chunked_documents = self.article_processor.chunk_documents(documents)

        if not chunked_documents:
            print("No valid documents found after processing.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }

        # Index documents using vector store
        index_stats = self.vector_store.index_documents(chunked_documents)

        # Update stats with collection info
        index_stats.update({
            "collection": collection or "all",
            "language": language,
            "articles_processed": len(articles)
        })

        return index_stats

    def index_html_articles(
        self,
        collection: Optional[str] = None,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Index articles from HTML files directly into the vector database.

        Uses structure-aware HTML chunking with optimized CDC headers
        to preserve document semantics during indexing.

        Args:
            collection: Collection to index ('pcd', 'eid', 'mmwr'). If None, indexes all
            language: Language filter for articles

        Returns:
            Dictionary with indexing statistics
        """
        # Load HTML articles using article processor
        html_articles = self.article_processor.load_html_articles(
            collection=collection,
            language=language
        )

        if not html_articles:
            print("No HTML articles found to index.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }

        # Chunk HTML documents using structure-aware splitting
        chunked_documents = self.article_processor.chunk_html_documents(html_articles)

        if not chunked_documents:
            print("No valid documents found after HTML processing.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }

        # Index documents using vector store
        index_stats = self.vector_store.index_documents(chunked_documents)

        # Update stats with collection info
        index_stats.update({
            "collection": collection or "all",
            "language": language,
            "articles_processed": len(html_articles)
        })

        return index_stats

    def semantic_search(
        self,
        query: str,
        k: int = 5,
        collection_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the indexed articles.

        Args:
            query: Search query
            k: Number of results to return
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')

        Returns:
            List of search results with metadata
        """
        return self.vector_store.search(
            query=query,
            k=k,
            collection_filter=collection_filter,
            search_type="mmr"
        )

    # Delegate vector store operations
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self.vector_store.get_stats()

    def check_index_availability(self) -> Dict[str, Any]:
        """
        Check if vector index is available and return structured status.

        Returns:
            Dictionary with index availability status, stats, and total document count
        """
        result: Dict[str, Any] = {
            "index_exists": False,
            "total_documents": 0,
            "stats": {},
            "error": None
        }

        try:
            vector_stats = self.get_vectorstore_stats()
            result["stats"] = vector_stats

            if "total_documents" in vector_stats and vector_stats["total_documents"] > 0:
                result["index_exists"] = True
                result["total_documents"] = vector_stats["total_documents"]
            else:
                result["index_exists"] = False
                result["total_documents"] = 0

        except Exception as e:
            result["index_exists"] = False
            result["total_documents"] = 0
            result["error"] = str(e)

        return result

    # LLM management methods

    def test_llm_connection(self, test_message: str = "Hello! Please respond with 'Connection successful.'") -> Dict[str, Any]:
        """
        Test the connection to the current LLM.

        Args:
            test_message: Simple message to send to the LLM for testing

        Returns:
            Dictionary with test results including success status, response, and timing
        """
        import time

        test_result: Dict[str, Any] = {
            "provider": self.llm_provider,
            "model": self.llm_model_name,
            "success": False,
            "response": None,
            "error": None,
            "response_time": None,
            "api_key_configured": False
        }

        # Check if API key is configured
        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            test_result["api_key_configured"] = bool(api_key and api_key.strip())
        elif self.llm_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            test_result["api_key_configured"] = bool(api_key and api_key.strip())

        if not test_result["api_key_configured"]:
            test_result["error"] = f"API key not configured for {self.llm_provider}. Please set {self.llm_provider.upper()}_API_KEY in your .env file."
            return test_result

        try:
            start_time = time.time()

            # Test the LLM with a simple message
            response = self.llm.invoke(test_message)

            end_time = time.time()

            test_result["success"] = True
            test_result["response"] = response.content if hasattr(response, 'content') else str(response)
            test_result["response_time"] = round(end_time - start_time, 2)

        except Exception as e:
            test_result["error"] = str(e)
            test_result["response_time"] = round(time.time() - start_time, 2) if 'start_time' in locals() else None

        return test_result

    def check_data_availability(self, collection_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if parsed JSON data is available for the RAG pipeline.

        This method focuses only on checking parsed JSON files and article availability,
        not vector index status. Use check_index_availability() for index checking.

        Args:
            collection_filter: Optional collection filter to check specific collections

        Returns:
            Dictionary with JSON data availability status
        """
        status: Dict[str, Any] = {
            "parsed_articles_available": False,
            "collections_found": [],
            "total_articles": 0,
            "recommendations": []
        }

        # Check for parsed JSON files
        json_parsed_dir = self.corpus_manager.get_data_directory() / "json-parsed"

        if json_parsed_dir.exists():
            # Determine collections to check based on filter
            collections_to_check = (
                [collection_filter]
                if collection_filter and collection_filter != 'all'
                else ['pcd', 'eid', 'mmwr']
            )

            for collection in collections_to_check:
                # Look for any language files for this collection
                pattern = f"{collection}_*_*.json"
                json_files = list(json_parsed_dir.glob(pattern))
                if json_files:
                    status["collections_found"].append(collection)

            if status["collections_found"]:
                status["parsed_articles_available"] = True

                # Count total articles using the corpus iterable
                try:
                    articles = self.corpus_manager.load_json_articles_as_iterable(
                        collection=collection_filter if collection_filter != 'all' else None,
                        language='en'  # Default to English for counting
                    )
                    status["total_articles"] = len(articles)
                except Exception:
                    status["total_articles"] = 0

        # Generate recommendations based on data availability
        if not status["parsed_articles_available"]:
            status["recommendations"].append("Parse some collections first using: cdc-corpus parse --collection <name>")

        return status

    def create_vector_index(
        self, 
        collection: Optional[str] = None, 
        language: str = 'en',
        source_type: str = 'json'
    ) -> Dict[str, Any]:
        """
        Create vector index for articles without UI interactions.

        This method supports two indexing approaches:
        - 'json': Uses parsed JSON articles with structured metadata
        - 'html': Uses raw HTML files with structure-aware chunking (preserves document semantics)

        Args:
            collection: Collection to index ('pcd', 'eid', 'mmwr'). If None, indexes all
            language: Language filter for articles
            source_type: Source data type to use ('json' or 'html'). Defaults to 'json'

        Returns:
            Dictionary with indexing results including success status and stats

        Raises:
            ValueError: If source_type is not 'json' or 'html'
        """
        # Validate source_type parameter
        valid_source_types = ['json', 'html']
        if source_type not in valid_source_types:
            raise ValueError(f"Invalid source_type '{source_type}'. Must be one of: {valid_source_types}")

        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "stats": {},
            "source_type": source_type
        }

        # Check if index already exists using the dedicated method
        index_status = self.check_index_availability()
        if index_status["index_exists"]:
            result["success"] = True
            result["stats"] = index_status["stats"]
            result["stats"]["already_exists"] = True
            return result

        # Conditional data availability check based on source type
        if source_type == 'json':
            # Check if we have parsed JSON articles to index
            status = self.check_data_availability(collection)
            if not status["parsed_articles_available"]:
                result["error"] = "No parsed JSON articles found for indexing. Run 'cdc-corpus parse' first."
                return result
        # For HTML indexing, we skip the parsed JSON check as it works directly with extracted HTML files

        # Perform indexing based on source type
        try:
            if source_type == 'json':
                index_stats = self.index_articles(
                    collection=collection,
                    language=language
                )
            else:  # source_type == 'html'
                index_stats = self.index_html_articles(
                    collection=collection,
                    language=language
                )

            result["success"] = True
            result["stats"] = index_stats
            result["stats"]["already_exists"] = False

        except Exception as e:
            source_context = "JSON articles" if source_type == 'json' else "HTML files"
            result["error"] = f"Error indexing {source_context}: {str(e)}"

        return result

    def generate_answer(
        self,
        question: str,
        collection_filter: Optional[str] = None,
        max_turns: int = 10,
        relevance_cutoff: Optional[int] = None,
        search_k: Optional[int] = None,
        max_evidence_pieces: Optional[int] = None,
        max_search_attempts: Optional[int] = None
    ) -> str:
        """
        Generate an answer to a question using the agentic RAG system.

        This method provides a simple synchronous interface to the sophisticated
        multi-agent Q&A system, handling the asyncio complexity internally.
        Uses the RAGEngine's configured agentic components.

        Args:
            question: The research question to answer
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr', 'all').
                             If None, uses RAGEngine's default_collection_filter
            max_turns: Maximum number of agent decision cycles
            relevance_cutoff: Minimum relevance score for evidence (1-10). If None, uses RAGEngine default
            search_k: Number of search results to retrieve. If None, uses RAGEngine default
            max_evidence_pieces: Maximum pieces of evidence to gather. If None, uses RAGEngine default
            max_search_attempts: Maximum search attempts before generating answer. If None, uses RAGEngine default

        Returns:
            The generated answer as a string

        Raises:
            Exception: If the agentic RAG system encounters an error
        """
        # Use instance defaults if parameters not provided, otherwise create custom config
        if any(param is not None for param in [collection_filter, relevance_cutoff, search_k, max_evidence_pieces, max_search_attempts]):
            # Create custom configuration for this call
            from .rag_agent import AgenticRAG, AgentConfig

            custom_config = AgentConfig(
                collection_filter=collection_filter or self.default_collection_filter,
                relevance_cutoff=relevance_cutoff or self.relevance_cutoff,
                search_k=search_k or self.search_k,
                max_evidence_pieces=max_evidence_pieces or self.max_evidence_pieces,
                max_search_attempts=max_search_attempts or self.max_search_attempts,
                model_name=self.llm_model_name
            )

            custom_agentic_rag = AgenticRAG(corpus=self.corpus_manager, config=custom_config)
            agentic_system = custom_agentic_rag
        else:
            # Use the pre-configured instance agentic RAG system
            agentic_system = self.agentic_rag

        # Run the agentic workflow synchronously
        try:
            answer = asyncio.run(agentic_system.ask_question(question, max_turns=max_turns))
            return answer
        except Exception as e:
            raise Exception(f"Agentic RAG system error: {str(e)}")

    def __repr__(self) -> str:
        """String representation of the RAG engine."""
        return f"RAGEngine(embedding_model='{self.embedding_model_name}', llm_model='{self.llm_model_name}', llm_provider='{self.llm_provider}')"

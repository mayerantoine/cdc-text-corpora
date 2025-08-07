"""
Article Vector Store Indexing Module.

This module provides indexing functionality for parsed articles into a vector store.
It uses existing HTMLArticleLoader and parser components for consistency.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

# Import existing components
from cdc_text_corpora.core.parser import (
    HTMLArticleLoader,
    Article, 
    CDCCollections,
    create_parser
)
from cdc_text_corpora.utils.config import ensure_data_directory

# LangChain components for indexing
try:
    from langchain_core.documents import Document
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class IndexConfig:
    """Configuration for article indexing pipeline."""
    
    # Indexing settings
    batch_size: int = 50
    max_workers: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector store settings
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_directory: Optional[str] = None
    collection_name: str = "cdc_articles"
    
    # Indexing options
    validate_articles: bool = True
    skip_existing: bool = True
    progress_bar: bool = True


@dataclass 
class IndexStats:
    """Statistics for indexing run."""
    
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ArticleIndexer:
    """
    Article vector store indexer.
    
    This class indexes parsed articles into a vector store using existing
    HTMLArticleLoader and parser components for consistency.
    """
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the article indexer.
        
        Args:
            config: Indexing configuration. If None, uses defaults.
            data_dir: Custom data directory. If None, uses default.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. "
                "Install with: uv add langchain-core langchain-chroma langchain-huggingface langchain-text-splitters"
            )
            
        self.config = config or IndexConfig()
        # Ensure data directory exists
        self.data_dir = ensure_data_directory(data_dir) if data_dir else ensure_data_directory()
        
        # Set up persistence directory
        if self.config.persist_directory is None:
            chroma_dir = self.data_dir / "chroma_db"
            chroma_dir.mkdir(exist_ok=True)  # Ensure chroma_db directory exists
            self.config.persist_directory = str(chroma_dir)
            
            # Path verification logging
            print(f"🗂️  ArticleIndexer using data directory: {self.data_dir}")
            print(f"🔍 ArticleIndexer chroma database path: {self.config.persist_directory}")
            print(f"📁 Chroma directory exists: {chroma_dir.exists()}")
        
        # Initialize components
        self.embeddings = None
        self.text_splitter = None
        self.vectorstore = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embeddings, text splitter, and vector store."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            show_progress=False  # Disable HuggingFace internal progress bars to avoid clutter
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.config.persist_directory
        )
    
    def index_collection(
        self,
        collection: Union[str, CDCCollections],
        language: Optional[str] = None
    ) -> IndexStats:
        """
        Index a complete collection using HTMLArticleLoader and parser.
        
        Args:
            collection: Collection name ('pcd', 'eid', 'mmwr') or CDCCollections enum
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht'). If None, uses 'en'.
            
        Returns:
            IndexStats with results summary
        """
        import time
        start_time = time.time()
        
        # Normalize collection name
        if isinstance(collection, str):
            collection = collection.lower()
            collection_enum = CDCCollections(collection)
        else:
            collection_enum = collection
            collection = collection.value
        
        stats = IndexStats()
        
        try:
            self.logger.info(f"Starting indexing of {collection.upper()} collection")
            
            # Use HTMLArticleLoader to load HTML files (same as CDCCorpus)
            loader = HTMLArticleLoader(collection, '', language or 'en')
            loader.load_from_file()
            
            if not loader.articles_html:
                error_msg = f"No HTML files loaded for {collection}"
                self.logger.warning(error_msg)
                stats.errors.append(error_msg)
                return stats
            
            # Index all loaded HTML articles
            stats = self.index_all_articles(
                html_articles=loader.articles_html,
                collection=collection_enum,
                language=language or 'en'
            )
        
        except Exception as e:
            error_msg = f"Failed to index collection {collection}: {e}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.processing_time = time.time() - start_time
        return stats
    
    def index_all_articles(
        self,
        html_articles: Dict[str, str],
        collection: Union[str, CDCCollections],
        language: str = 'en'
    ) -> IndexStats:
        """
        Index all HTML articles by parsing them and adding to vector store.
        
        Args:
            html_articles: Dictionary of {relative_url: html_content}
            collection: Collection name or enum
            language: Language code
            
        Returns:
            IndexStats with indexing results
        """
        import time
        from tqdm.auto import tqdm
        
        start_time = time.time()
        
        # Normalize collection
        if isinstance(collection, str):
            collection = collection.lower()
            collection_enum = CDCCollections(collection)
        else:
            collection_enum = collection
            collection = collection.value
        
        stats = IndexStats()
        stats.total_files = len(html_articles)
        
        if not html_articles:
            self.logger.warning("No HTML articles provided for indexing")
            return stats
        
        self.logger.info(f"Indexing {len(html_articles)} articles from {collection.upper()}")
        
        # Create parser for this collection
        parser = create_parser(collection, '', language, html_articles, validate_articles=self.config.validate_articles)
        
        # Parse articles first (parser shows its own progress bar)
        parsed_articles, parsing_stats = parser.parse_all_articles()
        
        # Now show our progress bar for indexing phase only
        if self.config.progress_bar:
            print(f"\n✓ Parsed {len(parsed_articles)} articles from {collection.upper()}")
            print("Starting vector store indexing...")
        
        main_progress = None
        
        if not parsed_articles:
            error_msg = f"No articles parsed from {collection}"
            self.logger.warning(error_msg)
            stats.errors.append(error_msg)
            if main_progress:
                main_progress.close()
            return stats
        
        # Convert parsed articles to documents
        all_documents = []
        
        for relative_url, article in parsed_articles.items():
            try:
                # Convert article to LangChain documents
                documents = self._article_to_documents(article, relative_url)
                all_documents.extend(documents)
                stats.processed_files += 1
                
            except Exception as e:
                error_msg = f"Failed to convert article {relative_url}: {e}"
                self.logger.error(error_msg)
                stats.errors.append(error_msg)
                stats.failed_files += 1
        
        # Index all documents in batches with progress bar
        if all_documents:
            batch_size = self.config.batch_size
            total_batches = (len(all_documents) + batch_size - 1) // batch_size
            
            # Create progress bar for indexing phase
            if self.config.progress_bar:
                main_progress = tqdm(
                    total=total_batches,
                    desc=f"Indexing {len(all_documents)} document chunks",
                    unit="batches"
                )
            
            batches_processed = 0
            
            for i in range(0, len(all_documents), batch_size):
                batch_docs = all_documents[i:i + batch_size]
                
                try:
                    # Check for existing documents if skip_existing is enabled
                    if self.config.skip_existing:
                        batch_docs = self._filter_existing_documents(batch_docs)
                        if not batch_docs:
                            stats.skipped_files += len(all_documents[i:i + batch_size])
                            batches_processed += 1
                            if main_progress:
                                main_progress.set_postfix(chunks=stats.total_chunks, skipped=stats.skipped_files)
                                main_progress.update(1)
                            continue
                    
                    # Add documents to vector store
                    self.vectorstore.add_documents(batch_docs)
                    stats.total_chunks += len(batch_docs)
                    batches_processed += 1
                    
                    if main_progress:
                        main_progress.set_postfix(chunks=stats.total_chunks, processed=stats.processed_files)
                        main_progress.update(1)
                    
                except Exception as e:
                    error_msg = f"Failed to index batch {i//batch_size + 1}: {e}"
                    self.logger.error(error_msg)
                    stats.errors.append(error_msg)
                    stats.failed_files += len(batch_docs)
                    batches_processed += 1
                    
                    if main_progress:
                        main_progress.set_postfix(chunks=stats.total_chunks, errors=len(stats.errors))
                        main_progress.update(1)
            
            if main_progress:
                main_progress.close()
        
        # Final status message
        if self.config.progress_bar:
            print(f"✓ Indexed {stats.total_chunks} document chunks from {stats.processed_files} articles")
        
        self.logger.info(f"Successfully indexed {stats.total_chunks} document chunks")
        
        stats.processing_time = time.time() - start_time
        return stats
    
    
    def _filter_existing_documents(self, documents: List[Document]) -> List[Document]:
        """Filter out documents that already exist in vector store."""
        filtered_docs = []
        
        for doc in documents:
            # Check if document with this source already exists
            relative_url = doc.metadata.get("relative_url", "")
            chunk_index = doc.metadata.get("chunk_index", 0)
            
            if not self._is_document_indexed(relative_url, chunk_index):
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _is_document_indexed(self, relative_url: str, chunk_index: int) -> bool:
        """Check if a specific document chunk is already indexed."""
        try:
            # Query vector store for documents with this relative_url and chunk_index
            results = self.vectorstore.similarity_search(
                query="",  # Empty query
                k=1,
                filter={
                    "relative_url": relative_url,
                    "chunk_index": chunk_index
                }
            )
            return len(results) > 0
        except Exception:
            # If query fails, assume not indexed
            return False
    
    def _article_to_documents(self, article: Article, relative_url: str) -> List[Document]:
        """Convert Article to LangChain Documents with chunking."""
        documents = []
        
        # Combine text content for chunking
        content_parts = []
        if article.title:
            content_parts.append(f"Title: {article.title}")
        if article.abstract:
            content_parts.append(f"Abstract: {article.abstract}")
        if article.full_text:
            content_parts.append(f"Full Text: {article.full_text}")
        
        combined_content = "\n\n".join(content_parts)
        
        if not combined_content.strip():
            return documents
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(combined_content)
        
        # Create document for each chunk
        for i, chunk in enumerate(chunks):
            metadata = {
                "title": article.title,
                "collection": article.collection.value if article.collection else "unknown",
                "language": article.language or "unknown",
                "url": article.url or "",
                "relative_url": article.relative_url or relative_url,
                "authors": ", ".join(article.authors) if article.authors else "",
                "publication_date": article.publication_date or "",
                "journal": article.journal or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": relative_url
            }
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory,
                "embedding_model": self.config.embedding_model
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_vectorstore(self) -> bool:
        """Clear all documents from vector store."""
        try:
            # Delete and recreate collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.config.persist_directory
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear vector store: {e}")
            return False


def create_article_indexer(
    config: Optional[IndexConfig] = None,
    data_dir: Optional[str] = None
) -> ArticleIndexer:
    """
    Factory function to create ArticleIndexer instance.
    
    Args:
        config: Indexing configuration
        data_dir: Custom data directory
        
    Returns:
        ArticleIndexer instance
    """
    return ArticleIndexer(config=config, data_dir=data_dir)
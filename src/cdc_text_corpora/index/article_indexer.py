"""
HTML to Vector Store Indexing Module.

This module provides a streamlined pipeline that indexes HTML files directly
into a vector store without intermediate JSON storage. The workflow:
1. Unzip HTML collections
2. Parse HTML articles in memory
3. Index directly to vector store in batches
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import zipfile
from tqdm.auto import tqdm
import tempfile
import logging

# Import existing components
from cdc_text_corpora.core.parser import (
    CDCArticleParser, 
    Article, 
    CDCCollections,
    create_parser
)
from cdc_text_corpora.utils.config import get_data_directory, get_collection_zip_path, ensure_data_directory

# LangChain components for indexing
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    HTML to Vector Store article indexer.
    
    This class handles the complete pipeline from downloaded ZIP files
    to indexed vector store without intermediate JSON files.
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
        self.config = config or IndexConfig()
        # Ensure data directory exists
        if data_dir:
            self.data_dir = ensure_data_directory(data_dir)
        else:
            self.data_dir = ensure_data_directory()
        
        # Set up persistence directory
        if self.config.persist_directory is None:
            chroma_dir = self.data_dir / "chroma_db"
            chroma_dir.mkdir(exist_ok=True)  # Ensure chroma_db directory exists
            self.config.persist_directory = str(chroma_dir)
        
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
            show_progress=self.config.progress_bar
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
        Index a complete collection from ZIP to vector store.
        
        Args:
            collection: Collection name ('pcd', 'eid', 'mmwr') or CDCCollections enum
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht'). If None, indexes all.
            
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
            # Check if ZIP file exists
            zip_path = get_collection_zip_path(collection)
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file not found: {zip_path}")
            
            # Process the collection
            self.logger.info(f"Starting indexing of {collection.upper()} collection")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP to temporary directory
                extract_path = Path(temp_dir) / collection
                self._extract_zip(zip_path, extract_path)
                
                # Find HTML files to process
                html_files = self._find_html_files(extract_path, language)
                stats.total_files = len(html_files)
                
                if stats.total_files == 0:
                    self.logger.warning(f"No HTML files found for {collection}")
                    return stats
                
                # Process files in batches
                stats = self._index_html_files_batch(
                    html_files, 
                    collection_enum, 
                    stats
                )
        
        except Exception as e:
            error_msg = f"Failed to index collection {collection}: {e}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        stats.processing_time = time.time() - start_time
        return stats
    
    def _extract_zip(self, zip_path: Path, extract_path: Path):
        """Extract ZIP file to specified path."""
        extract_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        self.logger.info(f"Extracted {zip_path} to {extract_path}")
    
    def _find_html_files(
        self, 
        extract_path: Path, 
        language: Optional[str] = None
    ) -> List[Path]:
        """Find HTML files matching language filter."""
        html_files = []
        
        # Recursively find all HTML files
        for html_file in extract_path.rglob("*.htm*"):
            if language is None:
                html_files.append(html_file)
            else:
                # Filter by language using file path conventions
                if self._matches_language_filter(html_file, language):
                    html_files.append(html_file)
        
        return sorted(html_files)
    
    def _matches_language_filter(self, html_file: Path, language: str) -> bool:
        """Check if HTML file matches language filter."""
        # Simple heuristic based on path structure
        file_str = str(html_file).lower()
        
        language_patterns = {
            'en': ['_en_', '/en/', 'english'],
            'es': ['_es_', '/es/', 'spanish', 'espanol'],
            'fr': ['_fr_', '/fr/', 'french', 'francais'],
            'zhs': ['_zhs_', '/zhs/', 'chinese_simplified'],
            'zht': ['_zht_', '/zht/', 'chinese_traditional']
        }
        
        patterns = language_patterns.get(language.lower(), [])
        return any(pattern in file_str for pattern in patterns) if patterns else True
    
    def _index_html_files_batch(
        self,
        html_files: List[Path],
        collection: CDCCollections,
        stats: IndexStats
    ) -> IndexStats:
        """Index HTML files in batches."""
        
        # Create parser for this collection
        parser = create_parser(collection.value)
        
        # Process files in batches
        progress_desc = f"Indexing {collection.value.upper()} files"
        
        if self.config.progress_bar:
            file_progress = tqdm(
                total=len(html_files),
                desc=progress_desc,
                unit="files"
            )
        
        for i in range(0, len(html_files), self.config.batch_size):
            batch_files = html_files[i:i + self.config.batch_size]
            batch_stats = self._index_batch(batch_files, parser, collection)
            
            # Update stats
            stats.processed_files += batch_stats.processed_files
            stats.skipped_files += batch_stats.skipped_files
            stats.failed_files += batch_stats.failed_files
            stats.total_chunks += batch_stats.total_chunks
            stats.errors.extend(batch_stats.errors)
            
            if self.config.progress_bar:
                file_progress.update(len(batch_files))
        
        if self.config.progress_bar:
            file_progress.close()
        
        return stats
    
    def _index_batch(
        self,
        html_files: List[Path],
        parser: CDCArticleParser,
        collection: CDCCollections
    ) -> IndexStats:
        """Index a batch of HTML files."""
        batch_stats = IndexStats()
        documents = []
        
        # Parse HTML files to articles
        for html_file in html_files:
            try:
                # Check if already indexed (if skip_existing is True)
                if self.config.skip_existing and self._is_already_indexed(html_file):
                    batch_stats.skipped_files += 1
                    continue
                
                # Read and parse HTML
                article = self._parse_html_file(html_file, parser, collection)
                
                if article is None:
                    batch_stats.failed_files += 1
                    continue
                
                # Convert article to documents
                article_docs = self._article_to_documents(article, html_file)
                documents.extend(article_docs)
                batch_stats.processed_files += 1
                
            except Exception as e:
                error_msg = f"Failed to process {html_file}: {e}"
                self.logger.error(error_msg)
                batch_stats.errors.append(error_msg)
                batch_stats.failed_files += 1
        
        # Index documents in batch
        if documents:
            try:
                self.vectorstore.add_documents(documents)
                batch_stats.total_chunks = len(documents)
                self.logger.info(f"Indexed {len(documents)} chunks from {batch_stats.processed_files} articles")
            except Exception as e:
                error_msg = f"Failed to index batch: {e}"
                self.logger.error(error_msg)
                batch_stats.errors.append(error_msg)
        
        return batch_stats
    
    def _parse_html_file(
        self,
        html_file: Path,
        parser: CDCArticleParser,
        collection: CDCCollections
    ) -> Optional[Article]:
        """Parse a single HTML file to Article."""
        try:
            # Read HTML content
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Parse using collection-specific parser
            article = parser.parse_article(html_content)
            
            # Set collection and relative URL
            article.collection = collection
            article.relative_url = str(html_file.name)
            
            # Validate if enabled
            if self.config.validate_articles:
                if not self._validate_article(article):
                    self.logger.warning(f"Article validation failed for {html_file}")
                    return None
            
            return article
            
        except Exception as e:
            self.logger.error(f"Failed to parse {html_file}: {e}")
            return None
    
    def _validate_article(self, article: Article) -> bool:
        """Basic article validation."""
        return (
            bool(article.title.strip()) and
            bool(article.url or article.relative_url) and
            bool(article.full_text.strip() or article.abstract.strip())
        )
    
    def _article_to_documents(self, article: Article, html_file: Path) -> List[Document]:
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
                "relative_url": article.relative_url or str(html_file.name),
                "authors": ", ".join(article.authors) if article.authors else "",
                "publication_date": article.publication_date or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": str(html_file)
            }
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def _is_already_indexed(self, html_file: Path) -> bool:
        """Check if file is already indexed in vector store."""
        try:
            # Query vector store for documents with this source file
            results = self.vectorstore.similarity_search(
                query="",  # Empty query
                k=1,
                filter={"source_file": str(html_file)}
            )
            return len(results) > 0
        except Exception:
            # If query fails, assume not indexed
            return False
    
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
"""Article processing module for CDC Text Corpora.

This module provides functionality for loading, processing, and converting
CDC articles into LangChain Document objects for vector indexing.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.core.parser import Article, HTMLArticleLoader


class ArticleProcessor:
    """
    Handles article loading, processing, and document creation for RAG operations.

    This class provides methods to load articles from JSON and HTML sources,
    convert them to LangChain Documents, apply chunking strategies, and format
    documents for RAG contexts.
    """

    def __init__(
        self,
        corpus_manager: CDCCorpus,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """
        Initialize the ArticleProcessor.

        Args:
            corpus_manager: CDCCorpus instance for accessing articles
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks in characters
        """
        self.corpus_manager = corpus_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_json_articles(
        self,
        collection: Optional[str] = None,
        language: str = 'en'
    ) -> List[Article]:
        """
        Load articles from parsed JSON files.

        Args:
            collection: Collection to load ('pcd', 'eid', 'mmwr'). If None, loads all
            language: Language filter for articles

        Returns:
            List of Article objects
        """
        # Get articles from the corpus manager using the existing iterable method
        articles_iterable = self.corpus_manager.load_json_articles_as_iterable(
            collection=collection,
            language=language
        )

        # Convert to list for processing
        return list(articles_iterable)

    def load_html_articles(
        self,
        collection: Optional[str] = None,
        language: str = 'en'
    ) -> Dict[str, str]:
        """
        Load raw HTML content directly from HTML files.

        Args:
            collection: Collection to load ('pcd', 'eid', 'mmwr'). If None, loads all
            language: Language filter for articles

        Returns:
            Dictionary mapping relative file paths to HTML content strings
            Format: {relative_path: html_content}
        """
        # Determine collections to process
        if collection is None:
            collections_to_process = ['pcd', 'eid', 'mmwr']
        else:
            # Validate collection parameter
            valid_collections = ['pcd', 'eid', 'mmwr']
            if collection.lower() not in valid_collections:
                raise ValueError(f"Invalid collection '{collection}'. Valid options: {valid_collections}")
            collections_to_process = [collection.lower()]

        # Aggregate HTML content from all collections
        all_html_articles: Dict[str, str] = {}

        for coll in collections_to_process:
            try:
                # Create HTMLArticleLoader for this collection
                loader = HTMLArticleLoader(coll, '', language)

                # Load HTML files
                loader.load_from_file()

                # Add to aggregated results if HTML was loaded
                if loader.articles_html:
                    # Prefix keys with collection name to avoid conflicts
                    for relative_path, html_content in loader.articles_html.items():
                        # Use collection prefix to ensure unique keys
                        prefixed_path = f"{coll}/{relative_path}"
                        all_html_articles[prefixed_path] = html_content

                    print(f"✅ Loaded {len(loader.articles_html)} HTML articles from {coll.upper()}")
                else:
                    print(f"⚠️  No HTML articles found for {coll.upper()}")

            except Exception as e:
                print(f"❌ Error loading HTML articles from {coll.upper()}: {e}")
                continue

        if not all_html_articles:
            print("No HTML articles found for the specified criteria")
        else:
            total_articles = len(all_html_articles)
            collections_loaded = ", ".join([c.upper() for c in collections_to_process])
            print(f"✅ Total: {total_articles} HTML articles loaded from {collections_loaded} ({language})")

        return all_html_articles

    def create_documents(self, articles: List[Article]) -> List[Document]:
        """
        Convert Article objects to LangChain Document objects.

        Args:
            articles: List of Article objects to convert

        Returns:
            List of Document objects with content and metadata
        """
        documents: List[Document] = []

        try:
            from tqdm import tqdm
            articles_iterator: Union[List[Article], Any] = tqdm(articles, desc="Creating documents", unit=" articles")
        except ImportError:
            articles_iterator = articles

        for article in articles_iterator:
            # Skip articles without meaningful content
            if not article.title and not article.abstract and not article.full_text:
                continue

            # Create document content by combining title, abstract, and full text
            content_parts: List[str] = []

            if article.title:
                content_parts.append(f"Title: {article.title}")

            if article.abstract:
                content_parts.append(f"Abstract: {article.abstract}")

            if article.full_text:
                content_parts.append(f"Full Text: {article.full_text}")

            content = "\n\n".join(content_parts)

            # Create metadata dictionary
            metadata: Dict[str, Any] = {
                "title": article.title,
                "collection": article.collection.value if article.collection else "unknown",
                "journal": article.journal,
                "language": article.language,
                "url": article.url,
                "authors": ", ".join(article.authors) if article.authors else "",
                "publication_date": article.publication_date,
                "has_abstract": bool(article.abstract),
                "has_full_text": bool(article.full_text),
                "reference_count": len(article.references) if article.references else 0
            }

            # Create LangChain Document
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks using the configured text splitter.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []

        print("📄 Splitting documents into chunks...")

        try:
            from tqdm import tqdm

            # Use progress bar if available
            chunked_documents: List[Document] = []

            with tqdm(total=len(documents), desc="Chunking documents", unit="doc") as pbar:
                for doc in documents:
                    chunks = self.text_splitter.split_documents([doc])
                    chunked_documents.extend(chunks)
                    pbar.update(1)

            return chunked_documents

        except ImportError:
            print("   Splitting documents (this may take a moment)...")
            return self.text_splitter.split_documents(documents)

    def chunk_html_documents(
        self,
        html_articles: Dict[str, str],
        headers_to_split_on: Optional[List[Tuple[str, str]]] = None
    ) -> List[Document]:
        """
        Split HTML documents into chunks using HTMLSectionSplitter and RecursiveCharacterTextSplitter.

        This method provides structure-aware chunking that preserves HTML document semantics
        while ensuring manageable chunk sizes for RAG operations.

        Args:
            html_articles: Dictionary mapping relative paths to HTML content
            headers_to_split_on: List of (tag, header_name) tuples for section splitting.
                                If None, uses optimal headers based on CDC document structure analysis:
                                h1 (titles), h2 (major sections), h3 (sections), h4 (subsections), h5 (details).

        Returns:
            List of Document objects with section-aware chunks and enhanced metadata
        """
        if not html_articles:
            return []

        # Define optimal headers for CDC documents based on parser analysis if not provided
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("h1", "Title"),           # Article titles (PCD, MMWR)
                ("h2", "Major Section"),   # Main sections in PCD (Abstract, Methods, Results, Discussion)
                ("h3", "Section"),         # Main sections in EID/MMWR, subsections in PCD
                ("h4", "Subsection"),      # Author info, minor sections
                ("h5", "Detail")           # Detailed subsections
            ]

        # Create HTMLSectionSplitter
        html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

        all_chunked_documents: List[Document] = []

        try:
            from tqdm import tqdm
            articles_iterator: Union[Dict, Any] = tqdm(
                html_articles.items(),
                desc="Chunking HTML documents",
                unit="docs"
            )
        except ImportError:
            articles_iterator = html_articles.items()

        print("📄 Chunking HTML documents with structure-aware splitting...")

        for relative_path, html_content in articles_iterator:
            try:
                # Extract collection from prefixed path (e.g., "pcd/issues/2019/18_0093.htm")
                path_parts = relative_path.split('/', 1)
                if len(path_parts) == 2:
                    collection = path_parts[0]
                    clean_relative_path = path_parts[1]
                else:
                    collection = "unknown"
                    clean_relative_path = relative_path

                # First pass: Split by HTML sections
                html_sections = html_splitter.split_text(html_content)

                # Second pass: Apply RecursiveCharacterTextSplitter for size constraints
                size_constrained_chunks = self.text_splitter.split_documents(html_sections)

                # Create Document objects with enhanced metadata
                for i, chunk in enumerate(size_constrained_chunks):
                    # Extract section headers from HTMLSectionSplitter metadata
                    section_metadata = chunk.metadata.copy() if hasattr(chunk, 'metadata') else {}

                    # Add document-level metadata
                    enhanced_metadata = {
                        "source_path": relative_path,
                        "relative_path": clean_relative_path,
                        "collection": collection,
                        "chunk_index": i,
                        "total_chunks": len(size_constrained_chunks),
                        "chunk_type": "html_section",
                        "chunk_size": len(chunk.page_content),
                        # Include any section headers from HTMLSectionSplitter
                        **section_metadata
                    }

                    # Create new Document with enhanced metadata
                    chunked_document = Document(
                        page_content=chunk.page_content,
                        metadata=enhanced_metadata
                    )

                    all_chunked_documents.append(chunked_document)

            except Exception as e:
                print(f"Error chunking HTML document {relative_path}: {e}")
                continue

        total_chunks = len(all_chunked_documents)
        total_articles = len(html_articles)

        print(f"✅ Created {total_chunks} HTML chunks from {total_articles} articles")
        print(f"   Average chunks per article: {total_chunks/total_articles:.1f}")

        return all_chunked_documents

    def format_docs_for_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents for RAG context with numbered references.

        Args:
            docs: List of Document objects to format

        Returns:
            Formatted string with numbered document references
        """
        formatted_docs: List[str] = []

        for i, doc in enumerate(docs, 1):
            # Extract metadata for citation
            metadata = doc.metadata
            collection = metadata.get('collection', 'Unknown')
            title = metadata.get('title', 'Unknown Title')

            # Format the document with numbered reference
            formatted_doc = f"Source [{i}] - {collection.upper()}: {title}\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)

        return "\n---\n".join(formatted_docs)

    def validate_articles(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Validate a list of articles and return statistics.

        Args:
            articles: List of Article objects to validate

        Returns:
            Dictionary with validation statistics and issues
        """
        stats = {
            "total_articles": len(articles),
            "valid_articles": 0,
            "missing_title": 0,
            "missing_abstract": 0,
            "missing_full_text": 0,
            "missing_authors": 0,
            "missing_url": 0,
            "empty_articles": 0
        }

        for article in articles:
            # Check for completely empty articles
            if not article.title and not article.abstract and not article.full_text:
                stats["empty_articles"] += 1
                continue

            # Count valid articles (have at least title OR full text)
            if article.title or article.full_text:
                stats["valid_articles"] += 1

            # Count missing fields
            if not article.title:
                stats["missing_title"] += 1
            if not article.abstract:
                stats["missing_abstract"] += 1
            if not article.full_text:
                stats["missing_full_text"] += 1
            if not article.authors:
                stats["missing_authors"] += 1
            if not article.url:
                stats["missing_url"] += 1

        return stats

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing configuration statistics.

        Returns:
            Dictionary with processor configuration details
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.text_splitter._separators,
            "data_directory": str(self.corpus_manager.data_dir)
        }

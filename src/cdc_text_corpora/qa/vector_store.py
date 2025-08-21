"""Vector store management module for CDC Text Corpora.

This module provides functionality for vector database operations including
indexing, searching, and managing CDC article embeddings using ChromaDB.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.prompt import Confirm


class VectorStoreManager:
    """
    Manages vector database operations for CDC Text Corpora.
    
    This class handles ChromaDB initialization, document indexing, semantic search,
    and vector store maintenance operations.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        corpus_manager: Optional[Any] = None,
        console: Optional[Console] = None
    ) -> None:
        """
        Initialize the VectorStoreManager.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model to use
            persist_directory: Directory path for persisting ChromaDB data
            corpus_manager: Optional CDCCorpus instance for IndexManager integration
            console: Optional Rich console for user interaction
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.corpus_manager = corpus_manager
        self.console = console or Console()
        
        # Initialize components
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[Any] = None
        
        # Initialize IndexManager if corpus_manager is provided
        self.index_manager: Optional[Any] = None
        if self.corpus_manager:
            # Import here to avoid circular imports
            from cdc_text_corpora.utils.index_manager import IndexManager
            self.index_manager = IndexManager(self.corpus_manager, self.console)
        
        # Initialize the vector store
        self.initialize_store()
    
    def initialize_store(self) -> None:
        """Initialize the vector store with smart index resolution."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            show_progress=False
        )
        
        # Ensure persist directory exists
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Smart index resolution logic
        if self.index_manager:
            self._resolve_index_source()
        
        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize retriever with default configuration
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def _resolve_index_source(self) -> None:
        """
        Resolve the best index source using priority order:
        1. Existing local index
        2. HTML articles available → delegate to RAGEngine.index_articles() 
        3. No HTML articles → extract bundled pre-built index
        4. Fallback → show guidance
        """
        if not self.index_manager:
            return
            
        # 1. Check for existing valid local index
        if self.index_manager.has_existing_index():
            self.console.print("[green]✅ Using existing ChromaDB index[/green]")
            return
            
        # 2. Check for HTML articles to build fresh index
        if self.index_manager.has_html_articles():
            self.console.print("[yellow]📚 HTML articles found - will build fresh index when needed[/yellow]")
            # Note: Actual indexing will be handled by RAGEngine.index_articles() method
            return
            
        # 3. No HTML articles → offer bundled index extraction
        if self.index_manager.get_bundled_index_path():
            self._handle_bundled_index_extraction()
        else:
            # 4. Fallback → show guidance
            self._show_setup_guidance()
    
    def _handle_bundled_index_extraction(self) -> None:
        """Handle bundled index extraction with user confirmation."""
        if not self.index_manager:
            return
            
        self.console.print("[yellow]📦 No local index found, but bundled index is available[/yellow]")
        
        # Check if this is likely an interactive session
        try:
            should_extract = Confirm.ask(
                "Would you like to extract the bundled pre-built index for immediate use?",
                default=True
            )
            
            if should_extract:
                success = self.index_manager.extract_bundled_index()
                if success:
                    self.console.print("[green]🚀 Ready for questions! Bundled index extracted.[/green]")
                else:
                    self._show_setup_guidance()
            else:
                self._show_setup_guidance()
                
        except (KeyboardInterrupt, EOFError):
            # Non-interactive environment or user interrupted
            self.console.print("[yellow]⚠️  In non-interactive mode. Use 'cdc-corpus index --use-bundled' to extract.[/yellow]")
            self._show_setup_guidance()
    
    def _show_setup_guidance(self) -> None:
        """Show helpful setup guidance."""
        if self.index_manager:
            self.index_manager.show_guidance()
        else:
            self.console.print("""
[yellow]⚠️  No ChromaDB index available.[/yellow]

[bold cyan]Quick Setup Options:[/bold cyan]
1. Download data: [bold]cdc-corpus download --collection pcd[/bold]
2. Parse articles: [bold]cdc-corpus parse --collection pcd[/bold]
3. Start Q&A: [bold]cdc-corpus qa[/bold]
            """)
    
    def check_index_exists(self) -> bool:
        """
        Check if the vector index exists and contains documents.
        
        Returns:
            True if index exists and has documents, False otherwise
        """
        if not self.vectorstore:
            return False
        
        try:
            doc_count = self.vectorstore._collection.count()
            return doc_count > 0
        except Exception:
            return False
    
    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Index documents into the vector database.
        
        Args:
            documents: List of LangChain Document objects to index
            
        Returns:
            Dictionary with indexing statistics
        """
        if not documents:
            return {
                "documents_processed": 0,
                "total_chunks": 0,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        
        total_docs = len(documents)
        print(f"🔍 Indexing {total_docs} documents with {self.embedding_model_name}...")
        
        # Use batched approach for better progress tracking and memory management
        batch_size = 50
        processed_count = 0
        
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_docs, desc="Creating embeddings", unit="doc")
        except ImportError:
            progress_bar = None
            print("   Processing in batches (this may take several minutes)...")
        
        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                if i == 0:
                    # Create initial vectorstore with first batch
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                else:
                    # Add remaining batches to existing vectorstore
                    if self.vectorstore is not None:
                        self.vectorstore.add_documents(batch)
                
                processed_count += len(batch)
                
                # Update progress bar
                if progress_bar:
                    progress_bar.update(len(batch))
                else:
                    print(f"   Processed {processed_count}/{total_docs} documents")
        
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Update retriever with new vectorstore
        if self.vectorstore is not None:
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        
        print(f"✅ Indexing complete! Processed {processed_count} documents")
        
        return {
            "documents_processed": processed_count,
            "total_chunks": processed_count,
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory
        }
    
    def search(
        self,
        query: str,
        k: int = 5,
        collection_filter: Optional[str] = None,
        search_type: str = "mmr"
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on indexed documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            search_type: Type of search ('similarity', 'mmr')
            
        Returns:
            List of search results with content and metadata
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized")
        
        # Prepare search kwargs
        search_kwargs: Dict[str, Any] = {"k": k}
        
        # Apply collection filter if specified
        if collection_filter:
            filter_dict = {"collection": collection_filter.lower()}
            search_kwargs["filter"] = filter_dict
        
        # Update retriever with search parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        # Perform search
        docs = self.retriever.invoke(query)
        
        # Format results
        results: List[Dict[str, Any]] = []
        for doc in docs:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "title": doc.metadata.get("title", "Unknown"),
                "collection": doc.metadata.get("collection", "unknown"),
                "url": doc.metadata.get("url", ""),
                "relevance_score": getattr(doc, 'relevance_score', None)
            }
            results.append(result)
        
        return results
    
    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 5,
        collection_filter: Optional[str] = None
    ) -> Any:
        """
        Get a configured retriever for the vector store.
        
        Args:
            search_type: Type of search ('similarity', 'mmr')
            k: Number of documents to retrieve
            collection_filter: Optional collection filter
            
        Returns:
            Configured retriever object
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized")
        
        search_kwargs: Dict[str, Any] = {"k": k}
        
        if collection_filter:
            filter_dict = {"collection": collection_filter.lower()}
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        if not self.vectorstore:
            return {
                "error": "Vector store not initialized",
                "persist_directory": self.persist_directory
            }
        
        try:
            # Get the collection from ChromaDB
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample metadata to understand the data
            sample_results = collection.peek(limit=10)
            collections = set()
            languages = set()
            
            if sample_results and 'metadatas' in sample_results and sample_results['metadatas'] is not None:
                for metadata in sample_results['metadatas']:
                    if metadata:
                        collections.add(metadata.get('collection', 'unknown'))
                        languages.add(metadata.get('language', 'unknown'))
            
            return {
                "total_documents": count,
                "collections": list(collections),
                "languages": list(languages),
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        
        except Exception as e:
            return {
                "error": f"Could not retrieve stats: {str(e)}",
                "persist_directory": self.persist_directory
            }
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        if not self.vectorstore:
            print("Vector store not initialized")
            return
        
        try:
            # Delete and recreate the collection
            self.vectorstore.delete_collection()
            
            # Reinitialize the vector store
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Update retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print("Vector store cleared successfully")
        
        except Exception as e:
            print(f"Error clearing vector store: {e}")
    
    def remove_documents(self, filter_criteria: Dict[str, Any]) -> int:
        """
        Remove documents from the vector store based on filter criteria.
        
        Args:
            filter_criteria: Dictionary with filter conditions
            
        Returns:
            Number of documents removed
        """
        if not self.vectorstore:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Note: ChromaDB doesn't have direct document removal by filter
            # This would require implementing a more complex solution
            # For now, we'll raise a NotImplementedError
            raise NotImplementedError(
                "Document removal by filter is not yet implemented. "
                "Use clear() to remove all documents."
            )
        
        except Exception as e:
            print(f"Error removing documents: {e}")
            return 0
    
    def _show_progress(self, operation_name: str, emoji: str, operation_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Show progress indicator for long-running operations."""
        try:
            from tqdm import tqdm
            import time
            
            with tqdm(total=1, desc=operation_name, unit="batch") as pbar:
                start_time = time.time()
                result = operation_func(*args, **kwargs)
                operation_time = time.time() - start_time
                pbar.update(1)
                pbar.set_postfix({"time": f"{operation_time:.1f}s"})
                return result
        
        except ImportError:
            print(f"{emoji} {operation_name} (this may take a moment)...")
            return operation_func(*args, **kwargs)
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current vector store configuration.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory,
            "vectorstore_initialized": self.vectorstore is not None,
            "has_documents": self.check_index_exists()
        }
    
    def __repr__(self) -> str:
        """String representation of the VectorStoreManager."""
        return f"VectorStoreManager(embedding_model='{self.embedding_model_name}', persist_directory='{self.persist_directory}')"
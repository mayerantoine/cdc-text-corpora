"""RAG Engine for CDC Text Corpora using LangChain."""

from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.core.parser import Article, CDCCollections

# Load environment variables from .env file
load_dotenv()


class RAGEngine:
    """
    RAG-based question answering engine for CDC Text Corpora.
    
    This class provides semantic search and question-answering capabilities
    using LangChain, ChromaDB for vector storage, and various embedding models.
    """
    
    def __init__(
        self,
        corpus_manager: CDCCorpus,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the RAG engine.
        
        Args:
            corpus_manager: CDCCorpus instance for accessing parsed articles
            embedding_model: Name of the embedding model to use (defaults to env var or all-MiniLM-L6-v2)
            llm_model: Name of the LLM model for question answering (defaults to env var or gpt-3.5-turbo)
            llm_provider: LLM provider ('openai' or 'anthropic') (defaults to env var or openai)
            chunk_size: Size of text chunks for embedding (defaults to env var or 1000)
            chunk_overlap: Overlap between chunks (defaults to env var or 200)
            persist_directory: Directory to persist ChromaDB data (defaults to env var or data_dir/chroma_db)
        """
        self.corpus_manager = corpus_manager
        
        # Load configuration from environment variables with fallbacks
        self.embedding_model_name = embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.llm_model_name = llm_model or os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        provider = llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.llm_provider = provider.lower() if provider else "openai"
        self.chunk_size = chunk_size or int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
        
        # Validate LLM provider
        if self.llm_provider not in ["openai", "anthropic"]:
            raise ValueError(f"Invalid LLM provider '{self.llm_provider}'. Must be 'openai' or 'anthropic'")
        
        # Set up persistence directory
        if persist_directory is None:
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY") or str(corpus_manager.get_data_directory() / "chroma_db")
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.text_splitter = None
        self.rag_chain = None
        
        # Initialize the RAG components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all RAG components."""
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            show_progress=True
        )
        
        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Initialize retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Initialize LLM based on provider
        self.llm = self._initialize_llm()
        
        # Initialize RAG chain
        self._create_rag_chain()
    
    def _initialize_llm(self):
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
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering."""
        # Define the prompt template for CDC context
        prompt_template = """You are an expert assistant specializing in CDC (Centers for Disease Control and Prevention) health information. Use the following context from CDC publications to answer the question accurately and comprehensively.

Context from CDC publications:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided CDC context
2. If the context doesn't contain enough information, clearly state this limitation
3. Provide specific citations when possible (mention the collection: PCD, EID, or MMWR)
4. Focus on evidence-based health information
5. If discussing medical topics, remind users to consult healthcare professionals
6. Keep your response clear and accessible to a general audience

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        formatted_docs = []
        for doc in docs:
            # Extract metadata for citation
            metadata = doc.metadata
            collection = metadata.get('collection', 'Unknown')
            title = metadata.get('title', 'Unknown Title')
            
            # Format the document with citation info
            formatted_doc = f"[{collection.upper()}] {title}\n{doc.page_content}\n"
            formatted_docs.append(formatted_doc)
        
        return "\n---\n".join(formatted_docs)
    
    def _show_progress_for_operation(self, operation_name: str, emoji: str, operation_func, *args, **kwargs):
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
        # Get articles from the corpus manager
        articles = self.corpus_manager.load_json_articles_as_iterable(
            collection=collection,
            language=language
        )
        
        # Convert to list to get count for progress bar
        articles_list = list(articles)
        total_articles = len(articles_list)
        
        if total_articles == 0:
            print("No articles found to index.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }
        
        print(f"Creating documents from {total_articles} articles...")
        
        # Create documents from all articles
        documents = []
        processed_count = 0
        
        try:
            from tqdm import tqdm
            articles_iterator = tqdm(articles_list, desc="Processing articles", unit=" articles")
        except ImportError:
            print("Processing articles...")
            articles_iterator = articles_list
        
        for article in articles_iterator:
            # Skip articles without meaningful content
            if not article.title and not article.abstract and not article.full_text:
                continue
            
            # Create document content by combining title, abstract, and full text
            content_parts = []
            
            if article.title:
                content_parts.append(f"Title: {article.title}")
            
            if article.abstract:
                content_parts.append(f"Abstract: {article.abstract}")
            
            if article.full_text:
                content_parts.append(f"Full Text: {article.full_text}")
            
            content = "\n\n".join(content_parts)
            
            # Create metadata
            metadata = {
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
            
            # Create document
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
            
            processed_count += 1
        
        if not documents:
            print("No valid documents found to index.")
            return {
                "articles_processed": 0,
                "total_chunks": 0,
                "collection": collection or "all",
                "language": language,
                "embedding_model": self.embedding_model_name
            }
        
        # Split documents into chunks using RecursiveCharacterTextSplitter
        print("📄 Splitting documents into chunks...")
        chunked_documents = self._show_progress_for_operation(
            "Splitting documents",
            "📄",
            self.text_splitter.split_documents,
            documents
        )
        
        total_chunks = len(chunked_documents)
        print(f"✅ Created {total_chunks} chunks from {len(documents)} documents")
        
        # Create new vector store from documents
        print("🔍 Creating vector database with embeddings...")
        print(f"   Processing {total_chunks} chunks with {self.embedding_model_name}...")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Update retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Recreate RAG chain with new retriever
        self._create_rag_chain()
        
        stats = {
            "articles_processed": processed_count,
            "total_chunks": total_chunks,
            "collection": collection or "all",
            "language": language,
            "embedding_model": self.embedding_model_name
        }
        
        print(f"✅ Indexing complete! Processed {processed_count} articles into {total_chunks} chunks")
        return stats
    
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
        # Update retriever with new k value
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Apply collection filter if specified
        if collection_filter:
            filter_dict = {"collection": collection_filter.lower()}
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k, "filter": filter_dict}
            )
        
        # Perform search
        docs = self.retriever.get_relevant_documents(query)
        
        # Format results
        results = []
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
    
    def ask_question(
        self,
        question: str,
        collection_filter: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question using the RAG system.
        
        Args:
            question: The question to ask
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            include_sources: Whether to include source documents in the response
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Apply collection filter if specified
        if collection_filter:
            filter_dict = {"collection": collection_filter.lower()}
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "filter": filter_dict}
            )
        
        # Get the answer using the RAG chain
        answer = self.rag_chain.invoke(question)
        
        result = {
            "question": question,
            "answer": answer,
            "collection_filter": collection_filter
        }
        
        # Include source documents if requested
        if include_sources:
            source_docs = self.retriever.get_relevant_documents(question)
            sources = []
            
            for doc in source_docs:
                source = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "collection": doc.metadata.get("collection", "unknown"),
                    "url": doc.metadata.get("url", ""),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source)
            
            result["sources"] = sources
        
        return result
    
    def ask_question_stream(
        self,
        question: str,
        collection_filter: Optional[str] = None,
        include_sources: bool = True
    ):
        """
        Ask a question using the RAG system with streaming response.
        
        Args:
            question: The question to ask
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            include_sources: Whether to include source documents in the response
            
        Yields:
            Dictionary with streaming chunks and optional sources
        """
        # Suppress progress bars and verbose output during streaming
        import os
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        # Apply collection filter if specified
        if collection_filter:
            filter_dict = {"collection": collection_filter.lower()}
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5, "filter": filter_dict}
            )
        
        # Get source documents first (suppress any progress output)
        source_docs = []
        sources = []
        
        if include_sources:
            # Suppress output during document retrieval
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                source_docs = self.retriever.get_relevant_documents(question)
            
            for doc in source_docs:
                source = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "collection": doc.metadata.get("collection", "unknown"),
                    "url": doc.metadata.get("url", ""),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source)
        
        # Stream the answer using the RAG chain
        for chunk in self.rag_chain.stream(question):
            yield {
                "question": question,
                "chunk": chunk,
                "collection_filter": collection_filter,
                "sources": sources if include_sources else None
            }
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        try:
            # Get the collection from ChromaDB
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get some sample metadata to understand the data
            sample_results = collection.peek(limit=10)
            collections = set()
            languages = set()
            
            if sample_results and 'metadatas' in sample_results:
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
    
    def clear_vectorstore(self) -> None:
        """Clear all documents from the vector store."""
        try:
            # Delete and recreate the collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            print("Vector store cleared successfully")
        except Exception as e:
            print(f"Error clearing vector store: {e}")
    
    def switch_llm(self, llm_model: str, llm_provider: Optional[str] = None) -> None:
        """
        Switch to a different LLM model and optionally provider.
        
        Args:
            llm_model: New LLM model name
            llm_provider: New LLM provider ('openai' or 'anthropic'). If None, keeps current provider
        """
        if llm_provider is not None:
            if llm_provider.lower() not in ["openai", "anthropic"]:
                raise ValueError(f"Invalid LLM provider '{llm_provider}'. Must be 'openai' or 'anthropic'")
            self.llm_provider = llm_provider.lower()
        
        self.llm_model_name = llm_model
        self.llm = self._initialize_llm()
        
        # Recreate the RAG chain with the new LLM
        self._create_rag_chain()
        
        print(f"Switched to {self.llm_provider} model: {self.llm_model_name}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get a dictionary of available models by provider.
        
        Returns:
            Dictionary with provider as key and list of model names as values
        """
        return {
            "openai": [
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ],
            "anthropic": [
                "claude-3-5-sonnet",
                "claude-3-5-haiku",
                "claude-3-sonnet",
                "claude-3-haiku",
                "claude-3-opus"
            ]
        }
    
    def test_llm_connection(self, test_message: str = "Hello! Please respond with 'Connection successful.'") -> Dict[str, Any]:
        """
        Test the connection to the current LLM.
        
        Args:
            test_message: Simple message to send to the LLM for testing
            
        Returns:
            Dictionary with test results including success status, response, and timing
        """
        import time
        
        test_result = {
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
    
    @staticmethod
    def test_all_providers(test_message: str = "Hello! Please respond with 'Connection successful.'") -> Dict[str, Dict[str, Any]]:
        """
        Test connections to all available LLM providers.
        
        Args:
            test_message: Simple message to send to LLMs for testing
            
        Returns:
            Dictionary with test results for each provider
        """
        results = {}
        
        # Test OpenAI
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key.strip():
                openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
                
                import time
                start_time = time.time()
                response = openai_llm.invoke(test_message)
                end_time = time.time()
                
                results["openai"] = {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "success": True,
                    "response": response.content if hasattr(response, 'content') else str(response),
                    "response_time": round(end_time - start_time, 2),
                    "api_key_configured": True,
                    "error": None
                }
            else:
                results["openai"] = {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "success": False,
                    "response": None,
                    "response_time": None,
                    "api_key_configured": False,
                    "error": "OPENAI_API_KEY not configured"
                }
        except Exception as e:
            results["openai"] = {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "success": False,
                "response": None,
                "response_time": None,
                "api_key_configured": bool(openai_key and openai_key.strip()) if 'openai_key' in locals() else False,
                "error": str(e)
            }
        
        # Test Anthropic
        try:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key and anthropic_key.strip():
                anthropic_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
                
                import time
                start_time = time.time()
                response = anthropic_llm.invoke(test_message)
                end_time = time.time()
                
                results["anthropic"] = {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "success": True,
                    "response": response.content if hasattr(response, 'content') else str(response),
                    "response_time": round(end_time - start_time, 2),
                    "api_key_configured": True,
                    "error": None
                }
            else:
                results["anthropic"] = {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "success": False,
                    "response": None,
                    "response_time": None,
                    "api_key_configured": False,
                    "error": "ANTHROPIC_API_KEY not configured"
                }
        except Exception as e:
            results["anthropic"] = {
                "provider": "anthropic",
                "model": "claude-3-5-sonnet",
                "success": False,
                "response": None,
                "response_time": None,
                "api_key_configured": bool(anthropic_key and anthropic_key.strip()) if 'anthropic_key' in locals() else False,
                "error": str(e)
            }
        
        return results
    
    def __repr__(self) -> str:
        """String representation of the RAG engine."""
        return f"RAGEngine(embedding_model='{self.embedding_model_name}', llm_model='{self.llm_model_name}', llm_provider='{self.llm_provider}')"
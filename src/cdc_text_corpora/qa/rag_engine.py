"""Refactored RAG Engine for CDC Text Corpora using composition pattern."""

from typing import List, Dict, Any, Optional, Iterator
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
        persist_directory: Optional[str] = None
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
        """
        self.corpus_manager = corpus_manager
        
        # Load configuration from environment variables with fallbacks
        self.embedding_model_name = embedding_model or os.getenv("DEFAULT_EMBEDDING_MODEL") or "all-MiniLM-L6-v2"
        self.llm_model_name = llm_model or os.getenv("DEFAULT_LLM_MODEL") or "gpt-4o-mini"
        provider = llm_provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.llm_provider = provider.lower() if provider else "openai"
        self.chunk_size = chunk_size or int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
        
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
        self.rag_chain: Any = None
        
        # Initialize LLM and RAG chain
        self._initialize_llm_components()
    
    def _initialize_llm_components(self) -> None:
        """Initialize LLM and RAG chain components."""
        # Initialize LLM based on provider
        self.llm = self._initialize_llm()
        
        # Create RAG chain
        self._create_rag_chain()
    
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
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain for question answering."""
        # Define the prompt template for CDC context
        prompt_template = """You are an expert assistant specializing in CDC (Centers for Disease Control and Prevention) health information. Use the following context from CDC publications to answer the question accurately and comprehensively.

Context from CDC publications:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided CDC context
2. If the context doesn't contain enough information, clearly state this limitation
3. When referencing information from the context, use numbered citations like [1], [2], [3] etc. that correspond to the sources provided
4. Focus on evidence-based health information
5. If discussing medical topics, remind users to consult healthcare professionals
6. Keep your response clear and accessible to a general audience
7. Do NOT include a references section at the end - citations will be added automatically

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Get retriever from vector store
        retriever = self.vector_store.get_retriever()
        
        # Create the RAG chain
        self.rag_chain = (
            {"context": retriever | self.article_processor.format_docs_for_context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
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
        
        # Recreate RAG chain with updated retriever
        self._create_rag_chain()
        
        # Update stats with collection info
        index_stats.update({
            "collection": collection or "all",
            "language": language,
            "articles_processed": len(articles)
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
    
    def ask_question(
        self,
        question: str,
        collection_filter: Optional[str] = None,
        include_sources: bool = True,
        format_citations: bool = False
    ) -> Dict[str, Any]:
        """
        Ask a question using the RAG system.
        
        Args:
            question: The question to ask
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            include_sources: Whether to include source documents in the response
            format_citations: Whether to append formatted citations to the answer text
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Update retriever with collection filter if specified
        if collection_filter:
            retriever = self.vector_store.get_retriever(
                search_type="similarity",
                k=5,
                collection_filter=collection_filter
            )
            
            # Update RAG chain with filtered retriever
            prompt_template = """You are an expert assistant specializing in CDC (Centers for Disease Control and Prevention) health information. Use the following context from CDC publications to answer the question accurately and comprehensively.

Context from CDC publications:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided CDC context
2. If the context doesn't contain enough information, clearly state this limitation
3. When referencing information from the context, use numbered citations like [1], [2], [3] etc. that correspond to the sources provided
4. Focus on evidence-based health information
5. If discussing medical topics, remind users to consult healthcare professionals
6. Keep your response clear and accessible to a general audience
7. Do NOT include a references section at the end - citations will be added automatically

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create temporary RAG chain with filtered retriever
            rag_chain = (
                {"context": retriever | self.article_processor.format_docs_for_context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(question)
        else:
            # Use default RAG chain
            answer = self.rag_chain.invoke(question)
        
        result = {
            "question": question,
            "answer": answer,
            "collection_filter": collection_filter
        }
        
        # Include source documents if requested
        if include_sources:
            # Get source documents using semantic search
            search_results = self.semantic_search(
                query=question,
                k=5,
                collection_filter=collection_filter
            )
            
            sources = []
            for search_result in search_results:
                source = {
                    "title": search_result["title"],
                    "collection": search_result["collection"],
                    "url": search_result["url"]
                }
                sources.append(source)
            
            result["sources"] = sources
            
            # Optionally append formatted citations to the answer
            if format_citations and sources:
                formatted_citations = self.format_sources_as_citations(sources)
                result["answer"] = result["answer"] + formatted_citations
        
        return result
    
    def ask_question_stream(
        self,
        question: str,
        collection_filter: Optional[str] = None,
        include_sources: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Ask a question using the RAG system with streaming response.
        
        Args:
            question: The question to ask
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            include_sources: Whether to include source documents in the response
            
        Yields:
            Dictionary with streaming chunks and optional sources
        """
        # Get sources first if requested
        sources = []
        if include_sources:
            search_results = self.semantic_search(
                query=question,
                k=5,
                collection_filter=collection_filter
            )
            
            for search_result in search_results:
                source = {
                    "title": search_result["title"],
                    "collection": search_result["collection"],
                    "url": search_result["url"]
                }
                sources.append(source)
        
        # Determine which RAG chain to use
        if collection_filter:
            retriever = self.vector_store.get_retriever(
                search_type="similarity",
                k=5,
                collection_filter=collection_filter
            )
            
            prompt_template = """You are an expert assistant specializing in CDC (Centers for Disease Control and Prevention) health information. Use the following context from CDC publications to answer the question accurately and comprehensively.

Context from CDC publications:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided CDC context
2. If the context doesn't contain enough information, clearly state this limitation
3. When referencing information from the context, use numbered citations like [1], [2], [3] etc. that correspond to the sources provided
4. Focus on evidence-based health information
5. If discussing medical topics, remind users to consult healthcare professionals
6. Keep your response clear and accessible to a general audience
7. Do NOT include a references section at the end - citations will be added automatically

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            rag_chain = (
                {"context": retriever | self.article_processor.format_docs_for_context, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            rag_chain = self.rag_chain
        
        # Stream the answer
        for chunk in rag_chain.stream(question):
            yield {
                "question": question,
                "chunk": chunk,
                "collection_filter": collection_filter,
                "sources": sources if include_sources else None
            }
    
    def format_sources_as_citations(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources as properly formatted citations text.
        
        Args:
            sources: List of source dictionaries with title, collection, url, excerpt
            
        Returns:
            Formatted citation text that can be appended to answers
        """
        if not sources:
            return ""
        
        citations = []
        citations.append("\n\n## References")
        
        for i, source in enumerate(sources, 1):
            collection = source.get('collection', 'unknown').upper()
            title = source.get('title', 'Unknown Title')
            url = source.get('url', '')
            
            # Format citation with available information
            citation_parts = [f"[{i}]"]
            
            if title != 'Unknown Title':
                citation_parts.append(f'"{title}"')
            
            citation_parts.append(f"CDC {collection} Collection")
            
            if url:
                citation_parts.append(f"Available at: {url}")
            
            citations.append(" ".join(citation_parts))
        
        return "\n".join(citations)
    
    # Delegate vector store operations
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return self.vector_store.get_stats()
    
    def clear_vectorstore(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear()
        # Recreate RAG chain after clearing
        self._create_rag_chain()
    
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
    
    def __repr__(self) -> str:
        """String representation of the RAG engine."""
        return f"RAGEngine(embedding_model='{self.embedding_model_name}', llm_model='{self.llm_model_name}', llm_provider='{self.llm_provider}')"
"""RAG Pipeline for CDC Text Corpora - End-to-end workflow management."""

from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.qa.rag_engine import RAGEngine


class RAGPipeline:
    """
    End-to-end RAG pipeline for CDC Text Corpora.
    
    This class orchestrates the complete workflow:
    1. Load and verify corpus data
    2. Index articles into vector database (if not already indexed)
    3. Interactive question/answer loop
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        collection_filter: Optional[str] = None,
        language: str = 'en'
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Optional custom data directory
            collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
            language: Language filter ('en', 'es', 'fr', 'zhs', 'zht')
        """
        self.console = Console()
        self.data_dir = data_dir
        self.collection_filter = collection_filter
        self.language = language
        
        # Initialize components
        self.corpus = None
        self.rag_engine = None
        self.is_indexed = False
        
        # Initialize the pipeline
        self._initialize()
    
    def _initialize(self):
        """Initialize the corpus and RAG engine."""
        try:
            # Initialize corpus manager
            self.corpus = CDCCorpus(data_dir=self.data_dir)
            
            # Initialize RAG engine
            self.rag_engine = RAGEngine(self.corpus)
            
            self.console.print("[green]✅ RAG Pipeline initialized successfully[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to initialize pipeline: {e}[/red]")
            raise
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check if required data is available for the pipeline.
        
        Returns:
            Dictionary with data availability status
        """
        status = {
            "parsed_articles_available": False,
            "collections_found": [],
            "total_articles": 0,
            "vector_index_exists": False,
            "recommendations": []
        }
        
        # Check for parsed JSON files
        json_parsed_dir = self.corpus.get_data_directory() / "json-parsed"
        
        if json_parsed_dir.exists():
            # Look for JSON files matching our criteria
            collections_to_check = [self.collection_filter] if self.collection_filter else ['pcd', 'eid', 'mmwr']
            
            for collection in collections_to_check:
                if self.language and self.language != "all":
                    pattern = f"{collection}_{self.language}_*.json"
                else:
                    pattern = f"{collection}_*_*.json"
                
                json_files = list(json_parsed_dir.glob(pattern))
                if json_files:
                    status["collections_found"].append(collection)
            
            if status["collections_found"]:
                status["parsed_articles_available"] = True
                
                # Count total articles
                articles = self.corpus.load_json_articles_as_iterable(
                    collection=self.collection_filter,
                    language=self.language
                )
                status["total_articles"] = len(articles)
        
        # Check for vector index
        try:
            vector_stats = self.rag_engine.get_vectorstore_stats()
            if "total_documents" in vector_stats and vector_stats["total_documents"] > 0:
                status["vector_index_exists"] = True
        except Exception:
            status["vector_index_exists"] = False
        
        # Generate recommendations
        if not status["parsed_articles_available"]:
            status["recommendations"].append("Parse some collections first using: cdc-corpus parse --collection <name>")
        
        if not status["vector_index_exists"] and status["parsed_articles_available"]:
            status["recommendations"].append("Articles will be indexed automatically when you start the Q&A session")
        
        return status
    
    def display_pipeline_status(self):
        """Display the current pipeline status."""
        status = self.check_data_availability()
        
        self.console.print(Panel(
            "[bold blue]CDC Text Corpora RAG Pipeline[/bold blue]",
            title="🤖 RAG System",
            border_style="blue"
        ))
        
        # Create status table
        table = Table(title="Pipeline Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # LLM Connection
        llm_status = "✅ Ready" if self.rag_engine else "❌ Not initialized"
        llm_details = f"{self.rag_engine.llm_provider} - {self.rag_engine.llm_model_name}" if self.rag_engine else ""
        table.add_row("LLM Connection", llm_status, llm_details)
        
        # Parsed Articles
        articles_status = "✅ Available" if status["parsed_articles_available"] else "❌ Missing"
        articles_details = f"{len(status['collections_found'])} collections, {status['total_articles']} articles"
        if status["collections_found"]:
            articles_details += f" ({', '.join(status['collections_found']).upper()})"
        table.add_row("Parsed Articles", articles_status, articles_details)
        
        # Vector Index
        index_status = "✅ Ready" if status["vector_index_exists"] else "⏳ Will be created"
        index_details = "Existing index found" if status["vector_index_exists"] else "Auto-indexing on first use"
        table.add_row("Vector Index", index_status, index_details)
        
        # Language/Collection Filter
        filter_details = []
        if self.collection_filter:
            filter_details.append(f"Collection: {self.collection_filter.upper()}")
        else:
            filter_details.append("Collection: All")
        
        if self.language and self.language != "all":
            filter_details.append(f"Language: {self.language.upper()}")
        else:
            filter_details.append("Language: All")
        
        filter_text = ", ".join(filter_details)
        table.add_row("Filters", "ℹ️  Active", filter_text)
        
        self.console.print(table)
        
        # Show recommendations if any
        if status["recommendations"]:
            self.console.print("\n[yellow]📋 Recommendations:[/yellow]")
            for rec in status["recommendations"]:
                self.console.print(f"  • {rec}")
        
        return status
    
    def ensure_vector_index(self) -> bool:
        """
        Ensure that articles are indexed in the vector database.
        
        Returns:
            True if indexing is successful or already exists, False otherwise
        """
        # Check if index already exists
        try:
            stats = self.rag_engine.get_vectorstore_stats()
            if "total_documents" in stats and stats["total_documents"] > 0:
                self.console.print(f"[green]✅ Vector index already exists ({stats['total_documents']} documents)[/green]")
                self.is_indexed = True
                return True
        except Exception:
            pass
        
        # Check if we have parsed articles to index
        status = self.check_data_availability()
        if not status["parsed_articles_available"]:
            self.console.print("[red]❌ No parsed articles found. Please parse collections first.[/red]")
            return False
        
        # Ask user if they want to index
        self.console.print(f"\n[yellow]📚 Found {status['total_articles']} articles to index[/yellow]")
        should_index = Confirm.ask("Would you like to index these articles for semantic search?", default=True)
        
        if not should_index:
            self.console.print("[yellow]⚠️  Skipping indexing. You can only ask questions about specific documents.[/yellow]")
            return False
        
        # Perform indexing
        try:
            index_stats = self.rag_engine.index_articles(
                collection=self.collection_filter,
                language=self.language
            )
            
            self.console.print(f"[green]Successfully indexed {index_stats['articles_processed']} articles into {index_stats['total_chunks']} chunks[/green]")
            self.is_indexed = True
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Indexing failed: {e}[/red]")
            return False
    
    def interactive_qa_loop(self):
        """Run the interactive question/answer loop."""
        self.console.print(Panel(
            "[bold green]🎯 Interactive Q&A Session[/bold green]\n"
            "[dim]Ask questions about CDC publications. Type 'quit', 'exit', or 'q' to stop.[/dim]",
            title="Q&A Mode",
            border_style="green"
        ))
        
        question_count = 0
        
        while True:
            try:
                # Get question from user
                question = Prompt.ask(f"\n[bold cyan]Question #{question_count + 1}[/bold cyan]")
                
                # Check for exit commands
                if question.lower().strip() in ['quit', 'exit', 'q', '']:
                    break
                
                # Process the question with streaming
                self.console.print("\n[bold blue]🤖 Answer:[/bold blue]")
                
                # Stream the response
                full_answer = ""
                sources = None
                
                for chunk_data in self.rag_engine.ask_question_stream(
                    question=question,
                    collection_filter=self.collection_filter,
                    include_sources=True
                ):
                    chunk = chunk_data.get("chunk", "")
                    if chunk:
                        self.console.print(chunk, end="", style="white")
                        full_answer += chunk
                    
                    # Store sources from the first chunk
                    if sources is None and chunk_data.get("sources"):
                        sources = chunk_data["sources"]
                
                # Add a newline after streaming is complete
                self.console.print()
                
                # Display sources if available
                if sources:
                    self.console.print("\n[bold yellow]📚 Sources:[/bold yellow]")
                    for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                        collection_badge = f"[{source['collection'].upper()}]"
                        self.console.print(f"  {i}. {collection_badge} {source['title']}")
                        if source.get('url'):
                            self.console.print(f"     [dim]{source['url']}[/dim]")
                        self.console.print(f"     [dim]...{source['excerpt']}...[/dim]")
                
                question_count += 1
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error processing question: {e}[/red]")
                continue
        
        self.console.print(f"\n[green]👋 Q&A session ended. Answered {question_count} questions.[/green]")
    
    def run(self):
        """Run the complete RAG pipeline."""
        try:
            # Display initial status
            status = self.display_pipeline_status()
            
            # Check if we can proceed
            if not status["parsed_articles_available"]:
                self.console.print("\n[red]❌ Cannot start Q&A session without parsed articles.[/red]")
                self.console.print("[yellow]Please run: cdc-corpus parse --collection <name> first[/yellow]")
                return
            
            # Test LLM connection
            self.console.print("\n[yellow]🔄 Testing LLM connection...[/yellow]")
            test_result = self.rag_engine.test_llm_connection()
            
            if not test_result['success']:
                self.console.print(f"[red]❌ LLM connection failed: {test_result['error']}[/red]")
                return
            
            self.console.print("[green]✅ LLM connection successful[/green]")
            
            # Ensure vector index exists
            if not self.ensure_vector_index():
                # User chose not to index or indexing failed
                # We can still proceed but with limited functionality
                pass
            
            # Start interactive Q&A
            self.interactive_qa_loop()
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]👋 Pipeline interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Pipeline error: {e}[/red]")
            raise

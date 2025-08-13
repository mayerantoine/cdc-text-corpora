"""Main CLI entry point for cdc-text-corpora."""

import typer
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import pathlib
from cdc_text_corpora.core.downloader import download_collection
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.utils.config import get_data_directory
from cdc_text_corpora.qa.rag_engine import RAGEngine

###TODO

## FIX RAG
# Fix progress for vector database operations using tqdm
# Stop agent early if there's no data
# Improve speed of vector operations
# Agentic RAG should work end-to-end

## FIX FILES SAVES and ADD pandas API
# add parse - parquet # Save files as Parquet during parsing
# Parquet vs hybrid database - vector db embed chunks VS full text db title+abstract
# Lazy loading API for parquet files


## Add citations to RAG Agent and RAP Pipeline
app = typer.Typer(
    name="cdc-corpus",
    help="CDC Text Corpora: Interactive access to PCD, EID, and MMWR collections. Run without arguments for guided setup.",
)

console = Console()



@app.command()
def download(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c",
        help="Collection to download: pcd, eid, mmwr, metadata, or all",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """Download CDC text corpora collections."""
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "metadata", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Display collection info
    collection_info = {
        "pcd": "Preventing Chronic Disease",
        "eid": "Emerging Infectious Diseases", 
        "mmwr": "Morbidity and Mortality Weekly Report",
        "metadata": "Collection metadata only",
        "all": "All collections and metadata"
    }
    
    collection_lower = collection.lower()
    
    console.print(Panel(
        f"[bold blue]Downloading: {collection_info[collection_lower]}[/bold blue]",
        title="CDC Text Corpora Downloader",
        border_style="blue"
    ))
    
    try:
        # Call the existing download function - it has its own rich progress bars
        download_collection(collection_lower)
        
        console.print(f"[green]✓ Successfully downloaded {collection_lower} collection[/green]")
        
        if verbose:
            # Get the actual data directory path
            data_path = get_data_directory()
            console.print(f"[dim]Files saved to: {data_path}[/dim]")
            
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)



@app.command()
def parse(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c",
        help="Collection to parse: pcd, eid, mmwr, or all",
    ),
    language: str = typer.Option(
        "all",
        "--language",
        "-l",
        help="Language to parse: en, es, fr, zhs, zht, or all",
    ),
    save_json: bool = typer.Option(
        True,
        "--save-json",
        "-j",
        help="Save parsed articles as JSON files",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Custom output directory for JSON files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output including validation details",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Enable/disable article validation during parsing",
    ),
) -> None:
    """Parse CDC text corpora collections into structured data."""
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate language input
    valid_languages = ["en", "es", "fr", "zhs", "zht", "all"]
    if language.lower() not in valid_languages:
        console.print(f"[red]Error: Invalid language '{language}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_languages)}[/yellow]")
        raise typer.Exit(1)
    
    # Display parsing info
    collection_info = {
        "pcd": "Preventing Chronic Disease",
        "eid": "Emerging Infectious Diseases", 
        "mmwr": "Morbidity and Mortality Weekly Report",
        "all": "All collections"
    }
    
    language_info = {
        "en": "English",
        "es": "Spanish",
        "fr": "French", 
        "zhs": "Simplified Chinese",
        "zht": "Traditional Chinese",
        "all": "All languages"
    }
    
    collection_lower = collection.lower()
    language_lower = language.lower() if language.lower() != "all" else None
    
    console.print(Panel(
        f"[bold blue]Parsing: {collection_info[collection_lower]} ({language_info[language.lower()]})[/bold blue]\n"
        f"[dim]Save JSON: {'Yes' if save_json else 'No'}[/dim]",
        title="CDC Text Corpora Parser",
        border_style="blue"
    ))
    
    try:
        # Initialize corpus manager
        corpus = CDCCorpus()
        
        # Determine collections to process
        collections_to_process = ['pcd', 'eid', 'mmwr'] if collection_lower == 'all' else [collection_lower]
        
        total_parsed = 0
        json_files_created = []
        
        for coll in collections_to_process:
            console.print(f"\n[bold cyan]Processing {coll.upper()} collection...[/bold cyan]")
            
            # Check if collection is downloaded
            if not corpus.is_collection_downloaded(coll):
                console.print(f"[yellow]Warning: {coll.upper()} collection not downloaded. Skipping...[/yellow]")
                console.print(f"[dim]Download with: cdc-corpus download -c {coll}[/dim]")
                continue
            
            # Parse the collection with validation
            result = corpus.load_parse_save_html_articles(
                collection=coll,
                language=language_lower,
                save_json=save_json,
                output_dir=output_dir,
                validate_articles=validate
            )
            
            # Check for errors
            if 'error' in result['stats']:
                console.print(f"[red]Error: {result['stats']['error']}[/red]")
                continue
            
            # Display results
            parsed_count = result['stats']['parsed_count']
            total_parsed += parsed_count
            
            console.print(f"[green]✓ Parsed {parsed_count} articles from {coll.upper()}[/green]")
            
            if save_json and 'json_file_path' in result:
                json_files_created.append(result['json_file_path'])
                console.print(f"[green]✓ Saved to: {result['json_file_path']}[/green]")
            
            if verbose:
                console.print(f"[dim]  HTML files loaded: {result['stats']['html_count']}[/dim]")
                console.print(f"[dim]  Language: {result['stats']['language']}[/dim]")
        
        # Final summary
        console.print(f"\n[bold green]✅ Parsing complete![/bold green]")
        console.print(f"[green]Total articles parsed: {total_parsed}[/green]")
        
        if json_files_created:
            console.print(f"[green]JSON files created: {len(json_files_created)}[/green]")
            if verbose:
                for json_file in json_files_created:
                    console.print(f"[dim]  {json_file}[/dim]")
        
        if verbose:
            data_path = get_data_directory()
            console.print(f"[dim]Data directory: {data_path}[/dim]")
            
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def index(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c",
        help="Collection to index: pcd, eid, mmwr, or all",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        "-l",
        help="Language filter: en, es, fr, zhs, zht, or all",
    ),
    source_type: str = typer.Option(
        "json",
        "--source-type",
        "-s",
        help="Source type: json (parsed articles) or html (raw HTML files)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output including detailed statistics",
    ),
) -> None:
    """Create vector index for semantic search and RAG operations."""
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate language input
    valid_languages = ["en", "es", "fr", "zhs", "zht", "all"]
    if language.lower() not in valid_languages:
        console.print(f"[red]Error: Invalid language '{language}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_languages)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate source_type input
    valid_source_types = ["json", "html"]
    if source_type.lower() not in valid_source_types:
        console.print(f"[red]Error: Invalid source type '{source_type}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_source_types)}[/yellow]")
        raise typer.Exit(1)
    
    # Display indexing configuration
    collection_info = {
        "pcd": "Preventing Chronic Disease",
        "eid": "Emerging Infectious Diseases", 
        "mmwr": "Morbidity and Mortality Weekly Report",
        "all": "All collections"
    }
    
    language_info = {
        "en": "English",
        "es": "Spanish",
        "fr": "French", 
        "zhs": "Simplified Chinese",
        "zht": "Traditional Chinese",
        "all": "All languages"
    }
    
    source_info = {
        "json": "Parsed JSON articles (structured metadata)",
        "html": "Raw HTML files (structure-aware chunking)"
    }
    
    collection_lower = collection.lower()
    language_lower = language.lower() if language.lower() != "all" else None
    source_type_lower = source_type.lower()
    
    console.print(Panel(
        f"[bold blue]Indexing: {collection_info[collection_lower]} ({language_info[language.lower()]})[/bold blue]\n"
        f"[dim]Source Type: {source_info[source_type_lower]}[/dim]\n"
        f"[dim]Embedding Model: sentence-transformers/all-MiniLM-L6-v2[/dim]",
        title="CDC Text Corpora Vector Indexer",
        border_style="blue"
    ))
    
    try:
        # Initialize corpus and RAG engine
        console.print("\n[yellow]🔧 Initializing RAG engine...[/yellow]")
        corpus = CDCCorpus()
        rag_engine = RAGEngine(corpus)
        
        # Prepare collection parameter for indexing
        collection_param = (
            collection_lower
            if collection_lower != 'all'
            else None
        )
        
        # Prepare language parameter for indexing
        language_param = language_lower if language_lower else 'en'
        
        # Display pre-indexing information
        console.print(f"[cyan]📊 Starting vector indexing...[/cyan]")
        if verbose:
            console.print(f"[dim]  Collection: {collection_param or 'all'}[/dim]")
            console.print(f"[dim]  Language: {language_param}[/dim]")
            console.print(f"[dim]  Source Type: {source_type_lower}[/dim]")
        
        # Perform indexing (progress bars are handled by create_vector_index)
        index_result = rag_engine.create_vector_index(
            collection=collection_param,
            language=language_param,
            source_type=source_type_lower
        )
        
        # Handle results
        if index_result["success"]:
            stats = index_result["stats"]
            
            if stats.get("already_exists", False):
                doc_count = stats.get("total_documents", 0)
                console.print(f"[green]✅ Vector index already exists ({doc_count} documents)[/green]")
            else:
                # New index created
                articles_processed = stats.get("articles_processed", 0)
                total_chunks = stats.get("total_chunks", 0)
                embedding_model = stats.get("embedding_model", "unknown")
                
                console.print(f"[green]✅ Successfully created vector index![/green]")
                console.print(f"[cyan]📈 Statistics:[/cyan]")
                console.print(f"  • Articles processed: {articles_processed}")
                console.print(f"  • Total chunks indexed: {total_chunks}")
                console.print(f"  • Embedding model: {embedding_model}")
                
                if verbose:
                    console.print(f"[dim]  • Collection: {stats.get('collection', 'unknown')}[/dim]")
                    console.print(f"  • Language: {stats.get('language', 'unknown')}")
                    console.print(f"  • Source type: {index_result.get('source_type', 'unknown')}")
            
            # Provide next steps
            console.print(f"\n[yellow]🚀 Next Steps:[/yellow]")
            console.print(f"  • Use [cyan]cdc-corpus qa[/cyan] for interactive question answering")
            console.print(f"  • Use the semantic search API for custom queries")
            
        else:
            # Indexing failed
            error_msg = index_result.get("error", "Unknown error")
            console.print(f"[red]❌ Indexing failed: {error_msg}[/red]")
            
            # Provide helpful suggestions based on error type
            if "No parsed JSON articles" in error_msg:
                console.print(f"[yellow]💡 Suggestion: Run parsing first with:[/yellow]")
                console.print(f"  [cyan]cdc-corpus parse -c {collection_lower}[/cyan]")
            elif "No HTML articles" in error_msg:
                console.print(f"[yellow]💡 Suggestion: Download collection first with:[/yellow]")
                console.print(f"  [cyan]cdc-corpus download -c {collection_lower}[/cyan]")
            
            raise typer.Exit(1)
            
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during indexing: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def qa(
    collection: str = typer.Option(
        "all",
        "--collection", 
        "-c",
        help="Collection to query: pcd, eid, mmwr, or all"
    ),
    language: str = typer.Option(
        "en", 
        "--language",
        "-l", 
        help="Language filter: en, es, fr, zhs, zht, or all"
    ),
    mode: str = typer.Option(
        "agentic",
        "--mode",
        "-m",
        help="RAG mode: agentic (multi-agent research system)"
    ),
    data_dir: str = typer.Option(
        None,
        "--data-dir", 
        "-d",
        help="Custom data directory path"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output"
    )
) -> None:
    """Interactive RAG-based question answering for CDC collections.
    
    Uses advanced RAG system for comprehensive, evidence-based answers.
    """
    
    # Validate collection input
    valid_collections = ["pcd", "eid", "mmwr", "all"]
    if collection.lower() not in valid_collections:
        console.print(f"[red]Error: Invalid collection '{collection}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_collections)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate language input
    valid_languages = ["en", "es", "fr", "zhs", "zht", "all"]
    if language.lower() not in valid_languages:
        console.print(f"[red]Error: Invalid language '{language}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_languages)}[/yellow]")
        raise typer.Exit(1)
    
    # Validate mode input
    valid_modes = ["agentic"]
    if mode.lower() not in valid_modes:
        console.print(f"[red]Error: Invalid mode '{mode}'[/red]")
        console.print(f"[yellow]Valid options: {', '.join(valid_modes)}[/yellow]")
        raise typer.Exit(1)
    
    # Convert "all" to None for pipeline
    collection_filter = None if collection.lower() == "all" else collection.lower()
    language_param = None if language.lower() == "all" else language.lower()
    
    try:
        # Display command info
        if verbose:
            console.print(Panel(
                f"[bold blue]Starting RAG Q&A Session[/bold blue]\n"
                f"Mode: {mode}\n"
                f"Collection: {collection}\n"
                f"Language: {language}\n"
                f"Data directory: {data_dir or 'default'}",
                title="Configuration",
                border_style="blue"
            ))
        
        # Initialize corpus and start RAG pipeline
        corpus = CDCCorpus(data_dir=data_dir)
        
        # Start RAG pipeline using simplified approach
        run_rag_pipeline(corpus, collection_filter, console)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Q&A session interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)



@app.command()
def search() -> None:
    """Search CDC text corpora collections."""
    print(f"RAG code")


def display_welcome_intro():
    """Display welcome message and tool introduction."""
    console.print(Panel(
        "[bold cyan]CDC Text Corpora[/bold cyan]\n\n"
        "[blue]🏥 Access to CDC Collections:[/blue]\n"
        "  • [green]PCD[/green] - Preventing Chronic Disease (2004-2023)\n"
        "  • [green]EID[/green] - Emerging Infectious Diseases (1995-2023)\n"
        "  • [green]MMWR[/green] - Morbidity and Mortality Weekly Report (1982-2023)\n\n"
        "[blue]🤖 Powered by AI:[/blue]\n"
        "  • Semantic search across 40+ years of research\n"
        "  • Multi-agent RAG system for comprehensive answers\n"
        "  • Citation-based responses with source links",
        title="🔬 Welcome to CDC Text Corpora",
        border_style="cyan"
    ))


def interactive_collection_selection():
    """Interactive menu for collection selection."""
    console.print("\n[bold blue]Step 1: Select Collections to Download[/bold blue]")
    
    collections_info = {
        "1": {"code": "pcd", "name": "Preventing Chronic Disease", "desc": "Chronic disease research (2004-2023)"},
        "2": {"code": "eid", "name": "Emerging Infectious Diseases", "desc": "Infectious disease research (1995-2023)"}, 
        "3": {"code": "mmwr", "name": "Morbidity and Mortality Weekly Report", "desc": "Weekly surveillance reports (1982-2023)"},
        "4": {"code": "all", "name": "All Collections", "desc": "Download all three collections"}
    }
    
    # Display collection options
    table = Table(title="Available Collections", box=box.ROUNDED)
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Collection", style="green")
    table.add_column("Description", style="dim")
    
    for key, info in collections_info.items():
        table.add_row(key, info["name"], info["desc"])
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\n[bold]Select collection(s) to download",
            choices=["1", "2", "3", "4"],
            default="4"
        )
        
        selected = collections_info[choice]
        
        # Confirm selection
        if Confirm.ask(f"\n[yellow]Download {selected['name']}?[/yellow]", default=True):
            return selected["code"]
        else:
            console.print("[dim]Please make another selection...[/dim]")


def interactive_language_selection(available_languages):
    """Interactive menu for language selection."""
    console.print("\n[bold blue]Step 3: Select Language for Parsing & Indexing[/bold blue]")
    
    language_info = {
        "1": {"code": "en", "name": "English", "desc": "Primary language for all collections"},
        "2": {"code": "es", "name": "Spanish", "desc": "Available for MMWR and some PCD articles"},
        "3": {"code": "fr", "name": "French", "desc": "Available for some MMWR articles"},
        "4": {"code": "all", "name": "All Languages", "desc": "Process all available languages"}
    }
    
    # Filter available options based on what's actually available
    if "es" not in available_languages:
        del language_info["2"]
    if "fr" not in available_languages:
        del language_info["3"]
    
    # Display language options
    table = Table(title="Available Languages", box=box.ROUNDED)
    table.add_column("Option", style="cyan", no_wrap=True) 
    table.add_column("Language", style="green")
    table.add_column("Description", style="dim")
    
    for key, info in language_info.items():
        table.add_row(key, info["name"], info["desc"])
    
    console.print(table)
    
    while True:
        choice = Prompt.ask(
            "\n[bold]Select language to process",
            choices=list(language_info.keys()),
            default="1"
        )
        
        selected = language_info[choice]
        
        # Confirm selection
        if Confirm.ask(f"\n[yellow]Process {selected['name']}?[/yellow]", default=True):
            return selected["code"]
        else:
            console.print("[dim]Please make another selection...[/dim]")


def run_interactive_qa_loop(rag_engine: RAGEngine, collection_filter: Optional[str], console: Console) -> None:
    """Run interactive loop for Q&A using RAGEngine.generate_answer."""
    console.print(Panel(
        "[bold green]🎯 Interactive Q&A Session[/bold green]\n"
        "[dim]Ask research questions. The system will search, gather evidence, and provide comprehensive answers.\n"
        "Type 'quit', 'exit', or 'q' to stop.[/dim]",
        title="RAG Q&A Mode", 
        border_style="green"
    ))
    
    question_count = 0
    
    while True:
        try:
            # Get question from user
            question = Prompt.ask(f"\n[bold cyan]Research Question #{question_count + 1}[/bold cyan]")
            
            # Check for exit commands
            if question.lower().strip() in ['quit', 'exit', 'q', '']:
                break
            
            # Show processing indicator
            console.print("[yellow]🔄 Processing with RAG system...[/yellow]")
            
            # Use the simple generate_answer method
            try:
                answer = rag_engine.generate_answer(
                    question=question,
                    collection_filter=collection_filter,
                    max_turns=10
                )
                
                # Display answer
                console.print(Panel(
                    f"[cyan]{answer}[/cyan]",
                    title="🤖 RAG Answer",
                    border_style="magenta"
                ))
                
                question_count += 1
                
            except Exception as e:
                console.print(f"[red]❌ Error processing question: {e}[/red]")
                continue
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]❌ Error in interactive loop: {e}[/red]")
            continue
    
    console.print(f"\n[green]👋 Q&A session ended. Answered {question_count} questions.[/green]")


def display_pipeline_status_cli(
    rag_engine: RAGEngine, 
    console: Console, 
    collection_filter: Optional[str] = None,
    status: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Display the current RAG pipeline status using CLI formatting.
    
    Args:
        rag_engine: RAGEngine instance
        console: Console for output
        collection_filter: Optional collection filter
        status: Optional pre-computed status data to avoid redundant checks
        
    Returns:
        Status dictionary with pipeline state
    """
    # Use pre-computed status or check data availability
    if status is None:
        status = rag_engine.check_data_availability(collection_filter)
    
    # Create status table
    table = Table(title="RAG Pipeline Status", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    # LLM Connection
    llm_status = "✅ Ready" if rag_engine.llm else "❌ Not initialized"
    llm_details = f"{rag_engine.llm_provider} - {rag_engine.llm_model_name}" if rag_engine.llm else ""
    table.add_row("LLM Connection", llm_status, llm_details)
    
    # Parsed Articles
    articles_status = "✅ Available" if status["parsed_articles_available"] else "❌ Missing"
    articles_details = f"{len(status['collections_found'])} collections, {status['total_articles']} articles"
    if status["collections_found"]:
        articles_details += f" ({', '.join(status['collections_found']).upper()})"
    table.add_row("Parsed Articles", articles_status, articles_details)
    
    # Vector Index (check separately from data availability)
    index_check = rag_engine.check_index_availability()
    index_status = "✅ Ready" if index_check["index_exists"] else "⏳ Will be created"
    if index_check["index_exists"]:
        index_details = f"Existing index found ({index_check['total_documents']} documents)"
    else:
        index_details = "Auto-indexing on first use"
    table.add_row("Vector Index", index_status, index_details)
    
    # Collection Filter
    filter_details = f"Collection: {collection_filter.upper() if collection_filter and collection_filter != 'all' else 'ALL'}"
    filter_details += f", Model: {rag_engine.llm_model_name}"
    table.add_row("Configuration", "ℹ️  Active", filter_details)
    
    console.print(table)
    
    # Show recommendations if any
    if status["recommendations"]:
        console.print("\n[yellow]📋 Recommendations:[/yellow]")
        for rec in status["recommendations"]:
            console.print(f"  • {rec}")
    
    return status


def ensure_vector_index_cli(
    rag_engine: RAGEngine, 
    console: Console, 
    collection_filter: Optional[str] = None,
    status: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Ensure that articles are indexed in the vector database using CLI interactions.
    
    Args:
        rag_engine: RAGEngine instance
        console: Console for user interaction and output
        collection_filter: Optional collection filter for indexing
        status: Optional pre-computed status data to avoid redundant checks
        
    Returns:
        True if indexing is successful or already exists, False otherwise
    """
    # Use pre-computed status or check data availability
    if status is None:
        status = rag_engine.check_data_availability(collection_filter)

    # Check if we have parsed articles to index
    if not status["parsed_articles_available"]:
        console.print("[red]❌ No parsed articles found. Please parse collections first.[/red]")
        return False
    
    # Ask user if they want to index
    console.print(f"\n[yellow]📚 Found {status['total_articles']} articles to index for semantic search[/yellow]")
    should_index = Confirm.ask("Would you like to index these articles for semantic search?", default=True)
    
    if not should_index:
        console.print("[yellow]⚠️  Skipping indexing. Search will have limited functionality.[/yellow]")
        return False
    
    # Perform indexing using the business logic method
    console.print("[yellow]🔄 Indexing articles for semantic search...[/yellow]")
    
    # Use the collection filter
    collection_param = (
        collection_filter 
        if collection_filter and collection_filter != 'all'
        else None
    )
    
    # Use the streamlined create_vector_index method
    index_result = rag_engine.create_vector_index(
        collection=collection_param,
        language='en'  # Default to English for indexing
    )
    
    if index_result["success"]:
        stats = index_result["stats"]
        if stats.get("already_exists", False):
            doc_count = stats.get("total_documents", 0)
            console.print(f"[green]✅ Vector index already exists ({doc_count} documents)[/green]")
        else:
            articles_processed = stats.get("articles_processed", 0)
            total_chunks = stats.get("total_chunks", 0)
            console.print(f"[green]✅ Successfully indexed {articles_processed} articles into {total_chunks} chunks[/green]")
        
        console.print("[green]🔍 Semantic search is now ready![/green]")
        return True
    else:
        error_msg = index_result.get("error", "Unknown error")
        console.print(f"[red]❌ Indexing failed: {error_msg}[/red]")
        console.print("[yellow]⚠️  Continuing without vector index. Search functionality will be limited.[/yellow]")
        return False


def run_rag_pipeline(
    corpus: CDCCorpus, 
    collection_filter: Optional[str], 
    console: Console
) -> None:
    """Run the complete RAG pipeline with optimized status checking."""
    try:
        # Initialize components
        rag_engine = RAGEngine(corpus)
        
        # Check vector index availability for Q&A
        console.print("[yellow]🔄 Checking vector index status...[/yellow]")
        index_status = rag_engine.check_index_availability()
        
        # If index doesn't exist, try to create it
        if not index_status["index_exists"]:
            if not ensure_vector_index_cli(rag_engine, console, collection_filter):
                console.print("\n[red]❌ Cannot proceed without vector index.[/red]")
                return
        console.print("[green]✅ Vector index status successful[/green]")

        # Test LLM connection
        console.print("\n[yellow]🔄 Testing LLM connection...[/yellow]")
        test_result = rag_engine.test_llm_connection()
        
        if not test_result['success']:
            console.print(f"[red]❌ LLM connection failed: {test_result['error']}[/red]")
            return
        
        console.print("[green]✅ LLM connection successful[/green]")
        # Start interactive Q&A using generate_answer
        run_interactive_qa_loop(rag_engine, collection_filter, console)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 RAG pipeline interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ RAG pipeline error: {e}[/red]")
        raise


@app.command()
def run() -> None:
    """Interactive setup and launch of CDC Text Corpora system.
    
    This command provides a user-friendly interface that guides you through:
    1. Collection download
    2. Language selection  
    3. Data parsing and indexing
    4. Launch of agentic Q&A system
    """
    try:
        # Step 1: Welcome and introduction
        display_welcome_intro()
        
        if not Confirm.ask("\n[bold green]Ready to get started?[/bold green]", default=True):
            console.print("[yellow]👋 Come back anytime![/yellow]")
            raise typer.Exit(0)
        
        # Step 2: Collection selection and download
        selected_collection = interactive_collection_selection()
        
        console.print(f"\n[bold blue]Step 2: Downloading {selected_collection.upper()} collection...[/bold blue]")
        console.print("[dim]This may take a few minutes depending on your internet connection...[/dim]")
        
        try:
            download_collection(selected_collection)
            console.print(f"[green]✅ Successfully downloaded {selected_collection.upper()} collection![/green]")
        except Exception as e:
            console.print(f"[red]❌ Download failed: {e}[/red]")
            if not Confirm.ask("[yellow]Continue anyway? (You may have existing data)[/yellow]", default=False):
                raise typer.Exit(1)
        
        # Step 3: Language selection
        # For simplicity, we'll offer the most common languages
        available_languages = ["en", "es", "fr"] 
        selected_language = interactive_language_selection(available_languages)
        
        # Step 4: Parsing
        console.print(f"\n[bold blue]Step 4: Parsing {selected_collection.upper()} articles ({selected_language})...[/bold blue]")
        console.print("[dim]Converting HTML articles to structured data...[/dim]")
        
        try:
            corpus = CDCCorpus()
            
            # Check for existing parsed files before starting
            collection_filter = None if selected_collection == "all" else selected_collection
            rag_engine = RAGEngine(corpus)
            status = rag_engine.check_data_availability(collection_filter or 'all')
            
            # If parsed files already exist, ask user if they want to re-parse
            skip_parsing = False
            if status["parsed_articles_available"]:
                console.print(f"\n[yellow]📄 Found existing parsed files with {status['total_articles']} articles[/yellow]")
                collections_found = ", ".join([c.upper() for c in status["collections_found"]])
                console.print(f"[dim]Collections: {collections_found}[/dim]")
                
                if not Confirm.ask("Re-parse and overwrite existing files?", default=False):
                    console.print("[green]✅ Using existing parsed files[/green]")
                    skip_parsing = True
                    total_parsed = status["total_articles"]
            
            # Only proceed with parsing if user wants to overwrite or no files exist
            if not skip_parsing:
                language_param = None if selected_language == "all" else selected_language
                collections_to_process = ['pcd', 'eid', 'mmwr'] if selected_collection == 'all' else [selected_collection]
                total_parsed = 0
                
                for coll in collections_to_process:
                    if corpus.is_collection_downloaded(coll):
                        result = corpus.load_parse_save_html_articles(
                            collection=coll,
                            language=language_param,
                            save_json=True,
                            validate_articles=True
                        )
                        
                        if 'error' not in result['stats']:
                            parsed_count = result['stats']['parsed_count']
                            total_parsed += parsed_count
                            console.print(f"[green]✅ Parsed {parsed_count} {coll.upper()} articles[/green]")
                        else:
                            console.print(f"[yellow]⚠️  Parsing issues with {coll.upper()}: {result['stats']['error']}[/yellow]")
                    else:
                        console.print(f"[yellow]⚠️  {coll.upper()} collection not found, skipping...[/yellow]")
            
            console.print(f"[green]✅ Parsing complete! Total articles: {total_parsed}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Parsing failed: {e}[/red]")
            if not Confirm.ask("[yellow]Continue to indexing anyway?[/yellow]", default=False):
                raise typer.Exit(1)
        
        # Step 5: Indexing
        console.print(f"\n[bold blue]Step 5: Creating vector search index...[/bold blue]")
        console.print("[dim]This creates embeddings for semantic search from parsed JSON files...[/dim]")
        
        try:
            # Create RAGEngine for indexing (reuse existing one if already created)
            if 'rag_engine' not in locals():
                corpus = CDCCorpus()
                collection_filter = None if selected_collection == "all" else selected_collection
                rag_engine = RAGEngine(corpus)
            
            # Use CLI helper function for indexing (handles all indexing logic)
            indexing_success = ensure_vector_index_cli(rag_engine, console, collection_filter or 'all')
            
            if not indexing_success:
                console.print("[yellow]You can still use the system, but search may be limited[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Indexing setup failed: {e}[/red]")
            console.print("[yellow]You can still use the system, but search may be limited[/yellow]")
        
        # Step 6: Launch Q&A system
        console.print(f"\n[bold blue]Step 6: Launching Q&A System...[/bold blue]")
        console.print("[dim]Starting RAG-powered research assistant...[/dim]")
        
        if Confirm.ask("\n[bold green]🚀 Launch Q&A system now?[/bold green]", default=True):
            try:
                # Initialize RAG system
                corpus = CDCCorpus()
                
                collection_filter = None if selected_collection == "all" else selected_collection
                
                console.print("\n[bold green]🤖 Welcome to CDC Text Corpora Q&A![/bold green]")
                console.print("[dim]Ask questions about CDC research and get comprehensive, cited answers...[/dim]")
                
                # Start the interactive Q&A loop using simplified approach
                run_rag_pipeline(corpus, collection_filter, console)
                
            except Exception as e:
                console.print(f"[red]❌ Failed to launch Q&A system: {e}[/red]")
                console.print("[yellow]You can try running: cdc-corpus qa[/yellow]")
                raise typer.Exit(1)
        else:
            console.print("\n[green]✅ Setup complete![/green]")
            console.print("[blue]To start Q&A later, run: [bold]cdc-corpus qa[/bold][/blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Setup interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
        console.print("[dim]For help, run: cdc-corpus --help[/dim]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information")
) -> None:
    """CDC Text Corpora: Interactive access to PCD, EID, and MMWR collections.
    
    Run without arguments for interactive setup, or use specific commands for direct access.
    """
    if version:
        console.print("CDC Text Corpora v1.0.0")
        raise typer.Exit()
    
    # If no subcommand is provided, run the interactive setup
    if ctx.invoked_subcommand is None:
        # Call the run command directly
        ctx.invoke(run)


if __name__ == "__main__":
    app()
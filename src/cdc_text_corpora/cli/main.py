"""Main CLI entry point for cdc-text-corpora."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import pathlib
import asyncio
from cdc_text_corpora.core.downloader import download_collection
from cdc_text_corpora.core.datasets import CDCCorpus
from cdc_text_corpora.utils.config import get_data_directory
from cdc_text_corpora.qa.rag_engine import RAGEngine
from cdc_text_corpora.qa.rag_pipeline import RAGPipeline
from cdc_text_corpora.qa.rag_agent import AgenticRAG, AgentConfig

###TODO

## FIX RAG
# Fix progress for vector database operations using tqdm
# Make rag sequential and improve data loading
# Stop agent early if there's no data
# Improve speed of vector operations
# BOTH RAG should work end-to-end

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
        "sequential",
        "--mode",
        "-m",
        help="RAG mode: sequential (traditional) or agentic (multi-agent)"
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
    
    Choose between sequential mode (traditional Q&A) or agentic mode (multi-agent research).
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
    valid_modes = ["sequential", "agentic"]
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
        
        # Create and run appropriate mode
        if mode.lower() == "sequential":
            # Use traditional RAGPipeline
            pipeline = RAGPipeline(
                data_dir=data_dir,
                collection_filter=collection_filter,
                language=language_param
            )
            pipeline.run()
            
        elif mode.lower() == "agentic":
            
            # Initialize corpus and check data availability
            corpus = CDCCorpus(data_dir=data_dir)
            
            # Create agent configuration
            config = AgentConfig(
                collection_filter=collection_filter or 'all',
                relevance_cutoff=8,
                search_k=10,
                max_evidence_pieces=5,
                max_search_attempts=3
            )
            
            # Initialize AgenticRAG
            agentic_rag = AgenticRAG(corpus=corpus, config=config)
            
            # Start interactive loop
            agentic_rag.run(console)
        
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
            config = AgentConfig(collection_filter=collection_filter or 'all')
            agentic_rag = AgenticRAG(corpus=corpus, config=config)
            status = agentic_rag.check_data_availability()
            
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
            # Create AgenticRAG config for indexing (reuse existing one if already created)
            if 'agentic_rag' not in locals():
                corpus = CDCCorpus()
                collection_filter = None if selected_collection == "all" else selected_collection
                config = AgentConfig(collection_filter=collection_filter or 'all')
                agentic_rag = AgenticRAG(corpus=corpus, config=config)
            
            # Use existing ensure_vector_index method (handles all indexing logic)
            indexing_success = agentic_rag.ensure_vector_index(console)
            
            if not indexing_success:
                console.print("[yellow]You can still use the system, but search may be limited[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Indexing setup failed: {e}[/red]")
            console.print("[yellow]You can still use the system, but search may be limited[/yellow]")
        
        # Step 6: Launch agentic Q&A
        console.print(f"\n[bold blue]Step 6: Launching Agentic Q&A System...[/bold blue]")
        console.print("[dim]Starting multi-agent research assistant...[/dim]")
        
        if Confirm.ask("\n[bold green]🚀 Launch Q&A system now?[/bold green]", default=True):
            try:
                # Initialize AgenticRAG system
                corpus = CDCCorpus()
                
                collection_filter = None if selected_collection == "all" else selected_collection
                config = AgentConfig(
                    collection_filter=collection_filter or 'all',
                    relevance_cutoff=8,
                    search_k=10,
                    max_evidence_pieces=5,
                    max_search_attempts=3
                )
                
                agentic_rag = AgenticRAG(corpus=corpus, config=config)
                
                console.print("\n[bold green]🤖 Welcome to CDC Text Corpora Agentic Q&A![/bold green]")
                console.print("[dim]Ask questions about CDC research and get comprehensive, cited answers...[/dim]")
                
                # Start the interactive Q&A loop
                agentic_rag.run(console)
                
            except Exception as e:
                console.print(f"[red]❌ Failed to launch Q&A system: {e}[/red]")
                console.print("[yellow]You can try running: cdc-corpus qa --mode agentic[/yellow]")
                raise typer.Exit(1)
        else:
            console.print("\n[green]✅ Setup complete![/green]")
            console.print("[blue]To start Q&A later, run: [bold]cdc-corpus qa --mode agentic[/bold][/blue]")
        
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
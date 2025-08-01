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
from cdc_text_corpora.index import ArticleIndexer, IndexConfig


app = typer.Typer(
    name="cdc-corpus",
    help="CDC Text Corpora: Access to PCD, EID, and MMWR collections with semantic search and QA",
)

console = Console()


def _run_agentic_interactive_loop(agentic_rag, console: Console) -> None:
    """Run interactive loop for agentic RAG mode."""
    console.print(Panel(
        "[bold green]🎯 Interactive Agentic Q&A Session[/bold green]\n"
        "[dim]Ask research questions. The agents will search, gather evidence, and provide comprehensive answers.\n"
        "Type 'quit', 'exit', or 'q' to stop.[/dim]",
        title="Agentic Q&A Mode", 
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
            console.print("[yellow]🔄 Processing with multi-agent system...[/yellow]")
            
            # Run async agentic RAG
            try:
                answer = asyncio.run(agentic_rag.ask_question(question, max_turns=10))
                
                # Display answer
                console.print(Panel(
                    answer,
                    title="🤖 Agentic Answer",
                    border_style="blue"
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
    
    console.print(f"\n[green]👋 Agentic Q&A session ended. Answered {question_count} questions.[/green]")


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
            # Use AgenticRAG with interactive wrapper
            console.print(Panel(
                "[bold green]🤖 Agentic RAG Mode[/bold green]\n"
                "[dim]Multi-agent system for advanced research question answering[/dim]",
                title="Agentic Mode",
                border_style="green"
            ))
            
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
            _run_agentic_interactive_loop(agentic_rag, console)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Q&A session interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

@app.command()
def index(
    collection: str = typer.Option(
        "all",
        "--collection",
        "-c", 
        help="Collection to index: pcd, eid, mmwr, or all"
    ),
    language: str = typer.Option(
        "all",
        "--language",
        "-l",
        help="Language filter: en, es, fr, zhs, zht, or all"
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        "-b",
        help="Number of files to index in each batch"
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        help="Size of text chunks for embedding"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip files already indexed in vector store"
    ),
    clear_existing: bool = typer.Option(
        False,
        "--clear-existing",
        help="Clear existing vector store before indexing"
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
    """Index HTML articles directly to vector store without intermediate JSON files."""
    
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
    
    try:
        # Display indexing info
        console.print(Panel(
            f"[bold blue]HTML to Vector Store Indexing[/bold blue]\n"
            f"Collection: {collection}\n"
            f"Language: {language}\n"
            f"Batch size: {batch_size}\n"
            f"Chunk size: {chunk_size}\n" 
            f"Skip existing: {'Yes' if skip_existing else 'No'}\n"
            f"Clear existing: {'Yes' if clear_existing else 'No'}",
            title="Article Indexing Configuration",
            border_style="blue"
        ))
        
        # Create indexing configuration
        config = IndexConfig(
            batch_size=batch_size,
            chunk_size=chunk_size,
            skip_existing=skip_existing,
            progress_bar=True,
            validate_articles=True
        )
        
        # Initialize article indexer
        indexer = ArticleIndexer(config=config, data_dir=data_dir)
        
        # Clear existing vector store if requested
        if clear_existing:
            console.print("[yellow]🗑️  Clearing existing vector store...[/yellow]")
            if indexer.clear_vectorstore():
                console.print("[green]✅ Vector store cleared successfully[/green]")
            else:
                console.print("[red]❌ Failed to clear vector store[/red]")
                raise typer.Exit(1)
        
        # Determine collections to process
        collections_to_process = ['pcd', 'eid', 'mmwr'] if collection.lower() == 'all' else [collection.lower()]
        
        total_processed = 0
        total_chunks = 0
        
        for coll in collections_to_process:
            console.print(f"\n[bold cyan]🔄 Indexing {coll.upper()} collection...[/bold cyan]")
            
            # Index collection
            language_param = None if language.lower() == 'all' else language.lower()
            stats = indexer.index_collection(coll, language_param)
            
            # Display results
            if stats.errors:
                console.print(f"[yellow]⚠️  {len(stats.errors)} errors occurred during indexing[/yellow]")
                if verbose:
                    for error in stats.errors[:5]:  # Show first 5 errors
                        console.print(f"[dim red]  • {error}[/dim red]")
                    if len(stats.errors) > 5:
                        console.print(f"[dim]  ... and {len(stats.errors) - 5} more errors[/dim]")
            
            console.print(f"[green]✅ {coll.upper()} indexing complete![/green]")
            console.print(f"[green]  • Processed: {stats.processed_files} files[/green]")
            console.print(f"[green]  • Indexed: {stats.total_chunks} chunks[/green]")
            console.print(f"[green]  • Indexing time: {stats.processing_time:.1f}s[/green]")
            
            if stats.skipped_files > 0:
                console.print(f"[yellow]  • Skipped: {stats.skipped_files} files (already indexed)[/yellow]")
            if stats.failed_files > 0:
                console.print(f"[red]  • Failed: {stats.failed_files} files[/red]")
            
            total_processed += stats.processed_files
            total_chunks += stats.total_chunks
        
        # Final summary
        console.print(f"\n[bold green]🎉 Article indexing complete![/bold green]")
        console.print(f"[green]Total files processed: {total_processed}[/green]")
        console.print(f"[green]Total chunks indexed: {total_chunks}[/green]")
        
        # Show vector store stats
        vectorstore_stats = indexer.get_vectorstore_stats()
        if "error" not in vectorstore_stats:
            console.print(f"[green]Vector store contains: {vectorstore_stats['total_documents']} documents[/green]")
        
        if verbose:
            console.print(f"[dim]Vector store location: {indexer.config.persist_directory}[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def search() -> None:
    """Search CDC text corpora collections."""
    print(f"RAG code")

def main() -> None:
    """Main CLI entry point."""
    app()

if __name__ == "__main__":
    main()